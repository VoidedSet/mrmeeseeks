"""
eyes_agent.py — Mr Meeseeks Eyes Agent
Uses AT-SPI2 to read accessibility tree on Linux desktop.

App resolution (in order):
  1. app_bridge.resolve(query) — PID-based bridge handles aliases, fuzzy names, window titles
  2. Direct AT-SPI name fallback if bridge not yet warmed

Default behavior (no args):
  - Scans top bar (gnome-shell, y < 55) + currently active app
  - This matches exactly what the user physically sees on screen

Region filter: {x1, y1, x2, y2} — elements must have center coords within box
"""
import logging
from core.ipc_bus import bus

log = logging.getLogger("eyes_agent")

try:
    import pyatspi
    HAS_PYATSPI = True
except ImportError:
    HAS_PYATSPI = False

_OFFSCREEN = -3221225472

_ROLE_CATEGORY = {
    "push button":    "buttons",
    "toggle button":  "buttons",
    "check box":      "buttons",
    "radio button":   "buttons",
    "menu item":      "menu",
    "menu":           "menu",
    "menu bar":       "menu",
    "page tab":       "tabs",
    "page tab list":  "tabs",
    "entry":          "inputs",
    "password text":  "inputs",
    "combo box":      "inputs",
    "spin box":       "inputs",
    "text":           "inputs",
    "link":           "links",
    "heading":        "content",
    "label":          "content",
    "list item":      "content",
    "tree item":      "content",
    "table cell":     "content",
    "document web":   "content",
    "document frame": "content",
    "static":         "content",
}


def _get_coords(node):
    """Returns (cx, cy) or None if off-screen/no geometry."""
    try:
        ext = node.get_extents(pyatspi.DESKTOP_COORDS)
        if (ext.x == _OFFSCREEN or ext.y == _OFFSCREEN
                or (ext.width == 0 and ext.height == 0)):
            return None
        return (ext.x + ext.width // 2, ext.y + ext.height // 2)
    except Exception:
        return None


def _is_interactive(node) -> bool:
    try:
        action = node.queryAction()
        return action is not None and action.nActions > 0
    except Exception:
        return False


def _collect_from_app(app_node, region, results, seen, cap):
    if app_node is None or len(results) >= cap:
        return
    try:
        role = app_node.getRoleName()
        name = (app_node.name or "").strip()

        if name and name != "[Unnamed]" and _is_interactive(app_node):
            coords = _get_coords(app_node)
            in_region = True
            if region and coords:
                cx, cy = coords
                in_region = (region["x1"] <= cx <= region["x2"]
                             and region["y1"] <= cy <= region["y2"])
            elif region and coords is None:
                in_region = False

            if in_region:
                key = f"{role}::{name}"
                if key not in seen:
                    seen.add(key)
                    entry = {"id": str(len(results)), "name": name, "role": role,
                             "cat": _ROLE_CATEGORY.get(role, "other")}
                    if coords:
                        entry["x"] = coords[0]
                        entry["y"] = coords[1]
                    results.append(entry)

        if len(results) < cap:
            for i in range(app_node.getChildCount()):
                _collect_from_app(app_node.getChildAtIndex(i), region, results, seen, cap)
                if len(results) >= cap:
                    break
    except Exception:
        pass


def _resolve_app(query: str):
    """
    Returns list of pyatspi app nodes matching the query.
    Uses app_bridge first, falls back to direct AT-SPI scan.
    """
    try:
        desktop = pyatspi.Registry.getDesktop(0)
    except Exception:
        return [], None

    # Try app_bridge resolution
    try:
        from kernel.app_bridge import bridge
        from kernel.kernel_state import state as kstate
        # Use live bridge if available
        if kstate.app_bridge:
            atspi_name, err = bridge.resolve(query)
            if err:
                return [], err
            if atspi_name:
                query = atspi_name  # use the resolved AT-SPI name
    except Exception:
        pass

    # Scan desktop for matching app nodes
    q = query.lower()
    matches = []
    try:
        for app in desktop:
            if app is None:
                continue
            try:
                name = (app.name or "").lower()
                if q in name or name in q:
                    matches.append(app)
            except Exception:
                continue
    except Exception:
        pass

    if not matches:
        return [], f"No app matching '{query}' found on AT-SPI bus. Try list_at_spi_apps to see registered names."

    return matches, None


def _summarize(elements) -> dict:
    by_cat: dict[str, list] = {}
    for el in elements:
        cat = el.get("cat", "other")
        by_cat.setdefault(cat, [])
        entry = {"id": el["id"], "name": el["name"], "role": el["role"]}
        if "x" in el:
            entry["x"] = el["x"]
            entry["y"] = el["y"]
        by_cat[cat].append(entry)
    return {"by_cat": by_cat, "total": len(elements)}


async def handle_get_ui_elements(args: dict) -> dict:
    """
    Get interactive UI elements on screen.

    Args (all optional):
        app    : str  — app name (fuzzy: "Firefox", "VS Code", "code", etc.)
        region : dict — {x1, y1, x2, y2} screen bounding box in pixels

    No args → auto-scope: top bar (gnome-shell) + currently active app.
    """
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed. Run: sudo apt install python3-pyatspi"}

    app_filter    = (args.get("app") or "").strip() or None
    region_filter = args.get("region") or None

    if region_filter is not None:
        if not isinstance(region_filter, dict):
            return {"error": "region must be {x1, y1, x2, y2}"}
        for k in ("x1", "y1", "x2", "y2"):
            if k not in region_filter:
                return {"error": f"region missing key '{k}'"}

    try:
        desktop = pyatspi.Registry.getDesktop(0)
    except Exception as e:
        return {"error": f"Cannot connect to AT-SPI: {e}"}

    results:  list = []
    seen:     set  = set()
    CAP = 40

    # ── Mode A: specific app requested ───────────────────────────────────────
    if app_filter:
        apps, err = _resolve_app(app_filter)
        if err:
            return {"error": err, "tip": "Enable accessibility: gsettings set org.gnome.desktop.interface toolkit-accessibility true"}
        for app in apps:
            _collect_from_app(app, region_filter, results, seen, CAP)
            if len(results) >= CAP:
                break
        summary = _summarize(results)
        summary["scanned_apps"] = [(a.name or "?") for a in apps]

    # ── Mode B: region only ───────────────────────────────────────────────────
    elif region_filter:
        try:
            for app in desktop:
                if app is None:
                    continue
                _collect_from_app(app, region_filter, results, seen, CAP)
                if len(results) >= CAP:
                    break
        except Exception:
            pass
        summary = _summarize(results)
        summary["region"] = region_filter

    # ── Mode C: no args → top bar + active app ───────────────────────────────
    else:
        # 1. Always include top bar (gnome-shell)
        top_bar_region = {"x1": 0, "y1": 0, "x2": 9999, "y2": 55}
        try:
            for app in desktop:
                if app and (app.name or "").lower() == "gnome-shell":
                    _collect_from_app(app, top_bar_region, results, seen, 15)
                    break
        except Exception:
            pass

        # 2. Scan active app via kernel state + bridge
        active_atspi = None
        try:
            from kernel.kernel_state import state as kstate
            from kernel.app_bridge import bridge
            if kstate.active_window and kstate.active_window != "unknown":
                active_atspi = bridge.resolve_active_window(kstate.active_window)
        except Exception:
            pass

        if active_atspi:
            q = active_atspi.lower()
            try:
                for app in desktop:
                    if app and (app.name or "").lower() == q:
                        _collect_from_app(app, None, results, seen, CAP)
                        break
            except Exception:
                pass

        summary = _summarize(results)
        summary["note"] = "Auto-scoped: top bar + active window"
        if active_atspi:
            summary["active_app"] = active_atspi

    if region_filter:
        summary["region"] = region_filter
    return summary


async def handle_find_element_by_label(args: dict) -> dict:
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}
    label = (args.get("label") or "").lower().strip()
    if not label:
        return {"error": "Missing 'label' argument"}
    app_filter = (args.get("app") or "").strip() or None
    try:
        if app_filter:
            apps, err = _resolve_app(app_filter)
            if err:
                return {"error": err}
        else:
            desktop = pyatspi.Registry.getDesktop(0)
            apps = [a for a in desktop if a]
        results: list = []
        seen: set = set()
        for app in apps:
            _collect_from_app(app, None, results, seen, 200)
        matches = [e for e in results if label in e.get("name", "").lower()]
        return {"matches": matches[:10]}
    except Exception as e:
        return {"error": str(e)}


async def handle_read_element_text(args: dict) -> dict:
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}
    el_id = str(args.get("id", "")).strip()
    if not el_id:
        return {"error": "Missing 'id' argument"}
    app_filter = (args.get("app") or "").strip() or None
    try:
        if app_filter:
            apps, err = _resolve_app(app_filter)
            if err:
                return {"error": err}
        else:
            desktop = pyatspi.Registry.getDesktop(0)
            apps = [a for a in desktop if a]
        results: list = []
        seen: set = set()
        for app in apps:
            _collect_from_app(app, None, results, seen, 200)
        match = next((e for e in results if e["id"] == el_id), None)
        return {"element": match} if match else {"error": f"No element id={el_id}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_list_at_spi_apps(args: dict) -> dict:
    """List all apps on the AT-SPI bus. Use to find process names for get_ui_elements."""
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}
    try:
        desktop = pyatspi.Registry.getDesktop(0)
        apps = []
        for app in desktop:
            if app is None:
                continue
            try:
                apps.append({"name": app.name or "(unnamed)", "child_count": app.childCount})
            except Exception:
                continue
        return {"apps": apps}
    except Exception as e:
        return {"error": str(e)}


def register():
    bus.register("get_ui_elements",       handle_get_ui_elements)
    bus.register("read_element_text",     handle_read_element_text)
    bus.register("find_element_by_label", handle_find_element_by_label)
    bus.register("list_at_spi_apps",      handle_list_at_spi_apps)
    log.info("Eyes agent registered ✓")
