"""
eyes_agent.py — Mr Meeseeks Eyes Agent
Uses AT-SPI2 to read accessibility tree on Linux desktop.

Collection strategy:
  - Walks tree using queryAction() to find TRULY interactive elements (like your test script)
  - Matches apps by process name (e.g. "firefox", "code") not display title
  - Filters to named elements only, deduplicates by name+role
  - Region filter applied on coords (only for elements that have valid coords)
  - Hard cap of 40 elements before sending to model, grouped by role category

Known AT-SPI app name mappings:
  Firefox       → "Firefox" (Wayland) or "firefox" or "Mozilla Firefox"
  VS Code       → "code"
  GNOME Shell   → "gnome-shell"
  Terminal      → "gnome-terminal-server" or "Terminal"
"""
import logging
from core.ipc_bus import bus

log = logging.getLogger("eyes_agent")

try:
    import pyatspi
    HAS_PYATSPI = True
except ImportError:
    HAS_PYATSPI = False

# Off-screen sentinel value from pyatspi
_OFFSCREEN = -3221225472

# Role categories for grouping output
_ROLE_CATEGORY = {
    # Controls
    "push button":    "buttons",
    "toggle button":  "buttons",
    "check box":      "buttons",
    "radio button":   "buttons",
    # Navigation
    "menu item":      "menu",
    "menu":           "menu",
    "menu bar":       "menu",
    "page tab":       "tabs",
    "page tab list":  "tabs",
    # Input
    "entry":          "inputs",
    "password text":  "inputs",
    "combo box":      "inputs",
    "spin box":       "inputs",
    "text":           "inputs",
    # Content
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


def _get_coords(node) -> tuple[int, int] | None:
    """Try to get valid center coordinates for a node. Returns None if off-screen."""
    try:
        ext = node.get_extents(pyatspi.DESKTOP_COORDS)
        if (ext.x == _OFFSCREEN or ext.y == _OFFSCREEN
                or (ext.width == 0 and ext.height == 0)):
            return None
        return (ext.x + ext.width // 2, ext.y + ext.height // 2)
    except Exception:
        return None


def _is_interactive(node) -> bool:
    """Check if a node has at least one action via queryAction()."""
    try:
        action = node.queryAction()
        return action is not None and action.nActions > 0
    except Exception:
        return False


def _collect_from_app(app_node, region: dict | None, results: list, seen_names: set, cap: int):
    """
    Walk the subtree of a single app node.
    Collects interactive, named elements respecting region + dedup + cap.
    """
    if app_node is None or len(results) >= cap:
        return

    try:
        role = app_node.getRoleName()
        name = (app_node.name or "").strip()

        # Skip unnamed unless it's a known structural role
        if name and name != "[Unnamed]" and _is_interactive(app_node):
            coords = _get_coords(app_node)

            # Region filter — only apply if region given AND we have valid coords
            in_region = True
            if region and coords:
                cx, cy = coords
                in_region = (region["x1"] <= cx <= region["x2"]
                             and region["y1"] <= cy <= region["y2"])
            elif region and coords is None:
                # No valid coords — skip when region filter is active
                in_region = False

            if in_region:
                dedup_key = f"{role}::{name}"
                if dedup_key not in seen_names:
                    seen_names.add(dedup_key)
                    entry = {
                        "id":   str(len(results)),
                        "name": name,
                        "role": role,
                        "cat":  _ROLE_CATEGORY.get(role, "other"),
                    }
                    if coords:
                        entry["x"] = coords[0]
                        entry["y"] = coords[1]
                    results.append(entry)

        if len(results) < cap:
            for i in range(app_node.getChildCount()):
                child = app_node.getChildAtIndex(i)
                _collect_from_app(child, region, results, seen_names, cap)
                if len(results) >= cap:
                    break

    except Exception:
        pass


def _find_apps(desktop, app_query: str | None) -> list:
    """
    Find matching app nodes from the desktop.
    app_query: case-insensitive substring to match against app.name.
    Returns list of matching pyatspi app nodes.
    If app_query is None, returns the active/focused app only.
    """
    if desktop is None:
        return []

    matches = []
    try:
        for app in desktop:
            if app is None:
                continue
            try:
                name = (app.name or "").lower()
                if app_query:
                    # Fuzzy match: query is substring of app name OR vice versa
                    q = app_query.lower()
                    if q in name or name in q:
                        matches.append(app)
                else:
                    # No filter → include all apps (will be capped later)
                    matches.append(app)
            except Exception:
                continue
    except Exception:
        pass

    return matches


def _summarize(elements: list) -> dict:
    """
    Group elements by role category for compact model output.
    Returns: {by_cat: {cat: [{id, name, role, x?, y?}]}, total: int}
    """
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
    Get visible/interactive UI elements on screen.

    Args (all optional):
        app    : str  — app name to filter (fuzzy match, e.g. "firefox", "code", "VS Code")
                        If omitted, scans the currently focused app.
        region : dict — {x1, y1, x2, y2} screen bounding box in pixels
                        e.g. {"x1": 0, "y1": 0, "x2": 1920, "y2": 50} for top bar

    Returns:
        {by_cat: {category: [{id, name, role, x?, y?}]}, total: int, scanned_apps: [...]}

    Categories: buttons, menu, tabs, inputs, links, content, other
    """
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed. Run: sudo apt install python3-pyatspi"}

    app_filter    = (args.get("app") or "").strip() or None
    region_filter = args.get("region") or None

    if region_filter is not None:
        if not isinstance(region_filter, dict):
            return {"error": "region must be an object: {x1, y1, x2, y2}"}
        for k in ("x1", "y1", "x2", "y2"):
            if k not in region_filter:
                return {"error": f"region missing key '{k}'"}

    try:
        desktop = pyatspi.Registry.getDesktop(0)
    except Exception as e:
        return {"error": f"Cannot connect to AT-SPI: {e}"}

    apps = _find_apps(desktop, app_filter)
    if not apps:
        return {
            "by_cat":      {},
            "total":       0,
            "scanned_apps": [],
            "note": (
                f"No app matching '{app_filter}' found on AT-SPI bus. "
                "Common names: 'code' (VS Code), 'firefox', 'gnome-terminal-server'. "
                "The app must have accessibility enabled."
            )
        }

    results   = []
    seen_names: set[str] = set()
    CAP = 40  # max elements to return

    for app in apps:
        _collect_from_app(app, region_filter, results, seen_names, CAP)
        if len(results) >= CAP:
            break

    summary = _summarize(results)
    summary["scanned_apps"] = [(a.name or "unknown") for a in apps]
    if region_filter:
        summary["region"] = region_filter

    return summary


async def handle_find_element_by_label(args: dict) -> dict:
    """Find interactive elements whose name contains the given label (case-insensitive)."""
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}

    label = (args.get("label") or "").lower().strip()
    if not label:
        return {"error": "Missing 'label' argument"}

    app_filter = (args.get("app") or "").strip() or None

    try:
        desktop = pyatspi.Registry.getDesktop(0)
        apps = _find_apps(desktop, app_filter)
        if not apps:
            return {"matches": [], "note": f"No app matching '{app_filter}' found."}

        results: list = []
        seen_names: set[str] = set()
        # Collect all (no region filter) then search
        for app in apps:
            _collect_from_app(app, None, results, seen_names, cap=200)

        matches = [e for e in results if label in e.get("name", "").lower()]
        return {"matches": matches[:10]}
    except Exception as e:
        return {"error": str(e)}


async def handle_read_element_text(args: dict) -> dict:
    """Read a specific element by id from the last scan. Requires re-scanning."""
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}

    el_id = str(args.get("id", "")).strip()
    if not el_id:
        return {"error": "Missing 'id' argument"}

    app_filter = (args.get("app") or "").strip() or None

    try:
        desktop = pyatspi.Registry.getDesktop(0)
        apps = _find_apps(desktop, app_filter)
        if not apps:
            return {"error": "No matching app found."}

        results: list = []
        seen_names: set[str] = set()
        for app in apps:
            _collect_from_app(app, None, results, seen_names, cap=200)

        match = next((e for e in results if e["id"] == el_id), None)
        if match:
            return {"element": match}
        return {"error": f"No element with id={el_id} found."}
    except Exception as e:
        return {"error": str(e)}


async def handle_list_at_spi_apps(args: dict) -> dict:
    """
    List all apps currently registered on the AT-SPI bus.
    Useful for finding the exact app name to pass to get_ui_elements.
    """
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}
    try:
        desktop = pyatspi.Registry.getDesktop(0)
        apps = []
        for app in desktop:
            if app is None:
                continue
            try:
                apps.append({
                    "name":       app.name or "(unnamed)",
                    "child_count": app.childCount,
                })
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
