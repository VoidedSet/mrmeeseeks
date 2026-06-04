"""
eyes_agent.py — Mr Meeseeks Eyes Agent
Uses AT-SPI2 to read accessibility tree on Linux desktop.

Supports targeted queries to avoid context explosion:
  - No args      → all visible interactive elements, grouped by app
  - app="Firefox"→ only Firefox's elements
  - region={x1,y1,x2,y2} → only elements whose center falls in that screen region

Output is always summarized before reaching the model to fit in 4k context.
"""
import logging
from core.ipc_bus import bus

log = logging.getLogger("eyes_agent")

try:
    import pyatspi
    HAS_PYATSPI = True
except ImportError:
    HAS_PYATSPI = False


# Structural noise — never useful for the model
_SKIP_ROLES = {
    "panel", "filler", "unknown", "separator", "scroll bar",
    "page tab list", "layered pane", "glass pane", "root pane",
    "split pane", "internal frame", "desktop icon", "desktop pane",
}

# Only these roles are worth reporting to the model
_INTERESTING_ROLES = {
    "push button", "toggle button", "check box", "radio button",
    "text", "entry", "password text", "combo box", "spin box",
    "list item", "menu item", "menu", "menu bar",
    "link", "image", "label", "heading", "tab", "tree item",
    "document web", "document frame", "status bar",
    "table cell", "column header",
}

# pyatspi off-screen sentinel value
_OFFSCREEN = -3221225472


def _collect_elements(node, results: list, app_name: str = ""):
    """Walk AT-SPI tree, collecting all named elements with valid coords."""
    try:
        if node is None:
            return

        role = node.getRoleName()
        name = (node.name or "").strip()

        if role == "application":
            app_name = name or app_name

        if name and role not in _SKIP_ROLES:
            try:
                ext = node.get_extents(pyatspi.DESKTOP_COORDS)
                if (ext.x != _OFFSCREEN and ext.y != _OFFSCREEN
                        and (ext.width > 0 or ext.height > 0)):
                    cx = ext.x + ext.width  // 2
                    cy = ext.y + ext.height // 2
                    results.append({
                        "app":  app_name,
                        "id":   str(len(results)),
                        "name": name,
                        "role": role,
                        "x":    cx,
                        "y":    cy,
                    })
            except Exception:
                pass

        for i in range(node.getChildCount()):
            _collect_elements(node.getChildAtIndex(i), results, app_name)

    except Exception:
        pass


def _apply_filters(elements: list, app: str | None, region: dict | None) -> list:
    """
    Apply app and/or region filters to the raw element list.

    app    : case-insensitive substring match on element['app']
    region : {'x1': int, 'y1': int, 'x2': int, 'y2': int} — bounding box
    """
    filtered = elements

    if app:
        app_lower = app.lower()
        filtered = [e for e in filtered if app_lower in e.get("app", "").lower()]

    if region:
        x1 = region.get("x1", 0)
        y1 = region.get("y1", 0)
        x2 = region.get("x2", 9999)
        y2 = region.get("y2", 9999)
        filtered = [
            e for e in filtered
            if x1 <= e["x"] <= x2 and y1 <= e["y"] <= y2
        ]

    return filtered


def _summarize(elements: list) -> dict:
    """
    Filter to interesting roles and group by app for model readability.
    Returns compact dict: {by_app: {app: [{id, name, role, x, y}]}, total: int}
    """
    interesting = [e for e in elements if e["role"] in _INTERESTING_ROLES]

    by_app: dict[str, list] = {}
    for el in interesting:
        app = el.get("app") or "unknown"
        by_app.setdefault(app, [])
        by_app[app].append({
            "id":   el["id"],
            "name": el["name"],
            "role": el["role"],
            "x":    el["x"],
            "y":    el["y"],
        })

    return {"by_app": by_app, "total": len(interesting)}


async def handle_get_ui_elements(args: dict) -> dict:
    """
    Get visible UI elements on screen.

    Args (all optional):
        app    : str  — filter to a specific application (e.g. "Firefox")
        region : dict — {x1, y1, x2, y2} screen bounding box in pixels
                        e.g. {"x1": 0, "y1": 0, "x2": 1920, "y2": 50} for top bar

    Returns:
        {"by_app": {app_name: [{id, name, role, x, y}]}, "total": int}
    """
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed. Run: sudo apt install python3-pyatspi"}

    app_filter    = args.get("app")    or None
    region_filter = args.get("region") or None

    # Validate region if provided
    if region_filter is not None:
        if not isinstance(region_filter, dict):
            return {"error": "region must be an object: {x1, y1, x2, y2}"}
        for key in ("x1", "y1", "x2", "y2"):
            if key not in region_filter:
                return {"error": f"region missing key '{key}'"}

    try:
        desktop = pyatspi.Registry.getDesktop(0)
        raw = []
        _collect_elements(desktop, raw)

        # Cap raw collection before filtering
        raw = raw[:300]

        # Apply user filters
        filtered = _apply_filters(raw, app_filter, region_filter)

        # Summarize for model
        result = _summarize(filtered)

        # Add filter info so model knows what it's looking at
        result["filtered_by"] = {}
        if app_filter:
            result["filtered_by"]["app"] = app_filter
        if region_filter:
            result["filtered_by"]["region"] = region_filter

        return result

    except Exception as e:
        return {"error": f"AT-SPI error: {e}"}


async def handle_read_element_text(args: dict) -> dict:
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}

    el_id = str(args.get("id", ""))
    if not el_id:
        return {"error": "Missing 'id' argument"}

    try:
        desktop = pyatspi.Registry.getDesktop(0)
        raw = []
        _collect_elements(desktop, raw)
        match = next((e for e in raw if e["id"] == el_id), None)
        if match:
            return {"element": match}
        return {"error": f"No element with id={el_id}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_find_element_by_label(args: dict) -> dict:
    if not HAS_PYATSPI:
        return {"error": "pyatspi not installed."}

    label = args.get("label", "").lower()
    if not label:
        return {"error": "Missing 'label' argument"}

    try:
        desktop = pyatspi.Registry.getDesktop(0)
        raw = []
        _collect_elements(desktop, raw)
        matches = [e for e in raw if label in e.get("name", "").lower()]
        return {"matches": matches[:10]}
    except Exception as e:
        return {"error": str(e)}


def register():
    bus.register("get_ui_elements",       handle_get_ui_elements)
    bus.register("read_element_text",     handle_read_element_text)
    bus.register("find_element_by_label", handle_find_element_by_label)
    log.info("Eyes agent registered ✓")
