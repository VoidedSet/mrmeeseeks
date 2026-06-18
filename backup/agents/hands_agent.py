"""
hands_agent.py — Mr Meeseeks Hands Agent
Mouse and keyboard control via pyautogui + xdotool.

Tools:
  click_at(x, y, btn="left")  — move to coords and click (atomic, replaces move_mouse + click)
  type_text(text)              — type a string
  key_press(keys)              — press key combo (xdotool)
  scroll(direction, amount)    — scroll at current position
"""
import subprocess
import logging

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
except ImportError:
    pyautogui = None

from core.ipc_bus import bus

log = logging.getLogger("hands_agent")


async def handle_click_at(args: dict) -> dict:
    """
    Move to (x, y) and click. btn defaults to 'left'.
    Accepts x/y as int OR string (coerced).
    Also handles common model output variants: x_to, x_pos, xcoord → x, etc.
    """
    if not pyautogui:
        return {"error": "pyautogui not installed. Run: pip install pyautogui"}

    # ── Normalize common model fumbles ───────────────────────────────────────
    # Some small models output x_to/y_to or x_pos/y_pos instead of x/y
    _x_aliases = ("x", "x_to", "x_pos", "xcoord", "coord_x", "target_x")
    _y_aliases = ("y", "y_to", "y_pos", "ycoord", "coord_y", "target_y")
    _btn_aliases = ("btn", "button", "mouse_btn", "mouse_button")

    x = None
    for k in _x_aliases:
        if k in args:
            x = args[k]
            break

    y = None
    for k in _y_aliases:
        if k in args:
            y = args[k]
            break

    btn = "left"
    for k in _btn_aliases:
        if k in args:
            btn = str(args[k]).lower()
            break

    # Also handle nested coords dict: {"coords": {"x": 41, "y": 20}}
    if (x is None or y is None) and isinstance(args.get("coords"), dict):
        coords = args["coords"]
        if x is None:
            x = coords.get("x")
        if y is None:
            y = coords.get("y")

    if x is None or y is None:
        return {"error": "click_at requires 'x' and 'y' coordinates. "
                         "Get them from get_ui_elements first."}

    try:
        x = int(float(str(x)))
        y = int(float(str(y)))
    except (ValueError, TypeError):
        return {"error": f"x and y must be integers, got x={x!r} y={y!r}"}

    if btn not in ("left", "right", "middle"):
        btn = "left"

    try:
        pyautogui.moveTo(x, y, duration=0.15)
        pyautogui.click(button=btn)
        log.info(f"click_at({x}, {y}, {btn})")
        return {"status": f"Clicked {btn} at ({x}, {y})"}
    except Exception as e:
        return {"error": str(e)}


async def handle_double_click_at(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui not installed."}
    x = args.get("x")
    y = args.get("y")
    if x is None or y is None:
        return {"error": "double_click_at requires 'x' and 'y'"}
    try:
        pyautogui.moveTo(int(x), int(y), duration=0.15)
        pyautogui.doubleClick()
        return {"status": f"Double-clicked at ({x}, {y})"}
    except Exception as e:
        return {"error": str(e)}


async def handle_type_text(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui not installed."}
    text = args.get("text", "")
    if not text:
        return {"error": "Missing 'text' argument"}
    try:
        # Use xdotool type for better Unicode/special char support
        result = subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "20", text],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            return {"status": f"Typed: {text[:40]}{'...' if len(text) > 40 else ''}"}
        # Fallback to pyautogui for ASCII
        pyautogui.write(text, interval=0.02)
        return {"status": f"Typed (fallback): {text[:40]}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_key_press(args: dict) -> dict:
    keys = args.get("keys", "")
    if not keys:
        return {"error": "Missing 'keys' argument. Example: 'ctrl+c', 'Return', 'ctrl+alt+t'"}
    try:
        result = subprocess.run(
            ["xdotool", "key", "--clearmodifiers", keys],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return {"status": f"Pressed: {keys}"}
        return {"error": f"xdotool failed: {result.stderr.strip()}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_scroll(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui not installed."}
    direction = (args.get("direction") or "down").lower()
    amount    = int(args.get("amount", 3))
    val = amount if direction == "up" else -amount
    try:
        pyautogui.scroll(val)
        return {"status": f"Scrolled {direction} by {amount}"}
    except Exception as e:
        return {"error": str(e)}


def register():
    bus.register("click_at",        handle_click_at)
    bus.register("double_click_at", handle_double_click_at)
    bus.register("type_text",       handle_type_text)
    bus.register("key_press",       handle_key_press)
    bus.register("scroll",          handle_scroll)
    log.info("Hands agent registered ✓")
