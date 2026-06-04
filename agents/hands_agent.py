"""
hands_agent.py — Mr Meeseeks Hands Agent
Uses pyautogui and xdotool to control mouse and keyboard.
"""
import subprocess
import logging

try:
    import pyautogui
    # fail-safe feature is annoying for AI agents, but keep it on for safety
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None

from core.ipc_bus import bus

log = logging.getLogger("hands_agent")

async def handle_move_mouse(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui is not installed. Run: pip install pyautogui"}
    x = args.get("x")
    y = args.get("y")
    try:
        pyautogui.moveTo(int(x), int(y), duration=0.2)
        return {"status": f"Mouse moved to {x},{y}"}
    except Exception as e:
        return {"error": str(e)}

async def handle_click(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui is not installed."}
    btn = args.get("btn", "left").lower()
    action = args.get("action", "click").lower()
    try:
        if action == "click":
            pyautogui.click(button=btn)
        elif action == "double":
            pyautogui.doubleClick(button=btn)
        elif action == "hold":
            pyautogui.mouseDown(button=btn)
        elif action == "release":
            pyautogui.mouseUp(button=btn)
        else:
            return {"error": f"Unknown click action: {action}"}
        return {"status": f"Mouse {action} ({btn}) executed."}
    except Exception as e:
        return {"error": str(e)}

async def handle_type_text(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui is not installed."}
    text = args.get("text", "")
    try:
        pyautogui.write(text, interval=0.01)
        return {"status": f"Typed text: {text[:20]}..."}
    except Exception as e:
        return {"error": str(e)}

async def handle_key_press(args: dict) -> dict:
    keys = args.get("keys", "")
    try:
        # Use xdotool for better Linux compatibility with combos like ctrl+c
        # Split e.g., ctrl+c -> ctrl+c for xdotool
        result = subprocess.run(["xdotool", "key", keys], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": f"Pressed key combo: {keys}"}
        else:
            return {"error": f"xdotool failed: {result.stderr}"}
    except Exception as e:
        return {"error": str(e)}

async def handle_scroll(args: dict) -> dict:
    if not pyautogui:
        return {"error": "pyautogui is not installed."}
    direction = args.get("direction", "down").lower()
    amount = int(args.get("amount", 10))
    # pyautogui.scroll takes positive for up, negative for down on Linux
    val = amount if direction == "up" else -amount
    try:
        pyautogui.scroll(val)
        return {"status": f"Scrolled {direction} by {amount}"}
    except Exception as e:
        return {"error": str(e)}

def register():
    bus.register("move_mouse", handle_move_mouse)
    bus.register("click", handle_click)
    bus.register("type_text", handle_type_text)
    bus.register("key_press", handle_key_press)
    bus.register("scroll", handle_scroll)
    log.info("Hands agent registered ✓")
