import pyatspi

def list_clickable(element):
    if element is None:
        return
    try:
        action_interface = element.queryAction()
        if action_interface and action_interface.nActions > 0:
            role = element.getRoleName().upper()
            name = element.name if element.name else "[Unnamed]"
            print(f"[{role}] -> Name: {name}")
    except NotImplementedError:
        pass

    try:
        for child in element:
            list_clickable(child)
    except Exception:
        pass

# Target apps we care about
TARGETS = ["firefox", "spotify"]

desktop = pyatspi.Registry.getDesktop(0)
for app in desktop:
    try:
        if app and app.name.lower() in TARGETS:
            print(f"\n--- CLICKABLE ITEMS IN {app.name.upper()} ---")
            list_clickable(app)
    except Exception:
        continue
