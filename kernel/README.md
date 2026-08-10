# Mr Meeseeks - Kernel Listener Subsystem

The `kernel/` module is responsible for capturing real-time events and polling OS-level states. It runs background loops that update the central `KernelState` cache, keeping it fresh for immediate brain queries with zero sub-process latency.

## Components and Processes

-   **[kernel_listener.py](kernel_listener.py)**: Spawns asynchronous tasks running concurrently to poll:
    -   *Active Window* (`xdotool`): Checks focused app title every 800ms.
    -   *Open Windows* (`wmctrl -l`): Re-lists window trees every 3 seconds.
    -   *Battery State* (`sysfs` capacity): Reads power source state every 30 seconds.
    -   *App Bridge Table*: Scans running processes every 5 seconds.
-   **[kernel_state.py](kernel_state.py)**: A thread-safe global cache storage (`KernelState` class) that provides quick snapshots of system data.
-   **[app_bridge.py](app_bridge.py)**: Maintains a mapping between running system PIDs, application binaries, and accessibility roles on the D-Bus/AT-SPI bus.

## Proactive Trigger Pattern

When polling detects a state change (such as window focus or low battery power), it calls `brain.handle_proactive_event()`. The Brain can then decide to intercept user activity and emit warning speeches or execute proactive commands (like closing resource-heavy processes).
