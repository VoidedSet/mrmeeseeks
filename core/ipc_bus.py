"""
ipc_bus.py — Mr Meeseeks IPC Message Bus
Brain dispatches tool calls here. Agents register handlers.
asyncio.Queue based — zero extra dependencies.
"""

import asyncio
import logging
from typing import Any, Callable, Awaitable

log = logging.getLogger("ipc_bus")

Handler = Callable[[dict], Awaitable[Any]]


class IPCBus:
    def __init__(self):
        self._handlers: dict[str, Handler] = {}
        self._listeners: list[Callable[[str, dict], None]] = []

    def register(self, tool_name: str, handler: Handler):
        """Agents call this at startup to register their tool handlers."""
        self._handlers[tool_name] = handler
        log.info(f"Registered handler: {tool_name}")

    def add_listener(self, callback: Callable[[str, dict], None]):
        """UI or other components register a listener to observe dispatches."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, dict], None]):
        if callback in self._listeners:
            self._listeners.remove(callback)

    async def dispatch(self, tool_name: str, args: dict) -> Any:
        """
        Brain calls this with a validated tool call.
        Routes to the correct agent handler.
        Returns the result.
        """
        # Notify listeners first (so UI can animate cursor flight/clicks)
        for cb in self._listeners:
            try:
                cb(tool_name, args)
            except Exception as e:
                log.warning(f"Error in IPCBus listener for '{tool_name}': {e}")

        handler = self._handlers.get(tool_name)

        if handler is None:
            log.warning(f"No handler registered for '{tool_name}'")
            return {"error": f"Tool '{tool_name}' not available yet. Use only registered tools: {self.registered_tools()}"}

        try:
            result = await handler(args)
            return result
        except Exception as e:
            log.error(f"Handler '{tool_name}' raised: {e}")
            return {"error": str(e)}

    def registered_tools(self) -> list[str]:
        return list(self._handlers.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────
bus = IPCBus()
