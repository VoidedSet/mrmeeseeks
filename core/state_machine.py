"""
state_machine.py — Mr Meeseeks State Machine
One state at a time. Hard rule.
"""

import asyncio
import logging
from enum import Enum

log = logging.getLogger("state_machine")


class State(Enum):
    IDLE      = "IDLE"
    LISTENING = "LISTENING"
    THINKING  = "THINKING"
    ACTING    = "ACTING"
    SPEAKING  = "SPEAKING"


# valid transitions
TRANSITIONS = {
    State.IDLE:      [State.LISTENING, State.THINKING],
    State.LISTENING: [State.THINKING, State.IDLE],
    State.THINKING:  [State.ACTING, State.IDLE],
    State.ACTING:    [State.SPEAKING, State.IDLE],
    State.SPEAKING:  [State.IDLE],
}

class StateMachine:
    def __init__(self):
        self.current = State.IDLE
        self._lock   = asyncio.Lock()

    async def transition(self, new_state: State):
        async with self._lock:
            allowed = TRANSITIONS.get(self.current, [])
            if new_state not in allowed:
                log.warning(f"Invalid transition {self.current} → {new_state}. Ignored.")
                return
            log.info(f"State: {self.current.value} → {new_state.value}")
            self.current = new_state

    def is_busy(self) -> bool:
        return self.current != State.IDLE
