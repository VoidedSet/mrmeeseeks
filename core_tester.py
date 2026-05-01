from core.ipc_bus import bus
from core.brain import brain

from agents.sysadmin_agent import register
register()

# # mock any tool
# async def fake_run_cmd(args): return {"output": "fake result"}
# bus.register("run_bg_cmd", fake_run_cmd)
# bus.register("done", lambda a: a)

import asyncio
result = asyncio.run(brain.process("create a file named test.txt and write 'hello world' in it, then check battery status"))
# print(result)