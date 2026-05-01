from ipc_bus import bus
from brain import brain

import sysadmin_agent
sysadmin_agent.register()  

# # mock any tool
# async def fake_run_cmd(args): return {"output": "fake result"}
# bus.register("run_bg_cmd", fake_run_cmd)
# bus.register("done", lambda a: a)

# test
import asyncio
result = asyncio.run(brain.process("list files in current directory"))
# print(result)