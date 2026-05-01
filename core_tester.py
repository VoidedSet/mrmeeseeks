import logging
import os
from datetime import datetime

def setup_file_logging():
    os.makedirs("logs/outputs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/outputs/run_{timestamp}.txt"

    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    print(f"[LOG] Writing full run to {log_file}")
    return log_file

log_file = setup_file_logging()

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