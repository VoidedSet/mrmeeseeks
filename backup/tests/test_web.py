import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
from agents.web_agent import handle_simple_scrape
from core.ipc_bus import bus

# Dummy update_memory to prevent errors since we aren't loading the full memory agent
async def fake_update(args):
    print("MOCK MEMORY UPDATE:", args)
    return {"ok": True}
bus.register("update_memory", fake_update)

async def main():
    res = await handle_simple_scrape({"query": "messi"})
    print("RESULT:", res)

asyncio.run(main())
