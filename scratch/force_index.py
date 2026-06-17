import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.idle_indexer_daemon import IdleIndexerDaemon

async def main():
    daemon = IdleIndexerDaemon(
        scan_paths=None,
        idle_threshold=0.0,
    )
    # Force the daemon to consider the system idle by setting last activity to 0
    daemon.monitor.last_activity = 0
    print("[Force Index] Starting scan...")
    await daemon.scan_and_index_batch()
    print("[Force Index] Scanning complete.")

if __name__ == "__main__":
    asyncio.run(main())
