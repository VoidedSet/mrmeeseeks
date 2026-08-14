import os
import sys
import asyncio
import sqlite3
import time

# Add the project root to python path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.idle_indexer_daemon import IdleIndexerDaemon

async def test_indexing_pipeline():
    print("=== STARTING INDEXER TESTS ===\n")

    # Paths
    vault_dir = "/home/kshayik/.gemini/antigravity/scratch/test_vault"
    test_db = "/home/kshayik/.gemini/antigravity/scratch/test_indexer_state.db"

    # Clean up previous test database files
    if os.path.exists(test_db):
        os.remove(test_db)

    # Test Idle Daemon Scanning
    print("[Test] Testing Idle Indexer Daemon Directory Crawling...")
    try:
        # Create a daemon scoped only to our scratch test_vault
        daemon = IdleIndexerDaemon(
            scan_paths=[vault_dir],
            idle_threshold=1.0,  # 1 second threshold for testing
            db_path=test_db
        )

        # Make sure the monitor is marked as idle for testing
        daemon.monitor.last_activity = time.time() - 5  # Force idle state

        print("Running first batch scan...")
        await daemon.scan_and_index_batch()

        # Check state db contents
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, status FROM indexed_files")
        records = cursor.fetchall()
        conn.close()

        print(f"\nIndexed files logged in state database:")
        for path, status in records:
            print(f" - {os.path.basename(path)}: {status}")

    except Exception as e:
        print(f"Daemon scanner test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_indexing_pipeline())
