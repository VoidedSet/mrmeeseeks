"""
meeseeks_service.py — Unified Mr. Meeseeks Backend (port 11400)

Serves two consumers:
  1. GNOME Shell extension  — GET /search?q=
  2. Meeseeks LLM (Qwen)   — GET /context?q=, POST /memory, GET /profile

All file indexing and semantic search is backed by Supermemory (port 6767).
CLIP is not used.
"""

import os
import sys
import json
import asyncio
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from subsystems.brain.supermemory_client import SupermemoryClient
from subsystems.brain.idle_indexer_daemon import IdleIndexerDaemon

# Search container tags — all indexed content lives here
SEARCH_TAGS = ["personal_notes", "projects"]

daemon_instance: Optional[IdleIndexerDaemon] = None


# ── Request Handler ────────────────────────────────────────────────────────────

class MeeseeksAPIHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Suppress default per-request logs; use print for meaningful events only
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/search":
            query = params.get("q", [""])[0].strip()
            if not query:
                self._send_json([])
                return
            print(f"[API] /search q={repr(query)}")
            results = self._unified_search(query)
            self._send_json(results)

        elif parsed.path == "/context":
            query = params.get("q", [""])[0].strip()
            if not query:
                self._send_json({"memories": [], "documents": [], "profile": ""})
                return
            print(f"[API] /context q={repr(query)}")
            ctx = self._build_llm_context(query)
            self._send_json(ctx)

        elif parsed.path == "/profile":
            tag = params.get("tag", ["personal_notes"])[0]
            sm = SupermemoryClient()
            self._send_json(sm.get_profile(container_tag=tag))

        elif parsed.path == "/status":
            self._send_json(self._get_status())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON payload.")
            return

        if parsed.path == "/memory":
            self._handle_store_memory(payload)

        elif parsed.path == "/index":
            self._handle_manual_index(payload)

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── Search ────────────────────────────────────────────────────────────────

    def _find_matching_page_in_pdf(self, filepath: str, snippet: str) -> int:
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            snippet_clean = snippet.lower().strip()
            if not snippet_clean:
                return 1
            
            # 1. Exact match search
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and snippet_clean in text.lower():
                    return i + 1
            
            # 2. Heuristic search: match lines of at least 15 characters
            lines = [l.strip() for l in snippet.split("\n") if len(l.strip()) > 15]
            if not lines:
                import re
                lines = [s.strip() for s in re.split(r'[.!?；。！？\n]', snippet) if len(s.strip()) > 15]
                
            if lines:
                best_page = 1
                max_matches = 0
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    text_lower = text.lower()
                    matches = sum(1 for line in lines if line.lower() in text_lower)
                    if matches > max_matches:
                        max_matches = matches
                        best_page = i + 1
                return best_page
        except Exception as e:
            print(f"[Preview] Error finding matching page in PDF {filepath}: {e}")
        return 1

    def _get_pdf_preview(self, filepath: str, snippet: str = "") -> Optional[str]:
        import hashlib
        import subprocess
        if not filepath.lower().endswith(".pdf"):
            return None
        try:
            mtime = os.path.getmtime(filepath)
            page_num = self._find_matching_page_in_pdf(filepath, snippet)
            
            hasher = hashlib.md5()
            hasher.update(f"{filepath}_{mtime}_page_{page_num}".encode("utf-8"))
            hash_str = hasher.hexdigest()
            
            preview_path = f"/tmp/meeseeks_pdf_{hash_str}"
            expected_png = f"{preview_path}-{page_num}.png"
            
            if os.path.exists(expected_png):
                return expected_png
                
            cmd = ["pdftoppm", "-png", "-f", str(page_num), "-l", str(page_num), "-r", "150", filepath, preview_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if os.path.exists(expected_png):
                return expected_png
        except Exception as e:
            print(f"[Preview] Error generating PDF preview for {filepath}: {e}")
        return None

    def _unified_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries Supermemory across all content containers.
        Groups results by filepath — one entry per unique file, best chunk wins.

        Returns GNOME-shell-compatible result list:
        [{"id", "title", "filepath", "type", "snippet", "score", "preview_image"}, ...]
        """
        sm = SupermemoryClient()
        grouped = sm.search_all(query=query, container_tags=SEARCH_TAGS, limit=8)

        results = []
        for item in grouped:
            filepath = item.get("filepath")
            snippet = item.get("content", "")
            score = item.get("similarity", 0)
            item_type = item.get("type", "document")

            # Strip the "Document content of X:" prefix we add during sanitization
            if snippet.startswith("Document content of") or snippet.startswith("OCR text from"):
                newline = snippet.find("\n\n")
                if newline != -1:
                    snippet = snippet[newline + 2:]

            title = os.path.basename(filepath) if filepath else "Memory"
            unique_id = f"{item_type}_{abs(hash(filepath or snippet))}"

            preview_image = None
            if filepath:
                preview_image = self._get_pdf_preview(filepath, snippet)

            results.append({
                "id": unique_id,
                "title": title,
                "filepath": filepath,
                "type": item_type,
                "snippet": snippet,
                "score": round(score, 4),
                "preview_image": preview_image,
            })

        return results

    # ── LLM Context ───────────────────────────────────────────────────────────

    def _build_llm_context(self, query: str) -> Dict[str, Any]:
        """
        Builds a context payload for the Meeseeks LLM to inject before replying.

        Returns:
            {
              "memories":  ["fact 1", "fact 2", ...],
              "documents": ["relevant chunk from file...", ...],
              "profile":   "User profile summary string"
            }
        """
        sm = SupermemoryClient()

        # Fetch relevant document chunks
        doc_chunks = sm.search_all(query=query, container_tags=SEARCH_TAGS, limit=5)
        documents = [item.get("content", "") for item in doc_chunks if item.get("content")]

        # Fetch relevant memories (extracted facts from conversations)
        memory_results = sm.search_memories(query=query, container_tag="chat_memory", limit=4)
        memories = [m.get("content", "") for m in memory_results if m.get("content")]

        # User profile (cached — light call)
        profile_data = sm.get_profile(container_tag="chat_memory", query=query)
        profile_text = profile_data.get("profile", "") or profile_data.get("summary", "")

        return {
            "memories": memories,
            "documents": documents,
            "profile": profile_text,
        }

    # ── Memory Storage ────────────────────────────────────────────────────────

    def _handle_store_memory(self, payload: Dict[str, Any]):
        """
        POST /memory — Store a conversation turn or direct fact into SM graph memory.

        Accepts either:
          { "conversationId": "...", "messages": [...], "tags": [...] }
        or:
          { "memories": ["fact 1", ...], "tag": "chat_memory" }
        """
        sm = SupermemoryClient()

        if "conversationId" in payload:
            conv_id = payload.get("conversationId")
            messages = payload.get("messages", [])
            tags = payload.get("tags", ["chat_memory"])
            if not conv_id or not messages:
                self._send_error(400, "Missing 'conversationId' or 'messages'.")
                return
            result = sm.ingest_conversation(conversation_id=conv_id, messages=messages, container_tags=tags)
            self._send_json(result or {"status": "error"})

        elif "memories" in payload:
            memories = payload.get("memories", [])
            tag = payload.get("tag", "chat_memory")
            is_static = payload.get("isStatic", False)
            result = sm.add_memories(memories=memories, container_tag=tag, is_static=is_static)
            self._send_json(result or {"status": "error"})

        else:
            self._send_error(400, "Payload must contain 'conversationId'+'messages' or 'memories'.")

    # ── Manual Index ──────────────────────────────────────────────────────────

    def _handle_manual_index(self, payload: Dict[str, Any]):
        """
        POST /index — Trigger immediate indexing of a given file path.

        Body: { "filepath": "/absolute/path/to/file" }
        """
        global daemon_instance
        filepath = payload.get("filepath")
        if not filepath or not os.path.exists(filepath):
            self._send_error(400, f"File not found: {filepath}")
            return

        if daemon_instance is None:
            self._send_error(503, "Indexer daemon not running.")
            return

        async def do_index():
            result = await daemon_instance.index_file(filepath)
            if result and result not in ("no_text",):
                mtime = os.path.getmtime(filepath)
                sm_id = result if result not in ("empty", "no_text_but_images") else None
                daemon_instance._mark_as_indexed(filepath, mtime, sm_doc_id=sm_id)

        # Run in the daemon's event loop
        loop = asyncio.new_event_loop()
        loop.run_until_complete(do_index())
        loop.close()

        self._send_json({"status": "ok", "filepath": filepath})

    # ── Status ────────────────────────────────────────────────────────────────

    def _get_status(self) -> Dict[str, Any]:
        import sqlite3
        global daemon_instance

        status: Dict[str, Any] = {
            "daemon_running": daemon_instance is not None,
            "user_idle": daemon_instance.monitor.is_idle if daemon_instance else True,
            "indexed_files_count": 0,
            "supermemory_url": os.getenv("SUPERMEMORY_URL", "http://localhost:6767"),
        }

        state_db = os.path.expanduser("~/.supermemory/indexer_state.db")
        if os.path.exists(state_db):
            try:
                conn = sqlite3.connect(state_db)
                count = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
                status["indexed_files_count"] = count
                conn.close()
            except Exception:
                pass

        return status

    # ── Response Helpers ──────────────────────────────────────────────────────

    def _send_json(self, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ── Server Entry Point ────────────────────────────────────────────────────────

def start_http_server(port: int = 11400):
    server = HTTPServer(("localhost", port), MeeseeksAPIHandler)
    print(f"[Service] API server listening at http://localhost:{port}")
    server.serve_forever()


async def main():
    global daemon_instance
    print("=" * 55)
    print("  MR. MEESEEKS UNIFIED BACKEND — starting up")
    print("=" * 55)

    # 1. HTTP server in background thread
    api_thread = threading.Thread(target=start_http_server, daemon=True)
    api_thread.start()
    print("[Service] HTTP API server started on port 11400.")

    # 2. Idle indexer daemon in async loop (automatically handles dynamic scan paths, checking mounts & excluding projects)
    daemon_instance = IdleIndexerDaemon(
        scan_paths=None,
        idle_threshold=60.0,
    )
    await daemon_instance.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Service] Shutting down.")
