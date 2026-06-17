import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

# Add the project root to python path so we can import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.supermemory_client import SupermemoryClient
from core.clip_indexer import CLIPImageIndexer

class SearchHandler(BaseHTTPRequestHandler):
    """
    Exposes a local unified search API on port 11400 for the Ubuntu search bar extension.
    """
    def do_GET(self):
        # Parse query parameters
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path != "/search":
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed_url.query)
        query = params.get("q", [""])[0]

        if not query:
            self._send_json([])
            return

        print(f"[SearchServer] Unified Query: '{query}'")

        # 1. Query Supermemory (Documents & Notes)
        results = []
        try:
            # We search across tags like 'projects' and 'personal_notes'
            sm_client = SupermemoryClient()
            # Search both documents and memories
            text_matches = sm_client.search_documents(query=query, container_tag="personal_notes", limit=3)
            # Add project matches
            proj_matches = sm_client.search_documents(query=query, container_tag="projects", limit=3)
            
            for match in text_matches + proj_matches:
                filepath = match.get("filepath")
                results.append({
                    "id": f"text_{hash(match.get('content'))}",
                    "filepath": filepath,
                    "title": os.path.basename(filepath) if filepath else "Notes Memory",
                    "type": "document",
                    "snippet": match.get("content", ""),
                    "score": float(match.get("similarity", 0))
                })
        except Exception as e:
            print(f"[SearchServer] Supermemory search failed: {e}")

        # 2. Query CLIP (Images)
        try:
            clip_indexer = CLIPImageIndexer()
            image_matches = clip_indexer.search(query=query, limit=3)
            for match in image_matches:
                filepath = match.get("filepath")
                results.append({
                    "id": f"image_{hash(filepath)}",
                    "filepath": filepath,
                    "title": os.path.basename(filepath),
                    "type": "image",
                    "snippet": "Visual image match.",
                    "score": float(match.get("similarity", 0))
                })
        except Exception as e:
            print(f"[SearchServer] CLIP search failed: {e}")

        # Sort combined results by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        self._send_json(results)

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    # Handle CORS preflight
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=11400):
    server = HTTPServer(("localhost", port), SearchHandler)
    print(f"[SearchServer] Exposing unified search API at http://localhost:{port}/search")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping search server.")
        server.server_close()

if __name__ == "__main__":
    run_server()
