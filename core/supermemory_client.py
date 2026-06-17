import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


class SupermemoryClient:
    """
    Python wrapper for the local Supermemory.ai server (port 6767).
    Handles text document ingestion, image OCR ingestion, semantic search,
    conversation memory, and user profile retrieval.

    Image ingestion strategy: OCR is performed locally via pytesseract,
    and the extracted text is sent to SM as a regular text document.
    If no text is found in an image, ingestion is skipped (zero-noise policy).
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url or os.getenv("SUPERMEMORY_URL", "http://localhost:6767")
        self.api_key = api_key or os.getenv("SUPERMEMORY_API_KEY", "")
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }

    # ── Document Ingestion ────────────────────────────────────────────────────

    def add_document(
        self,
        content: str,
        container_tag: str,
        filepath: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_id: Optional[str] = None,
        task_type: str = "superrag",
        sanitize: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Adds a raw text document to Supermemory.

        Returns the SM response dict (contains 'id' and 'status') or None on error.
        """
        if sanitize:
            is_url = content.strip().startswith(("http://", "https://")) and "\n" not in content.strip()
            if not is_url:
                content = content.replace("<", "[").replace(">", "]")
                prefix = f"Document content of {os.path.basename(filepath)}:\n\n" if filepath else "Document content:\n\n"
                content = prefix + content

        if metadata is None:
            metadata = {}
        if filepath:
            metadata["filepath"] = filepath

        payload: Dict[str, Any] = {
            "content": content,
            "containerTag": container_tag,
            "taskType": task_type,
            "metadata": metadata,
        }
        if filepath:
            payload["filepath"] = filepath
        if custom_id:
            payload["customId"] = custom_id

        try:
            r = requests.post(f"{self.base_url}/v3/documents", headers=self.headers, json=payload, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                print(f"[SM] Document already indexed (409): {filepath or 'inline'}")
                return {"id": None, "status": "already_exists"}
            print(f"[SM] add_document HTTP error: {e}")
            return None
        except Exception as e:
            print(f"[SM] add_document error: {e}")
            return None

    def add_image(
        self,
        filepath: str,
        container_tag: str,
        parent_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Performs local OCR on an image file and indexes the extracted text in SM.

        Zero-noise policy: if OCR yields no usable text, returns None and skips indexing.

        Args:
            filepath: Absolute path to the image file (.png/.jpg/.jpeg/.webp).
            container_tag: SM container to store the document in.
            parent_path: If the image was extracted from a PDF, the PDF's path.

        Returns:
            SM response dict (with 'id') if text was found and indexed, None otherwise.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            print("[SM] pytesseract or Pillow not installed — cannot OCR images. Run: pip install pytesseract Pillow")
            return None

        try:
            img = Image.open(filepath)
            # Tesseract works best on RGB/L mode images
            if img.mode not in ("RGB", "L", "RGBA"):
                img = img.convert("RGB")
            ocr_text = pytesseract.image_to_string(img, timeout=10.0).strip()
        except Exception as e:
            print(f"[SM] OCR failed for {filepath}: {e}")
            return None

        if not ocr_text or len(ocr_text) < 10:
            print(f"[SM] No usable text found in image {os.path.basename(filepath)} — skipping.")
            return None

        # Build a meaningful prefix so SM understands the source
        source_label = os.path.basename(filepath)
        if parent_path:
            source_label = f"{os.path.basename(parent_path)} [embedded image: {source_label}]"

        content = f"OCR text from {source_label}:\n\n{ocr_text}"
        metadata: Dict[str, Any] = {"source_type": "image_ocr", "image_path": filepath}
        if parent_path:
            metadata["parent_document"] = parent_path

        return self.add_document(
            content=content,
            container_tag=container_tag,
            filepath=parent_path or filepath,
            metadata=metadata,
            sanitize=False,  # Content is already clean OCR text
        )

    # ── Document Management ───────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a document from SM by its ID.

        Returns True on success, False if the document is still processing or not found.
        """
        try:
            r = requests.delete(
                f"{self.base_url}/v3/documents/{doc_id}",
                headers={"Authorization": self.api_key},
                timeout=5,
            )
            if r.status_code == 200:
                return True
            # SM returns this if still processing
            if "still processing" in r.text:
                print(f"[SM] Document {doc_id} still processing — cannot delete yet.")
            return False
        except Exception as e:
            print(f"[SM] delete_document error: {e}")
            return False

    def get_document_status(self, doc_id: str) -> Optional[str]:
        """Returns the processing status of a document ('queued', 'done', etc.)."""
        try:
            r = requests.get(
                f"{self.base_url}/v3/documents/{doc_id}",
                headers={"Authorization": self.api_key},
                timeout=5,
            )
            if r.ok:
                return r.json().get("status")
            return None
        except Exception:
            return None

    # ── Search ────────────────────────────────────────────────────────────────

    def search_documents(
        self,
        query: str,
        container_tag: str,
        limit: int = 5,
        chunk_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Queries document chunks via SuperRAG semantic search.

        Returns a flat list of chunk dicts:
            [{"content": "...", "filepath": "...", "similarity": 0.87, "doc_id": "..."}, ...]
        """
        try:
            r = requests.post(
                f"{self.base_url}/v3/search",
                headers=self.headers,
                json={
                    "q": query,
                    "containerTags": [container_tag],
                    "limit": limit,
                    "chunkThreshold": chunk_threshold,
                    "onlyMatchingChunks": True,
                },
                timeout=10,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"[SM] search_documents error: {e}")
            return []

        results = []
        for doc in r.json().get("results", []):
            filepath = doc.get("filepath") or doc.get("metadata", {}).get("filepath") or doc.get("metadata", {}).get("image_path")
            doc_id = doc.get("documentId")
            doc_type = doc.get("type", "text")
            source_type = doc.get("metadata", {}).get("source_type", "document")

            for chunk in doc.get("chunks", []):
                results.append({
                    "content": chunk.get("content", ""),
                    "filepath": filepath,
                    "similarity": chunk.get("score", doc.get("score", 0)),
                    "doc_id": doc_id,
                    "type": "image" if source_type == "image_ocr" else "document",
                })
        return results

    def search_all(
        self,
        query: str,
        container_tags: List[str],
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Searches across multiple container tags and returns a merged, deduplicated,
        filepath-grouped list sorted by best score descending.

        One entry per unique filepath — uses the highest-scoring chunk as the snippet.
        """
        all_chunks: List[Dict[str, Any]] = []
        for tag in container_tags:
            all_chunks.extend(self.search_documents(query=query, container_tag=tag, limit=limit))

        # Group by filepath, keep best-score chunk per file
        best: Dict[str, Dict[str, Any]] = {}
        for chunk in all_chunks:
            fp = chunk.get("filepath") or chunk.get("content", "")[:40]
            score = chunk.get("similarity", 0)
            if fp not in best or score > best[fp].get("similarity", 0):
                best[fp] = chunk

        return sorted(best.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:limit]

    # ── Memory & Conversations ────────────────────────────────────────────────

    def ingest_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        container_tags: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a conversation turn into SM graph memory for long-term context.

        Args:
            messages: [{"role": "user"|"assistant", "content": "..."}, ...]
        """
        payload: Dict[str, Any] = {
            "conversationId": conversation_id,
            "messages": messages,
            "containerTags": container_tags,
        }
        if metadata:
            payload["metadata"] = metadata
        try:
            r = requests.post(f"{self.base_url}/v4/conversations", headers=self.headers, json=payload, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[SM] ingest_conversation error: {e}")
            return None

    def search_memories(
        self,
        query: str,
        container_tag: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Searches high-level extracted memories (facts, preferences)."""
        try:
            r = requests.post(
                f"{self.base_url}/v4/search",
                headers=self.headers,
                json={"q": query, "containerTag": container_tag, "limit": limit, "searchMode": "memories"},
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            print(f"[SM] search_memories error: {e}")
            return []

    def get_profile(self, container_tag: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Returns the aggregated user profile for a given container."""
        payload: Dict[str, Any] = {"containerTag": container_tag}
        if query:
            payload["q"] = query
        try:
            r = requests.post(f"{self.base_url}/v4/profile", headers=self.headers, json=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[SM] get_profile error: {e}")
            return {}

    def add_memories(
        self,
        memories: List[str],
        container_tag: str,
        is_static: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Manually creates memory entries."""
        try:
            r = requests.post(
                f"{self.base_url}/v4/memories",
                headers=self.headers,
                json={
                    "memories": [{"content": m, "isStatic": is_static} for m in memories],
                    "containerTag": container_tag,
                },
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[SM] add_memories error: {e}")
            return None
