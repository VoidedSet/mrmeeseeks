import os
import time
import uuid
import logging
import requests
from typing import Dict, Any, List, Optional
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

log = logging.getLogger("chroma_store")

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, url: str = "http://localhost:11434/api/embed", model_name: str = "nomic-embed-text"):
        if url.endswith("/api/embeddings"):
            url = url.replace("/api/embeddings", "/api/embed")
        self.url = url
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = requests.post(
                self.url,
                json={"model": self.model_name, "input": input},
                timeout=30
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if embeddings:
                return embeddings
        except Exception as e:
            log.warning(f"Ollama /api/embed failed, trying fallback: {e}")
            
        fallback_url = self.url.replace("/api/embed", "/api/embeddings")
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    fallback_url,
                    json={"model": self.model_name, "prompt": text},
                    timeout=15
                )
                response.raise_for_status()
                emb = response.json().get("embedding")
                if emb:
                    embeddings.append(emb)
                    continue
            except Exception:
                pass
            # Default zero vector if embedding fails completely
            embeddings.append([0.0] * 768)
        return embeddings

class FastCpuEmbeddingFunction(EmbeddingFunction):
    """
    Local CPU ONNX embedding function to prevent Ollama from swapping/unloading Qwen in GPU VRAM.
    Uses CPUExecutionProvider explicitly to prevent TensorRT/CUDA probing warnings during vector DB queries.
    """
    def __init__(self):
        self._onnx_ef = None
        self._fallback_ef = None
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
            class ExplicitCpuMiniLM(ONNXMiniLM_L6_V2):
                def __init__(self):
                    import onnxruntime as ort
                    # Suppress ORT C++ provider probing logs
                    sess_opts = ort.SessionOptions()
                    sess_opts.log_severity_level = 3
                    super().__init__()
                    # Override underlying session to use CPU provider explicitly
                    try:
                        model_path = self._model_path if hasattr(self, "_model_path") else None
                        if model_path and os.path.exists(model_path):
                            self._session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
                    except Exception:
                        pass

            self._onnx_ef = ExplicitCpuMiniLM()
            log.info("Initialized local ONNX CPU embedding function (prevents Ollama VRAM swapping) ✓")
        except Exception as e:
            log.warning(f"Could not load local ONNX embedding function: {e}")
            self._fallback_ef = OllamaEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        if self._onnx_ef is not None:
            try:
                return self._onnx_ef(input)
            except Exception as e:
                log.warning(f"ONNX CPU embedding execution failed, using fallback: {e}")
        if self._fallback_ef is None:
            self._fallback_ef = OllamaEmbeddingFunction()
        return self._fallback_ef(input)


def _get_or_create_compat_collection(client, name: str, embedding_function):
    """
    Get or create collection. If existing collection has incompatible vector dimensions
    (e.g., 768 from old Ollama embed vs 384 from local ONNX), delete and recreate it.
    """
    try:
        col = client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        if col.count() > 0:
            col.query(query_texts=["test"], n_results=1)
        return col
    except Exception as e:
        if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
            log.warning(f"[ChromaStore] Resetting collection '{name}' due to dimension mismatch: {e}")
            try:
                client.delete_collection(name=name)
            except Exception:
                pass
            return client.create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        # If any other error, fallback to get_or_create
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )


class ChromaStore:
    def __init__(self):
        # Database directory
        self.db_dir = os.path.expanduser("~/.meeseeks/chroma_db")
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize chroma client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        
        # Initialize local CPU embedding function to keep GPU VRAM 100% dedicated to Qwen
        self.embedding_function = FastCpuEmbeddingFunction()
        
        # Initialize or get collections with dimension safety
        self.documents_col = _get_or_create_compat_collection(self.client, "meeseeks_documents", self.embedding_function)
        self.memories_col = _get_or_create_compat_collection(self.client, "meeseeks_memories", self.embedding_function)
        self.conversations_col = _get_or_create_compat_collection(self.client, "meeseeks_conversations", self.embedding_function)
        self.emails_col = _get_or_create_compat_collection(self.client, "meeseeks_emails", self.embedding_function)
        self.news_col = _get_or_create_compat_collection(self.client, "meeseeks_news", self.embedding_function)
        self.goals_col = _get_or_create_compat_collection(self.client, "meeseeks_goals", self.embedding_function)
        self.web_cache_col = _get_or_create_compat_collection(self.client, "meeseeks_web_cache", self.embedding_function)
        
        log.info(f"ChromaStore initialized at {self.db_dir}")

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
        Adds a document to the Chroma DB 'documents' collection.
        Returns a dict containing 'id' and 'status'.
        """
        if sanitize:
            is_url = content.strip().startswith(("http://", "https://")) and "\n" not in content.strip()
            if not is_url:
                content = content.replace("<", "[").replace(">", "]")
                prefix = f"Document content of {os.path.basename(filepath)}:\n\n" if filepath else "Document content:\n\n"
                content = prefix + content

        doc_id = custom_id or f"doc_{uuid.uuid4().hex[:12]}"
        
        # Build Chroma-compatible flat metadata
        meta = {
            "filepath": filepath or "",
            "container_tag": container_tag,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "is_latest": True,
            "source_type": "document"
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        try:
            # Check for existing document with same filepath to prevent duplicate insertion
            if filepath:
                existing = self.documents_col.get(where={"filepath": filepath})
                if existing and existing.get("ids"):
                    # Document already exists, delete older version
                    self.documents_col.delete(ids=existing["ids"])
                    log.info(f"[ChromaStore] Replacing existing document for {filepath}")

            self.documents_col.add(
                documents=[content],
                ids=[doc_id],
                metadatas=[meta]
            )
            return {"id": doc_id, "status": "success"}
        except Exception as e:
            log.error(f"[ChromaStore] add_document error: {e}")
            return None

    def add_image(
        self,
        filepath: str,
        container_tag: str,
        parent_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Performs local OCR on an image file and indexes the extracted text.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            log.error("[ChromaStore] pytesseract or Pillow not installed. Run: pip install pytesseract Pillow")
            return None

        try:
            img = Image.open(filepath)
            if img.mode not in ("RGB", "L", "RGBA"):
                img = img.convert("RGB")
            ocr_text = pytesseract.image_to_string(img, timeout=10.0).strip()
        except Exception as e:
            log.error(f"[ChromaStore] OCR failed for {filepath}: {e}")
            return None

        if not ocr_text or len(ocr_text) < 10:
            log.info(f"[ChromaStore] No usable text found in image {os.path.basename(filepath)} — skipping.")
            return None

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
            sanitize=False,
        )

    # ── Document Management ───────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the documents collection by ID."""
        try:
            self.documents_col.delete(ids=[doc_id])
            return True
        except Exception as e:
            log.error(f"[ChromaStore] delete_document error: {e}")
            return False

    def get_document_status(self, doc_id: str) -> Optional[str]:
        """Ingestion is synchronous, so if it exists in the DB, it is 'done'."""
        try:
            existing = self.documents_col.get(ids=[doc_id])
            if existing and existing.get("ids"):
                return "done"
            return None
        except Exception:
            return None

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query text using the configured Ollama embedding function."""
        embeddings = self.embedding_function([query])
        return embeddings[0]

    # ── Search ────────────────────────────────────────────────────────────────

    def search_documents(
        self,
        query: str,
        container_tag: str,
        limit: int = 5,
        chunk_threshold: float = 0.3,
        query_embeddings: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Queries document chunks via semantic search."""
        try:
            if query_embeddings is not None:
                query_res = self.documents_col.query(
                    query_embeddings=[query_embeddings],
                    n_results=limit,
                    where={"container_tag": container_tag}
                )
            else:
                query_res = self.documents_col.query(
                    query_texts=[query],
                    n_results=limit,
                    where={"container_tag": container_tag}
                )
            
            results = []
            if not query_res or not query_res.get("documents") or len(query_res["documents"]) == 0:
                return []
                
            ids = query_res["ids"][0]
            documents = query_res["documents"][0]
            metadatas = query_res["metadatas"][0]
            distances = query_res["distances"][0]
            
            for i in range(len(ids)):
                doc_id = ids[i]
                content = documents[i]
                meta = metadatas[i] or {}
                distance = distances[i]
                similarity = 1.0 - (distance / 2.0)
                
                if similarity >= chunk_threshold:
                    results.append({
                        "content": content,
                        "filepath": meta.get("filepath"),
                        "similarity": similarity,
                        "doc_id": doc_id,
                        "type": "image" if meta.get("source_type") == "image_ocr" else "document"
                    })
            return results
        except Exception as e:
            log.error(f"[ChromaStore] search_documents error: {e}")
            return []

    def search_all(
        self,
        query: str,
        container_tags: List[str],
        limit: int = 8,
        query_embeddings: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Searches across multiple container tags and returns a merged list grouped by filepath."""
        try:
            where_clause = {}
            if len(container_tags) == 1:
                where_clause = {"container_tag": container_tags[0]}
            elif len(container_tags) > 1:
                where_clause = {"container_tag": {"$in": container_tags}}

            if query_embeddings is not None:
                query_res = self.documents_col.query(
                    query_embeddings=[query_embeddings],
                    n_results=limit * 2,
                    where=where_clause
                )
            else:
                query_res = self.documents_col.query(
                    query_texts=[query],
                    n_results=limit * 2,
                    where=where_clause
                )
            
            if not query_res or not query_res.get("documents") or len(query_res["documents"]) == 0:
                return []

            all_chunks = []
            ids = query_res["ids"][0]
            documents = query_res["documents"][0]
            metadatas = query_res["metadatas"][0]
            distances = query_res["distances"][0]

            for i in range(len(ids)):
                meta = metadatas[i] or {}
                distance = distances[i]
                similarity = 1.0 - (distance / 2.0)
                
                all_chunks.append({
                    "content": documents[i],
                    "filepath": meta.get("filepath"),
                    "similarity": similarity,
                    "doc_id": ids[i],
                    "type": "image" if meta.get("source_type") == "image_ocr" else "document"
                })

            # Group by filepath, keep best similarity
            best: Dict[str, Dict[str, Any]] = {}
            for chunk in all_chunks:
                fp = chunk.get("filepath") or chunk.get("content")[:40]
                score = chunk.get("similarity", 0)
                if fp not in best or score > best[fp].get("similarity", 0):
                    best[fp] = chunk

            return sorted(best.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:limit]
        except Exception as e:
            log.error(f"[ChromaStore] search_all error: {e}")
            return []

    # ── Memory & Conversations ────────────────────────────────────────────────

    def ingest_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        container_tags: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ingests a conversation history block into the database."""
        try:
            text_content = "\n".join(f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages)
            doc_id = f"conv_{conversation_id}_{int(time.time())}"
            
            meta = {
                "conversation_id": conversation_id,
                "container_tags": ",".join(container_tags),
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "is_latest": True,
                "source_type": "conversation"
            }
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v

            self.conversations_col.add(
                documents=[text_content],
                ids=[doc_id],
                metadatas=[meta]
            )
            return {"id": doc_id, "status": "success"}
        except Exception as e:
            log.error(f"[ChromaStore] ingest_conversation error: {e}")
            return None

    def search_memories(
        self,
        query: str,
        container_tag: str,
        limit: int = 3,
        query_embeddings: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Searches high-level memories."""
        try:
            if query_embeddings is not None:
                query_res = self.memories_col.query(
                    query_embeddings=[query_embeddings],
                    n_results=limit,
                    where={"$and": [{"container_tag": container_tag}, {"is_latest": True}]}
                )
            else:
                query_res = self.memories_col.query(
                    query_texts=[query],
                    n_results=limit,
                    where={"$and": [{"container_tag": container_tag}, {"is_latest": True}]}
                )
            
            results = []
            if not query_res or not query_res.get("documents") or len(query_res["documents"]) == 0:
                return []
                
            ids = query_res["ids"][0]
            documents = query_res["documents"][0]
            metadatas = query_res["metadatas"][0]
            distances = query_res["distances"][0]

            for i in range(len(ids)):
                distance = distances[i]
                similarity = 1.0 - (distance / 2.0)
                results.append({
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "similarity": similarity
                })
            return results
        except Exception as e:
            log.error(f"[ChromaStore] search_memories error: {e}")
            return []

    def get_profile(self, container_tag: str, query: Optional[str] = None, query_embeddings: Optional[List[float]] = None) -> Dict[str, Any]:
        """Returns the aggregated user profile for a given container tag."""
        try:
            # Query the memories collection
            if query_embeddings is not None:
                query_res = self.memories_col.query(
                    query_embeddings=[query_embeddings],
                    n_results=10,
                    where={"$and": [{"container_tag": container_tag}, {"is_latest": True}]}
                )
                docs = query_res.get("documents", [[]])[0] if query_res else []
            elif query:
                query_res = self.memories_col.query(
                    query_texts=[query],
                    n_results=10,
                    where={"$and": [{"container_tag": container_tag}, {"is_latest": True}]}
                )
                docs = query_res.get("documents", [[]])[0] if query_res else []
            else:
                query_res = self.memories_col.get(
                    where={"$and": [{"container_tag": container_tag}, {"is_latest": True}]},
                    limit=30
                )
                docs = query_res.get("documents", []) if query_res else []

            profile_text = "\n".join(docs)
            return {"profile": profile_text, "summary": profile_text}
        except Exception as e:
            log.error(f"[ChromaStore] get_profile error: {e}")
            return {"profile": "", "summary": ""}

    def add_memories(
        self,
        memories: List[str],
        container_tag: str,
        is_static: bool = False,
        session_id: Optional[str] = None,
        memory_type: str = "fact"
    ) -> Optional[Dict[str, Any]]:
        """Manually creates memory entries."""
        try:
            ids = []
            metadatas = []
            for m in memories:
                doc_id = f"mem_{uuid.uuid4().hex[:12]}"
                ids.append(doc_id)
                meta = {
                    "container_tag": container_tag,
                    "is_static": is_static,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                    "is_latest": True,
                    "memory_type": memory_type,
                    "session_id": session_id or "",
                    "source_type": "manual"
                }
                metadatas.append(meta)

            self.memories_col.add(
                documents=memories,
                ids=ids,
                metadatas=metadatas
            )
            return {"ids": ids, "status": "success"}
        except Exception as e:
            log.error(f"[ChromaStore] add_memories error: {e}")
            return None

    def consolidate_session(self, session_id: str):
        """
        Deduplicates newly added memories from the session and updates metadata.
        Purges temporary memories older than 24 hours.
        """
        try:
            log.info(f"Consolidating session {session_id}...")
            
            # 1. Fetch memories from this session
            session_mems = self.memories_col.get(
                where={"$and": [{"session_id": session_id}, {"memory_type": "fact"}]}
            )
            
            if session_mems and session_mems.get("ids"):
                ids = session_mems["ids"]
                documents = session_mems["documents"]
                metadatas = session_mems["metadatas"]
                
                for i in range(len(ids)):
                    new_id = ids[i]
                    new_fact = documents[i]
                    new_meta = metadatas[i] or {}
                    new_created_at = new_meta.get("created_at", 0)
                    
                    # Search for semantic duplicates of this fact in all older latest facts
                    # Exclude the new fact's ID
                    search_res = self.memories_col.query(
                        query_texts=[new_fact],
                        n_results=3,
                        where={"$and": [{"is_latest": True}, {"memory_type": "fact"}]}
                    )
                    
                    if search_res and search_res.get("ids") and len(search_res["ids"][0]) > 0:
                        match_ids = search_res["ids"][0]
                        match_distances = search_res["distances"][0]
                        match_metadatas = search_res["metadatas"][0]
                        
                        for idx, match_id in enumerate(match_ids):
                            # Skip if matching ourselves
                            if match_id == new_id:
                                continue
                            
                            similarity = 1.0 - (match_distances[idx] / 2.0)
                            if similarity > 0.85:
                                matched_meta = match_metadatas[idx] or {}
                                old_created_at = matched_meta.get("created_at", 0)
                                
                                # Tie-breaker: matched metadata is older
                                if old_created_at < new_created_at or (old_created_at == new_created_at and match_id < new_id):
                                    log.info(f"Deduplicating: New fact '{new_fact}' supersedes old fact (ID: {match_id}) with similarity {similarity:.2f}")
                                    
                                    matched_meta["is_latest"] = False
                                    matched_meta["updated_at"] = int(time.time())
                                    
                                    self.memories_col.update(
                                        ids=[match_id],
                                        metadatas=[matched_meta]
                                    )
                                
            # 2. Purge temporary memories older than 24 hours
            temp_mems = self.memories_col.get(
                where={"memory_type": "temporary"}
            )
            if temp_mems and temp_mems.get("ids"):
                temp_ids = temp_mems["ids"]
                temp_metadatas = temp_mems["metadatas"]
                now = int(time.time())
                ids_to_delete = []
                for idx, temp_id in enumerate(temp_ids):
                    meta = temp_metadatas[idx] or {}
                    created_at = meta.get("created_at", now)
                    if now - created_at > 86400:
                        ids_to_delete.append(temp_id)
                
            # 3. Purge web search cache older than 24 hours (decay stale search results)
            self.purge_stale_web_cache(max_age_hours=24)

            log.info(f"Session {session_id} consolidated successfully.")
        except Exception as e:
            log.error(f"[ChromaStore] consolidate_session error: {e}")

    def purge_stale_web_cache(self, max_age_hours: float = 24.0) -> None:
        """Purge web search cache items older than max_age_hours (temporary memory decay)."""
        try:
            cached = self.web_cache_col.get()
            if cached and cached.get("ids"):
                ids = cached["ids"]
                metadatas = cached["metadatas"]
                now = int(time.time())
                max_age_sec = max_age_hours * 3600
                stale_ids = []
                for idx, cid in enumerate(ids):
                    meta = metadatas[idx] or {}
                    ts = meta.get("timestamp", 0)
                    if (now - ts) > max_age_sec:
                        stale_ids.append(cid)

                if stale_ids:
                    self.web_cache_col.delete(ids=stale_ids)
                    log.info(f"[ChromaStore] Purged {len(stale_ids)} web search cache items older than {max_age_hours}h.")
        except Exception as e:
            log.error(f"[ChromaStore] purge_stale_web_cache error: {e}")

    # ── Email, News & Goal Operations ─────────────────────────────────────────

    def add_email(self, doc_text: str, message_id: str, metadata: Dict[str, Any]) -> None:
        """Add or update an email document in meeseeks_emails collection."""
        try:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, message_id))
            self.emails_col.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata]
            )
        except Exception as e:
            log.error(f"[ChromaStore] add_email error: {e}")

    def search_emails(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic search in meeseeks_emails collection."""
        try:
            res = self.emails_col.query(
                query_texts=[query],
                n_results=limit
            )
            results = []
            if res and res.get("documents") and len(res["documents"][0]) > 0:
                for i, doc in enumerate(res["documents"][0]):
                    meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                    results.append({
                        "text": doc,
                        "sender": meta.get("sender", ""),
                        "sender_name": meta.get("sender_name", ""),
                        "subject": meta.get("subject", ""),
                        "date": meta.get("date", ""),
                        "uid": meta.get("email_uid", "")
                    })
            return results
        except Exception as e:
            log.error(f"[ChromaStore] search_emails error: {e}")
            return []

    def save_web_cache(self, query: str, result: str) -> None:
        """Cache a web search query and its result in meeseeks_web_cache."""
        try:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"web_cache:{query.lower().strip()}"))
            metadata = {
                "query": query,
                "timestamp": int(time.time()),
                "date": time.strftime("%Y-%m-%d")
            }
            self.web_cache_col.upsert(
                ids=[doc_id],
                documents=[result],
                metadatas=[metadata]
            )
            log.info(f"[ChromaStore] Web search cached for: '{query}'")
        except Exception as e:
            log.error(f"[ChromaStore] save_web_cache error: {e}")

    def get_web_cache(self, query: str, max_age_hours: float = 4.0) -> Optional[str]:
        """Retrieve cached web search result if fresh and semantically matching."""
        try:
            res = self.web_cache_col.query(
                query_texts=[query],
                n_results=1
            )
            if res and res.get("documents") and len(res["documents"][0]) > 0:
                distance = res["distances"][0][0] if res.get("distances") else 1.0
                similarity = 1.0 - (distance / 2.0)
                meta = res["metadatas"][0][0] if res.get("metadatas") else {}
                cached_time = meta.get("timestamp", 0)
                now = int(time.time())

                if similarity >= 0.85 and (now - cached_time) <= (max_age_hours * 3600):
                    doc = res["documents"][0][0]
                    log.info(f"[ChromaStore] Web cache HIT (sim={similarity:.2f}, age={(now - cached_time)/60:.1f}m) for '{query}'")
                    return doc
        except Exception as e:
            log.error(f"[ChromaStore] get_web_cache error: {e}")
        return None

chroma_store = ChromaStore()
