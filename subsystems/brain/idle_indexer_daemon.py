import os
import io
import time
import sqlite3
import asyncio
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
import pypdf
import evdev

from subsystems.brain.supermemory_client import SupermemoryClient


# ── Activity Monitor ──────────────────────────────────────────────────────────

class ActivityMonitor:
    """Monitors raw mouse/keyboard events via evdev to determine user idle state."""

    def __init__(self, idle_threshold_seconds: float = 60.0):
        self.idle_threshold = idle_threshold_seconds
        self.last_activity = time.time()
        self.devices = []

        for path in evdev.list_devices():
            try:
                self.devices.append(evdev.InputDevice(path))
            except Exception:
                pass
        print(f"[ActivityMonitor] Monitoring {len(self.devices)} input devices.")

    @property
    def is_idle(self) -> bool:
        return (time.time() - self.last_activity) > self.idle_threshold

    async def start_listening(self):
        """Listens for raw input events asynchronously."""
        if not self.devices:
            print("[ActivityMonitor] No accessible input devices — defaulting to ALWAYS idle.")
            return

        from selectors import DefaultSelector, EVENT_READ
        selector = DefaultSelector()
        for dev in self.devices:
            try:
                selector.register(dev, EVENT_READ)
            except Exception:
                pass

        while True:
            events = selector.select(timeout=0.1)
            if events:
                self.last_activity = time.time()
                for key, _ in events:
                    try:
                        for _ in key.fileobj.read():
                            pass
                    except Exception:
                        pass
            await asyncio.sleep(0.2)


# ── Idle Indexer Daemon ───────────────────────────────────────────────────────

class IdleIndexerDaemon:
    """
    Background daemon that indexes user files when the system is idle.

    All content — text documents, PDFs (text + embedded images), DOCX, and
    standalone images — is routed through Supermemory via OCR or direct text.
    CLIP is not used. Images with no extractable text are skipped.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".webp"}

    def __init__(
        self,
        scan_paths: Optional[List[str]] = None,
        idle_threshold: float = 60.0,
        db_path: Optional[str] = None,
    ):
        self.configured_scan_paths = scan_paths
        self.db_path = db_path or os.path.expanduser("~/.supermemory/indexer_state.db")
        self.monitor = ActivityMonitor(idle_threshold_seconds=idle_threshold)
        self.sm = SupermemoryClient()
        self._init_state_db()

    def _get_active_scan_paths(self) -> List[str]:
        """
        Dynamically resolves and filters the active scan paths.
        1. Excludes all 'Projects' or 'projects' directories.
        2. Checks if '/media/kshayik/New Volume' is mounted.
           Only includes paths on New Volume if it is mounted.
        """
        if self.configured_scan_paths is not None:
            new_vol = "/media/kshayik/New Volume"
            new_vol_mounted = False
            try:
                if os.path.exists(new_vol):
                    if os.path.ismount(new_vol) or os.path.exists(os.path.join(new_vol, "Sem 6")):
                        new_vol_mounted = True
            except Exception:
                pass

            filtered_paths = []
            for path in self.configured_scan_paths:
                path_parts = [p.lower() for p in path.split(os.sep)]
                if "projects" in path_parts:
                    continue
                if path.startswith(new_vol) and not new_vol_mounted:
                    continue
                filtered_paths.append(path)
            return filtered_paths

        # Default dynamic resolution
        base_paths = [
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Pictures"),
        ]

        new_vol = "/media/kshayik/New Volume"
        new_vol_mounted = False
        try:
            if os.path.exists(new_vol):
                if os.path.ismount(new_vol) or os.path.exists(os.path.join(new_vol, "Sem 6")):
                    new_vol_mounted = True
        except Exception as e:
            print(f"[Indexer] Error checking mount status for {new_vol}: {e}")

        vol_paths = []
        if new_vol_mounted:
            vol_subdirs = ["Sem 6", "Journal", "Resumes", "Pictures/Adobe Scan Exports"]
            for subdir in vol_subdirs:
                vol_paths.append(os.path.join(new_vol, subdir))
        else:
            print("[Indexer] New Volume is not mounted. Skipping shared volume paths.")
        
        all_paths = base_paths + vol_paths

        filtered_paths = []
        for path in all_paths:
            path_parts = [p.lower() for p in path.split(os.sep)]
            if "projects" in path_parts:
                continue
            filtered_paths.append(path)

        return filtered_paths

    # ── State Database ────────────────────────────────────────────────────────

    def _init_state_db(self):
        """
        Local SQLite DB tracks which files are indexed and their SM document IDs
        so we can delete them from SM when files are removed from disk.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                filepath     TEXT PRIMARY KEY,
                last_modified REAL NOT NULL,
                sm_doc_id    TEXT,
                status       TEXT DEFAULT 'indexed'
            )
        """)
        # Migration: add sm_doc_id column if upgrading from old schema
        try:
            conn.execute("ALTER TABLE indexed_files ADD COLUMN sm_doc_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()
        conn.close()

    def _get_indexed_files(self) -> Dict[str, float]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT filepath, last_modified FROM indexed_files").fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def _mark_as_indexed(self, filepath: str, mtime: float, sm_doc_id: Optional[str] = None):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO indexed_files (filepath, last_modified, sm_doc_id, status) VALUES (?, ?, ?, ?)",
            (os.path.abspath(filepath), mtime, sm_doc_id, "indexed"),
        )
        conn.commit()
        conn.close()

    def _get_sm_doc_id(self, filepath: str) -> Optional[str]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT sm_doc_id FROM indexed_files WHERE filepath = ?", (os.path.abspath(filepath),)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _remove_from_index(self, filepath: str):
        """Removes from state DB and attempts to delete from SM."""
        abs_path = os.path.abspath(filepath)
        sm_doc_id = self._get_sm_doc_id(abs_path)
        if sm_doc_id:
            self.sm.delete_document(sm_doc_id)
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM indexed_files WHERE filepath = ?", (abs_path,))
        conn.commit()
        conn.close()

    # ── File Parsers ──────────────────────────────────────────────────────────

    def _parse_docx(self, filepath: str) -> str:
        """Extracts plain text from a DOCX file."""
        try:
            with zipfile.ZipFile(filepath) as docx:
                tree = ET.fromstring(docx.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                return "\n".join(el.text for el in tree.iter(f"{{{ns['w']}}}t") if el.text)
        except Exception as e:
            print(f"[Indexer] Error parsing DOCX {filepath}: {e}")
            return ""

    def _parse_pdf_text(self, filepath: str) -> str:
        """Extracts text from all pages of a PDF."""
        try:
            reader = pypdf.PdfReader(filepath)
            return "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        except Exception as e:
            print(f"[Indexer] Error parsing PDF text {filepath}: {e}")
            return ""

    def _extract_pdf_images(self, filepath: str) -> List[bytes]:
        """
        Extracts embedded image XObjects from a PDF.
        Returns a list of raw image bytes (PNG/JPEG).
        """
        images = []
        try:
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                resources = page.get("/Resources")
                if not resources:
                    continue
                xobjects = resources.get("/XObject")
                if not xobjects:
                    continue
                for name, obj_ref in xobjects.items():
                    obj = obj_ref.get_object() if hasattr(obj_ref, "get_object") else obj_ref
                    if obj.get("/Subtype") == "/Image":
                        try:
                            data = obj.get_data()
                            if data:
                                images.append(data)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Indexer] Error extracting PDF images from {filepath}: {e}")
        return images

    def _ocr_bytes(self, image_bytes: bytes, label: str) -> str:
        """Runs OCR on raw image bytes. Returns extracted text or empty string."""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return pytesseract.image_to_string(img, timeout=10.0).strip()
        except ImportError:
            print("[Indexer] pytesseract/Pillow not installed — skipping OCR")
            return ""
        except Exception as e:
            print(f"[Indexer] OCR error for {label}: {e}")
            return ""

    # ── Core Indexing Logic ───────────────────────────────────────────────────

    async def index_file(self, filepath: str) -> Optional[str]:
        """
        Indexes a single file by type. Routes all content through Supermemory.

        Returns the SM document ID on success, None on failure or skip.
        """
        abs_path = os.path.abspath(filepath)
        ext = os.path.splitext(abs_path)[1].lower()
        basename = os.path.basename(abs_path)
        tag = "projects" if "/Projects/" in abs_path else "personal_notes"

        # 1. Plain text / Markdown / README
        if ext in (".txt", ".md") or basename.lower() in ("readme", "readme.md", "readme.txt"):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if not content.strip():
                    return "empty"
                resp = self.sm.add_document(content=content, container_tag=tag, filepath=abs_path)
                return resp.get("id") if resp else None
            except Exception as e:
                print(f"[Indexer] Failed indexing text {abs_path}: {e}")
                return None

        # 2. PDF — text + embedded images
        elif ext == ".pdf":
            sm_doc_id = None

            # 2a. Text extraction
            text = self._parse_pdf_text(abs_path)
            if text.strip():
                resp = self.sm.add_document(content=text, container_tag=tag, filepath=abs_path)
                if resp:
                    sm_doc_id = resp.get("id")

            # 2b. Embedded image extraction → OCR → SM
            images = self._extract_pdf_images(abs_path)
            print(f"[Indexer] PDF {basename}: found {len(images)} embedded image(s)")
            for i, img_bytes in enumerate(images):
                ocr_text = self._ocr_bytes(img_bytes, f"{basename}[img{i}]")
                if ocr_text and len(ocr_text) >= 10:
                    content = f"OCR text from embedded image {i+1} in {basename}:\n\n{ocr_text}"
                    self.sm.add_document(
                        content=content,
                        container_tag=tag,
                        filepath=abs_path,
                        metadata={"source_type": "image_ocr", "parent_document": abs_path, "image_index": i},
                        sanitize=False,
                    )
                else:
                    print(f"[Indexer] No usable text in image {i+1} of {basename} — skipping.")
                await asyncio.sleep(0.1)  # Yield to event loop

            return sm_doc_id or ("no_text_but_images" if images else None)

        # 3. Word documents
        elif ext == ".docx":
            content = self._parse_docx(abs_path)
            if not content.strip():
                return "empty"
            resp = self.sm.add_document(content=content, container_tag=tag, filepath=abs_path)
            return resp.get("id") if resp else None

        # 4. Images — OCR via pytesseract, skip if no text
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            resp = self.sm.add_image(filepath=abs_path, container_tag=tag)
            if resp:
                return resp.get("id")
            # No text found — don't index, return special sentinel
            return "no_text"

        return None

    # ── Scan Loop ─────────────────────────────────────────────────────────────

    async def scan_and_index_batch(self):
        """Walks scan paths, finds new/modified files, and indexes them while idle."""
        indexed = self._get_indexed_files()
        current_files: Dict[str, float] = {}

        active_paths = self._get_active_scan_paths()
        for scan_path in active_paths:
            if not os.path.exists(scan_path):
                continue

            for root, dirs, files in os.walk(scan_path):
                # Skip noise directories
                dirs[:] = [d for d in dirs if d not in (
                    "node_modules", "venv", ".git", "__pycache__", "dist", "build",
                    ".venv", ".tox", "target", ".next",
                )]

                # Skip any projects directories completely
                path_parts = [p.lower() for p in root.split(os.sep)]
                if "projects" in path_parts:
                    dirs.clear()
                    continue

                for fname in files:
                    fpath = os.path.join(root, fname)
                    ext = os.path.splitext(fpath)[1].lower()
                    if ext in self.SUPPORTED_EXTENSIONS or fname.lower() == "readme":
                        try:
                            current_files[os.path.abspath(fpath)] = os.path.getmtime(fpath)
                        except Exception:
                            pass

        # Clean up deleted files
        for filepath in list(indexed.keys()):
            if filepath not in current_files:
                print(f"[Indexer] Deleted: {filepath} — removing from index.")
                self._remove_from_index(filepath)

        # Find new/modified files
        pending = [
            (fp, mtime) for fp, mtime in current_files.items()
            if fp not in indexed or mtime > indexed[fp]
        ]

        if not pending:
            return

        print(f"[Indexer] {len(pending)} file(s) to index.")

        for filepath, mtime in pending:
            if not self.monitor.is_idle:
                print("[Indexer] Activity detected — pausing.")
                return

            basename = os.path.basename(filepath)
            print(f"[Indexer] Processing: {basename}")
            result = await self.index_file(filepath)

            if result == "no_text":
                # Image had no OCR text — mark anyway so we don't retry every cycle
                self._mark_as_indexed(filepath, mtime, sm_doc_id=None)
                print(f"[Indexer] {basename} — no text, marked to skip future retries.")
            elif result is not None:
                # Successfully indexed (result is SM doc ID or 'empty'/'no_text_but_images')
                sm_id = result if result not in ("empty", "no_text_but_images") else None
                self._mark_as_indexed(filepath, mtime, sm_doc_id=sm_id)
            else:
                print(f"[Indexer] {basename} — indexing failed, will retry next cycle.")

            await asyncio.sleep(0.5)

    async def start(self):
        """Starts the main daemon loops."""
        print("[Indexer] Starting Idle Indexer Daemon...")
        asyncio.create_task(self.monitor.start_listening())

        while True:
            try:
                if self.monitor.is_idle:
                    await self.scan_and_index_batch()
                else:
                    await asyncio.sleep(5)
            except Exception as e:
                print(f"[Indexer] Loop error: {e}")
            await asyncio.sleep(10)
