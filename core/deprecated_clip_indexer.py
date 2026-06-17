import os
import sqlite3
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
from transformers import CLIPProcessor, CLIPModel

class CLIPImageIndexer:
    """
    Client-side visual semantic indexer using a CLIP model running on GPU/CPU.
    Stores embeddings in a local SQLite database (~/.supermemory/clip_index.db).
    Computes cosine similarity in PyTorch for ultra-fast, zero-dependency search.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", db_path: Optional[str] = None):
        self.model_name = model_name
        self.db_path = db_path or os.path.expanduser("~/.supermemory/clip_index.db")
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database table for storing image paths and raw binary embeddings."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _lazy_load_model(self):
        """Loads CLIP model lazily to save RAM/GPU memory when the indexer is not actively running."""
        if self.model is None or self.processor is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[CLIPIndexer] Loading model '{self.model_name}' on device: {self.device}")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Loads an image and extracts its normalized CLIP embedding."""
        try:
            self._lazy_load_model()
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                image_features = self.model.visual_projection(vision_outputs.pooler_output)
                # Normalize features to compute cosine similarity via dot product
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy()[0]
        except Exception as e:
            print(f"[CLIPIndexer] Error embedding image {image_path}: {e}")
            return None

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Extracts normalized text embedding for matching against image features."""
        try:
            self._lazy_load_model()
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_outputs = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                text_features = self.model.text_projection(text_outputs.pooler_output)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy()[0]
        except Exception as e:
            print(f"[CLIPIndexer] Error embedding text '{text}': {e}")
            return None

    def index_image(self, filepath: str) -> bool:
        """Embeds an image and stores/updates its vector in the SQLite database."""
        abs_path = os.path.abspath(filepath)
        embedding = self.get_image_embedding(abs_path)
        if embedding is None:
            return False

        # Convert numpy array to binary blob
        embedding_blob = embedding.astype(np.float32).tobytes()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO image_index (filepath, embedding) VALUES (?, ?)",
                (abs_path, embedding_blob)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"[CLIPIndexer] Database save error for {abs_path}: {e}")
            return False
        finally:
            conn.close()

    def remove_image(self, filepath: str):
        """Removes an image from the index (e.g. if deleted)."""
        abs_path = os.path.abspath(filepath)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_index WHERE filepath = ?", (abs_path,))
        conn.commit()
        conn.close()

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic text-to-image search by computing cosine similarities.
        Runs in PyTorch on GPU/CPU in milliseconds.
        """
        query_vec = self.get_text_embedding(query)
        if query_vec is None:
            return []

        # Load all embeddings from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, embedding FROM image_index")
        records = cursor.fetchall()
        conn.close()

        if not records:
            return []

        filepaths = []
        embeddings_list = []
        proj_dim = len(query_vec)

        for filepath, blob in records:
            # Reconstruct numpy array from blob
            arr = np.frombuffer(blob, dtype=np.float32)
            if len(arr) == proj_dim:
                filepaths.append(filepath)
                embeddings_list.append(arr)

        if not embeddings_list:
            return []

        # Stack into PyTorch tensors for vectorized cosine similarity
        db_tensor = torch.tensor(np.array(embeddings_list), dtype=torch.float32).to(self.device)  # shape (N, D)
        q_tensor = torch.tensor(query_vec, dtype=torch.float32).unsqueeze(0).to(self.device)     # shape (1, D)

        # Compute dot product
        similarities = torch.mm(db_tensor, q_tensor.t()).squeeze(-1)  # shape (N,)

        # Sort similarities
        scores, indices = torch.topk(similarities, min(limit, len(filepaths)))

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            results.append({
                "filepath": filepaths[idx],
                "similarity": score
            })

        return results
