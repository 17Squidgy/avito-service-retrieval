"""Optional E5 embeddings. Local weights only; content-addressed embedding cache."""
import hashlib
import json
from pathlib import Path

import numpy as np

REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class DenseEncoder:
    def __init__(self, model_path, cache_dir):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(str(model_path), device=self.device, local_files_only=True)
        self.model.max_seq_length = 96
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"E5 local encoder on {self.device}; max_length=96", flush=True)

    def encode(self, texts):
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True).astype(np.float32)

    def items(self, frame):
        # Put the title first; metadata and a short description add disambiguation.
        texts = ("passage: " + frame.item_title_raw.fillna("") + ". " +
                 frame.item_infm_params_text.fillna("").str.slice(0, 180) + ". " +
                 frame.item_description_raw.fillna("").str.slice(0, 400)).tolist()
        keys = [hashlib.sha256((REVISION + ":96:" + text).encode()).hexdigest() for text in texts]
        key_path, vector_path = self.cache_dir / "keys.json", self.cache_dir / "vectors.npy"
        old_keys = json.loads(key_path.read_text()) if key_path.exists() and vector_path.exists() else []
        old_vectors = np.load(vector_path, mmap_mode="r")[:len(old_keys)] if old_keys else None
        if old_keys and len(old_vectors) != len(old_keys):
            raise ValueError("Embedding cache is incomplete; regenerate its keys/vectors pair")
        lookup = {key: i for i, key in enumerate(old_keys)}
        missing = {}
        for key, text in zip(keys, texts):
            if key not in lookup:
                missing[key] = text
        if missing:
            print(f"Encoding {len(missing):,} unique item texts", flush=True)
            new_vectors = self.encode(list(missing.values()))
            vectors = np.concatenate([old_vectors, new_vectors]) if old_keys else new_vectors
            all_keys = old_keys + list(missing)
            # Replace complete files, never expose partially written arrays.
            # If interrupted between replacements, old keys still address the
            # unchanged prefix of the new vector array (sliced when loading).
            temporary_vectors = self.cache_dir / "vectors.pending.npy"
            temporary_keys = self.cache_dir / "keys.pending.json"
            np.save(temporary_vectors, vectors)
            temporary_keys.write_text(json.dumps(all_keys))
            temporary_vectors.replace(vector_path)
            temporary_keys.replace(key_path)
            lookup = {key: i for i, key in enumerate(all_keys)}
        else:
            vectors = old_vectors
        return np.asarray(vectors[[lookup[key] for key in keys]], dtype=np.float32)

    def queries(self, frame):
        texts = ("query: " + frame.search_query.fillna("") + " " +
                 frame.search_infm_params_text.fillna("").str.slice(0, 180)).tolist()
        return self.encode(texts)
