"""Precompute content-cached dense embeddings separately from evaluation."""
from pathlib import Path
import torch
from avito_retrieval.data import load_data, validation_corpus
from avito_retrieval.dense import DenseEncoder


def main():
    root = Path(__file__).resolve().parents[1]
    print(f"MPS available: {torch.backends.mps.is_available()}", flush=True)
    torch.set_num_threads(6)
    train, _, items = load_data(root / "data/raw")
    corpus = validation_corpus(train, items)
    encoder = DenseEncoder(root / "artifacts/models/multilingual-e5-small", root / "artifacts/models/e5_cache")
    if encoder.device == "cpu":
        raise RuntimeError("GPU unavailable. Use pipeline --dense-model to explicitly permit CPU encoding.")
    vectors = encoder.items(corpus)
    print(f"Cached embeddings: {vectors.shape}", flush=True)


if __name__ == "__main__":
    main()
