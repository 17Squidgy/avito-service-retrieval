"""Build a larger fit/dev candidate cache; never evaluate the old reserved test."""
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from avito_retrieval.data import load_data, split_queries, validation_corpus
from avito_retrieval.retrieval import FEATURE_NAMES, Retriever


def main():
    output = Path("artifacts/pool300")
    output.mkdir(parents=True, exist_ok=True)
    train, _, items = load_data("data/raw")
    history, splits = split_queries(train)
    corpus = validation_corpus(train, items)
    with threadpool_limits(limits=8):
        retriever = Retriever(corpus, history, 300, "artifacts/models/multilingual-e5-small")
        results = {}
        for name in ("fit", "dev"):
            part = output / f"{name}.joblib"
            if part.exists():
                results[name] = joblib.load(part)
            else:
                results[name] = retriever.retrieve(splits[name])
                joblib.dump(results[name], part, compress=3)
        temporary = output / "validation_candidates.pending.joblib"
        joblib.dump({"results": results, "splits": {k: splits[k] for k in results},
            "ids": retriever.ids, "history_texts": set(retriever.history_texts), "pool": 300,
            "feature_names": FEATURE_NAMES, "dense_model": "artifacts/models/multilingual-e5-small"},
            temporary, compress=3)
        temporary.replace(output / "validation_candidates.joblib")


if __name__ == "__main__":
    main()
