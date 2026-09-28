"""Write a compact, machine-readable audit before choosing validation/retrieval."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"


def main():
    frames = {name: pd.read_parquet(DATA / f"{name}.parquet") for name in
              ("train", "benchmark_queries", "benchmark_items")}
    report = {}
    for name, frame in frames.items():
        report[name] = {
            "rows": len(frame), "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing": frame.isna().sum().to_dict(),
            "unique": {c: int(frame[c].nunique()) for c in frame if "description" not in c},
            "sample": frame.head(3).to_dict("records"),
        }
    train, queries, items = (frames[n] for n in ("train", "benchmark_queries", "benchmark_items"))
    keys = [c for c in queries if c.startswith("search_")]
    norm = lambda f: f[keys].fillna("").astype(str)
    train_contexts = set(norm(train).itertuples(index=False, name=None))
    report["overlap"] = {
        "benchmark_query_text_seen_fraction": float(queries.search_query.isin(train.search_query).mean()),
        "benchmark_context_seen_fraction": sum(r in train_contexts for r in norm(queries).itertuples(index=False, name=None)) / len(queries),
        "benchmark_items_seen_fraction": float(items.item_id.isin(train.item_id).mean()),
        "train_pairs_in_benchmark_corpus_fraction": float(train.item_id.isin(items.item_id).mean()),
        "train_unique_contexts": len(train_contexts),
        "train_same_location_fraction": float((train.search_location_id == train.item_location_id).mean()),
    }
    out = ROOT / "reports/data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report["overlap"], indent=2))
    print(out)


if __name__ == "__main__":
    main()
