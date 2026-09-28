"""Strict submission validation. No ID normalization or implicit coercion."""
import argparse
import json
from pathlib import Path

import pandas as pd


def validate(answer, queries, items):
    if list(answer.columns) != ["query_id", "answer"]:
        raise ValueError("Columns must be exactly query_id,answer in that order")
    expected = set(queries.query_id)
    if len(answer) != len(queries) or answer.query_id.duplicated().any():
        raise ValueError("Wrong row count or duplicate query_id")
    if set(answer.query_id) != expected:
        raise ValueError("Missing or unexpected query IDs")
    if not answer.query_id.map(lambda x: isinstance(x, str) and len(x) == 16).all():
        raise ValueError("query_id must be a 16-character string")
    allowed = set(items.item_id)
    lengths = []
    for query_id, value in answer.itertuples(index=False, name=None):
        if not isinstance(value, str):
            raise ValueError(f"Non-string answer for {query_id}")
        ids = value.split(" ") if value else []
        if len(ids) > 50 or len(ids) != len(set(ids)):
            raise ValueError(f"Too many or repeated items for {query_id}")
        if any(item not in allowed for item in ids):
            raise ValueError(f"Unknown or malformed item for {query_id}")
        lengths.append(len(ids))
    return {"queries": len(answer), "min_candidates": min(lengths),
            "max_candidates": max(lengths), "valid": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("answer", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    answer = pd.read_csv(args.answer, dtype=str, keep_default_na=False)
    queries = pd.read_parquet(args.data / "benchmark_queries.parquet", columns=["query_id"])
    items = pd.read_parquet(args.data / "benchmark_items.parquet", columns=["item_id"])
    print(json.dumps(validate(answer, queries, items), indent=2))


if __name__ == "__main__":
    main()
