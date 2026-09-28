"""Shared data contracts and deterministic, context-disjoint validation."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd

QUERY_FIELDS = ["search_query", "search_location_id", "search_is_delivery_search",
                "search_infm_params_text", "search_category"]


def identifier(value):
    """Canonical metadata keys across integer/nullable-float Parquet columns."""
    if value is None or pd.isna(value):
        return "-1"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), ".15g")
    return str(value)


def normalize(value):
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text))


def context_keys(frame):
    """Hash all supplied query attributes, not just query text or query_id.

    IDs are never features. Numeric attributes are canonically formatted so an
    integer 1 and parquet nullable float 1.0 describe the same context.
    """
    parts = []
    for name in QUERY_FIELDS:
        if name in {"search_query", "search_infm_params_text"}:
            parts.append(frame[name].map(normalize))
        else:
            parts.append(frame[name].map(identifier))
    return pd.Series([
        hashlib.blake2b(json.dumps(row, ensure_ascii=False).encode(), digest_size=12).hexdigest()
        for row in zip(*parts)
    ], index=frame.index)


def load_data(directory):
    directory = Path(directory)
    train = pd.read_parquet(directory / "train.parquet")
    queries = pd.read_parquet(directory / "benchmark_queries.parquet")
    items = pd.read_parquet(directory / "benchmark_items.parquet")
    for frame in (train, items):
        if not frame.item_id.map(lambda x: isinstance(x, str)).all():
            raise ValueError("item_id must already be a string; do not silently coerce it")
        if not frame.item_id.str.fullmatch(r"[0-9a-f]{16}").all():
            raise ValueError("Invalid item_id")
    if items.item_id.duplicated().any():
        raise ValueError("Duplicate corpus item_id")
    if queries.query_id.duplicated().any():
        raise ValueError("Duplicate query_id")
    train["context_id"] = context_keys(train)
    queries["context_id"] = context_keys(queries)
    return train, queries, items


def split_queries(train, seed=20260928, n_fit=4000, n_dev=1000, n_test=1000, cold_fraction=.6):
    """All positives of a context stay together; no random pair-level split.

    fit: labels for learning the candidate selector; dev: model selection;
    test: untouched final evaluation. All three are excluded from retrieval's
    interaction statistics, including popularity and historical query indexing.
    """
    grouped = train.groupby("context_id", sort=True)
    queries = grouped[QUERY_FIELDS].first()
    queries["relevant"] = grouped.item_id.agg(lambda x: sorted(set(x)))
    queries = queries.reset_index().rename(columns={"context_id": "query_id"})
    rng = np.random.default_rng(seed)
    total = n_fit + n_dev + n_test
    if len(queries) <= total:
        raise ValueError("Not enough query groups for requested validation sizes")
    queries["normalized_text"] = queries.search_query.map(normalize)
    text_groups = queries.groupby("normalized_text", sort=True).indices
    texts = np.array(sorted(text_groups))
    n_cold = sum(round(count * cold_fraction) for count in (n_fit, n_dev, n_test))
    if len(texts) < n_cold:
        raise ValueError("Not enough distinct query texts for cold-query validation")
    cold_texts = set(rng.choice(texts, size=n_cold, replace=False))
    # Uniformly sampled texts prevent head queries from exhausting the history.
    # One full context per cold text is evaluated; all of that text's other
    # contexts are excluded from retrieval history as well.
    cold_rows = rng.permutation([rng.choice(text_groups[text]) for text in sorted(cold_texts)])
    warm_rows = rng.permutation(queries.index[~queries.normalized_text.isin(cold_texts)])
    result, cold_start, warm_start = {}, 0, 0
    for name, count in (("fit", n_fit), ("dev", n_dev), ("test", n_test)):
        cold_count = round(count * cold_fraction)
        warm_count = count - cold_count
        chosen = np.concatenate([cold_rows[cold_start:cold_start + cold_count],
                                 warm_rows[warm_start:warm_start + warm_count]])
        result[name] = queries.iloc[rng.permutation(chosen)].reset_index(drop=True)
        cold_start += cold_count
        warm_start += warm_count
    held_out = set().union(*(set(q.query_id) for q in result.values()))
    history = train.loc[~train.context_id.isin(held_out) &
                        ~train.search_query.map(normalize).isin(cold_texts)].copy()
    assert not set(history.context_id) & held_out
    assert not set(history.search_query.map(normalize)) & cold_texts
    return history, result


def validation_corpus(train, benchmark_items):
    """Full corpus, not sampled negatives; include every held-out positive.

    Benchmark metadata is authoritative for overlapping IDs. Using item text
    from held-out pairs is legitimate corpus construction; their interactions
    never enter retrieval history. This corpus is larger than the benchmark.
    """
    cols = list(benchmark_items.columns)
    return pd.concat([benchmark_items, train[cols]], ignore_index=True).drop_duplicates("item_id").reset_index(drop=True)
