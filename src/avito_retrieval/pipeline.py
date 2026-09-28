"""CLI: evaluate candidate generators/selector, then produce an offline submission."""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .data import load_data, normalize, split_queries, validation_corpus
from .retrieval import FEATURE_NAMES, Retriever, topk
from .submission import validate


def rank(result, method, model=None):
    features = result["features"]
    if method == "learned":
        scores = model.predict_proba(features)[:, 1]
    elif method == "rrf":
        scores = features[:, FEATURE_NAMES.index("rrf")]
    elif method == "heuristic":
        # An interpretable score baseline; held-out dev decides whether it helps.
        feature = lambda name: features[:, FEATURE_NAMES.index(name)]
        context = (.15 + .65 * feature("same_location") +
                   .6 * np.sqrt(feature("location_prior")) + .2 * np.sqrt(feature("category_prior")))
        scores = (feature("bm25_relative") + feature("char_relative") + feature("history_relative")) * context
    else:
        raise ValueError(method)
    return result["candidates"][topk(scores, 50)]


def evaluate(results, queries, ids, method, model=None, history_texts=None):
    values, unions = [], []
    generator_values = {}
    unique_contribution = {}
    for result, relevant in zip(results, queries.relevant):
        truth = set(relevant)
        chosen = set(ids[rank(result, method, model)])
        values.append(len(chosen & truth) / len(truth))
        unions.append(len(set(ids[result["candidates"]]) & truth) / len(truth))
        for name, rows in result["generators"].items():
            generator_values.setdefault(name, []).append(len(set(ids[rows[:50]]) & truth) / len(truth))
        hits = {name: set(ids[rows]) & truth for name, rows in result["generators"].items()}
        for name, found in hits.items():
            other_hits = set().union(*(value for key, value in hits.items() if key != name))
            unique_contribution.setdefault(name, []).append(len(found - other_hits) / len(truth))
    report = {"recall_at_50": float(np.mean(values)), "candidate_union_recall": float(np.mean(unions)),
              "queries": len(values), "mean_pool_size": float(np.mean([len(r["candidates"]) for r in results])),
              "generators_recall_at_50": {k: float(np.mean(v)) for k, v in generator_values.items()},
              "unique_union_recall_contribution": {k: float(np.mean(v)) for k, v in unique_contribution.items()}}
    if history_texts is not None:
        seen = queries.search_query.map(normalize).isin(history_texts).to_numpy()
        report["segments"] = {}
        masks = {"seen_text": seen, "unseen_text": ~seen}
        for category in queries.search_category.unique():
            masks[f"category={category}"] = (queries.search_category == category).to_numpy()
        for name, mask in masks.items():
            if mask.any():
                report["segments"][name] = {"queries": int(mask.sum()),
                                            "recall_at_50": float(np.array(values)[mask].mean())}
    return report


def fit_selector(results, queries, ids):
    xx, yy, weights = [], [], []
    for result, relevant in zip(results, queries.relevant):
        labels = np.isin(ids[result["candidates"]], relevant).astype(np.int8)
        xx.append(result["features"])
        yy.append(labels)
        # Equal positive mass per query approximates macro recall; positives
        # outside the retrieved pool stay misses and are never injected.
        weights.append(np.where(labels == 1, 50 / len(relevant), 1).astype(np.float32))
    x, y, weight = np.concatenate(xx), np.concatenate(yy), np.concatenate(weights)
    print(f"Selector: {len(y):,} candidates, {y.sum():,} positives", flush=True)
    model = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=31,
        learning_rate=.08, l2_regularization=10, min_samples_leaf=50,
        early_stopping=False, random_state=20260928)
    model.fit(x, y, sample_weight=weight)
    return model


def run_evaluate(args):
    started = time.time()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    cache = args.artifacts / "validation_candidates.joblib"
    if args.reuse_candidates and cache.exists():
        payload = joblib.load(cache)
    else:
        train, _, items = load_data(args.data)
        history, splits = split_queries(train, n_fit=args.fit_queries,
                                        n_dev=args.dev_queries, n_test=args.test_queries)
        corpus = validation_corpus(train, items)
        retriever = Retriever(corpus, history, args.pool, args.dense_model)
        results = {name: retriever.retrieve(queries) for name, queries in splits.items()}
        payload = {"results": results, "splits": splits, "ids": retriever.ids,
                   "history_texts": set(retriever.history_texts), "pool": args.pool,
                   "feature_names": FEATURE_NAMES, "dense_model": args.dense_model}
        joblib.dump(payload, cache, compress=3)
    if payload["feature_names"] != FEATURE_NAMES:
        raise ValueError("Candidate cache schema differs; regenerate without --reuse-candidates")
    results, splits, ids = (payload[k] for k in ("results", "splits", "ids"))
    model = fit_selector(results["fit"], splits["fit"], ids)
    reports = {}
    for method in ("rrf", "heuristic", "learned"):
        reports[method] = evaluate(results["dev"], splits["dev"], ids, method, model, payload["history_texts"])
        print(f"Dev {method}: {reports[method]['recall_at_50']:.5f}", flush=True)
    best = max(reports, key=lambda method: reports[method]["recall_at_50"])
    # Freeze selection before looking at the final test partition.
    test_report = (evaluate(results["test"], splits["test"], ids, best, model, payload["history_texts"])
                   if args.final_test else {"status": "withheld; pass --final-test after selecting the final configuration"})
    report = {"selected_method": best, "dev": reports, "test": test_report,
              "corpus_items": len(ids), "fit_queries": len(splits["fit"]),
              "elapsed_seconds": time.time() - started,
              "validation_note": "Context-disjoint labels; full union of training and benchmark items. Larger corpus than benchmark; no temporal split is possible."}
    (args.reports / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    joblib.dump({"model": model, "method": best, "feature_names": FEATURE_NAMES,
                 "pool": payload["pool"], "dense_model": payload.get("dense_model")}, args.artifacts / "selector.joblib")
    print(json.dumps({"selected": best, "test": test_report}, indent=2, ensure_ascii=False))


def run_predict(args):
    bundle = joblib.load(args.artifacts / "selector.joblib")
    if bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError("Selector feature schema differs from code")
    train, queries, items = load_data(args.data)
    retriever = Retriever(items, train, bundle["pool"], bundle.get("dense_model"))
    results = retriever.retrieve(queries)
    answer = pd.DataFrame({"query_id": queries.query_id,
        "answer": [" ".join(retriever.ids[rank(r, bundle["method"], bundle["model"])]) for r in results]})
    report = validate(answer, queries, items)
    answer.to_csv(args.output, index=False)
    # Validate the actual on-disk serialization, including leading zeros.
    validate(pd.read_csv(args.output, dtype=str, keep_default_na=False), queries, items)
    (args.reports / "submission_validation.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.output}: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate", "predict"])
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--fit-queries", type=int, default=4000)
    parser.add_argument("--dev-queries", type=int, default=1000)
    parser.add_argument("--test-queries", type=int, default=1000)
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument("--dense-model", type=str, default=None, help="Local E5-small model directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    # Sandboxed macOS may hide sysctl physical-core counts from joblib.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.threads))
    with threadpool_limits(limits=args.threads):
        (run_evaluate if args.command == "evaluate" else run_predict)(args)


if __name__ == "__main__":
    main()
