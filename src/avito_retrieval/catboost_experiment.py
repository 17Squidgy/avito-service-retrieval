"""Compare selectors using frozen candidate caches and development labels only.

Checkpoint selection uses end-to-end macro Recall@50, whose denominator includes
positives absent from the pool. CatBoost's within-pool metrics cannot do that.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from .data import load_data, validation_corpus
from .features import EXTRA_NAMES, FeatureBuilder
from .retrieval import FEATURE_NAMES, topk


def frame(x, names):
    result = pd.DataFrame(x, columns=names)
    for name in names:
        if name.startswith("cat_"):
            result[name] = result[name].astype(np.int64).astype(str)
    return result


def arrays(results, queries, ids):
    x = np.concatenate([r["features"] for r in results])
    labels, weights, groups = [], [], []
    offsets = [0]
    for i, (r, relevant) in enumerate(zip(results, queries.relevant)):
        y = np.isin(ids[r["candidates"]], relevant).astype(np.int8)
        labels.append(y)
        weights.append(np.where(y, 50 / len(relevant), 1).astype(np.float32))
        groups.append(np.full(len(y), i, dtype=np.int64))
        offsets.append(offsets[-1] + len(y))
    return x, np.concatenate(labels), np.concatenate(weights), np.concatenate(groups), np.array(offsets)


def query_recalls(scores, results, queries, ids, offsets):
    return np.array([len(set(ids[r["candidates"][topk(scores[offsets[i]:offsets[i+1]], 50)]]) & set(relevant)) / len(relevant)
        for i, (r, relevant) in enumerate(zip(results, queries.relevant))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/dense/validation_candidates.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/catboost"))
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--numeric-only", action="store_true", help="Ablate categorical metadata from enhanced features")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--kinds", nargs="+", default=["classifier", "ranker"])
    parser.add_argument("--ranking-loss", default="YetiRank:mode=NDCG;top=50")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = joblib.load(args.cache)
    results = {name: payload["results"][name] for name in ("fit", "dev")}
    queries = payload["splits"]
    names = FEATURE_NAMES.copy()
    if args.enhanced:
        train, _, items = load_data("data/raw")
        corpus = validation_corpus(train, items)
        assert np.array_equal(corpus.item_id.to_numpy(), payload["ids"])
        builder = FeatureBuilder(corpus)
        for name in results:
            results[name] = builder.enrich(results[name], queries[name])
        names += EXTRA_NAMES
        del builder, train, corpus, items
    feature_indices = list(range(len(names)))
    if args.numeric_only:
        feature_indices = [i for i, name in enumerate(names) if not name.startswith("cat_")]
        names = [names[i] for i in feature_indices]
        results = {part: [{**r, "features": r["features"][:, feature_indices]} for r in rows]
                   for part, rows in results.items()}
    train_x, train_y, weights, group, _ = arrays(results["fit"], queries["fit"], payload["ids"])
    dev_x, _, _, _, offsets = arrays(results["dev"], queries["dev"], payload["ids"])
    train_x, dev_x = frame(train_x, names), frame(dev_x, names)
    cats = [name for name in names if name.startswith("cat_")]
    # Fixed development halves report sensitivity; neither is an untouched test.
    order = np.random.default_rng(731).permutation(len(queries["dev"]))
    tune, confirm = order[:len(order)//2], order[len(order)//2:]
    reports = {}
    for kind in args.kinds:
        started = time.time()
        common = dict(iterations=args.iterations, depth=6, learning_rate=.06,
                      l2_leaf_reg=10, random_seed=20260928, thread_count=args.threads,
                      allow_writing_files=False, verbose=50)
        if kind == "classifier":
            model = CatBoostClassifier(loss_function="Logloss", **common)
            pool = Pool(train_x, train_y, weight=weights, cat_features=cats)
        else:
            model = CatBoostRanker(loss_function=args.ranking_loss, **common)
            pool = Pool(train_x, train_y, group_id=group, cat_features=cats)
        model.fit(pool)
        checkpoints = []
        best = None
        for trees in range(50, args.iterations + 1, 50):
            kw = {"prediction_type": "RawFormulaVal"} if kind == "classifier" else {}
            scores = model.predict(dev_x, ntree_end=trees, thread_count=args.threads, **kw)
            recall = query_recalls(scores, results["dev"], queries["dev"], payload["ids"], offsets)
            record = {"trees": trees, "tune_recall": float(recall[tune].mean()),
                      "confirmation_recall": float(recall[confirm].mean()), "dev_recall": float(recall.mean())}
            checkpoints.append(record)
            if best is None or record["tune_recall"] > best["tune_recall"]:
                best = record
        model.shrink(best["trees"])
        kw = {"prediction_type": "RawFormulaVal"} if kind == "classifier" else {}
        selected_scores = model.predict(dev_x, thread_count=args.threads, **kw)
        np.save(args.output / f"{kind}_dev_scores.npy", selected_scores)
        np.save(args.output / f"{kind}_dev_recalls.npy", query_recalls(
            selected_scores, results["dev"], queries["dev"], payload["ids"], offsets))
        model.save_model(str(args.output / f"{kind}.cbm"))
        bundle = {"model": model, "kind": kind, "feature_names": names, "enhanced": args.enhanced,
                  "feature_indices": feature_indices,
                  "pool": payload["pool"], "dense_model": payload.get("dense_model"), "cache": str(args.cache)}
        joblib.dump(bundle, args.output / f"{kind}.joblib")
        reports[kind] = {"selected": best, "checkpoints": checkpoints, "seconds": time.time() - started,
                         "loss": "Logloss" if kind == "classifier" else args.ranking_loss}
        (args.output / "report.json").write_text(json.dumps(reports, indent=2))
        print(kind, json.dumps(reports[kind]), flush=True)


if __name__ == "__main__":
    main()
