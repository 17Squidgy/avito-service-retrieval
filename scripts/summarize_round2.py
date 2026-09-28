"""Summarize development-only experiments and recommend a conservative winner.

The original reserved test is never scored here. Confidence intervals are paired
development bootstraps, not independent or multiple-comparison-adjusted evidence.
"""
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from avito_retrieval.catboost_experiment import arrays, query_recalls


def prepare_original_diagnostics():
    """Rebuild comparison arrays from saved models, without reserved-test scoring."""
    directory = ROOT / "artifacts/original"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "dev_recalls.npy").exists() and (directory / "ensemble_dev_recalls.npy").exists():
        return
    payload = joblib.load(ROOT / "artifacts/dense/validation_candidates.joblib")
    results, queries = payload["results"]["dev"], payload["splits"]["dev"]
    model_path = directory / "selector.joblib"
    if not model_path.exists():
        shutil.copy2(ROOT / "artifacts/dense/selector.joblib", model_path)
    model = joblib.load(model_path)["model"]
    x, _, _, _, offsets = arrays(results, queries, payload["ids"])
    with threadpool_limits(limits=4):
        scores = model.predict_proba(x)[:, 1]
    np.save(directory / "dev_scores.npy", scores)
    np.save(directory / "dev_recalls.npy", query_recalls(scores, results, queries, payload["ids"], offsets))
    other = np.load(ROOT / "artifacts/catboost/classifier_dev_scores.npy")
    fused = np.zeros_like(scores)
    for start, end in zip(offsets[:-1], offsets[1:]):
        for source in (scores, other):
            order = np.argsort(-source[start:end], kind="stable")
            fused[start + order] += 1 / (60 + np.arange(1, end-start+1))
    np.save(directory / "ensemble_dev_recalls.npy", query_recalls(fused, results, queries, payload["ids"], offsets))

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ["catboost", "catboost_enhanced", "catboost_numeric", "catboost_numeric800",
               "catboost_pool300", "catboost_pool300_numeric", "catboost_softmax"]


def main():
    prepare_original_diagnostics()
    baseline = np.load(ROOT / "artifacts/original/dev_recalls.npy")
    order = np.random.default_rng(731).permutation(len(baseline))
    tune, confirmation = order[:len(order)//2], order[len(order)//2:]
    bootstrap = np.random.default_rng(419).integers(0, len(baseline), size=(2000, len(baseline)))
    records = []

    def add(name, recalls, backend, model_path):
        diff = recalls - baseline
        ci = np.quantile(diff[bootstrap].mean(axis=1), [.025, .975])
        records.append({"name": name, "dev_recall": float(recalls.mean()),
            "tune_recall": float(recalls[tune].mean()), "confirmation_recall": float(recalls[confirmation].mean()),
            "delta": float(diff.mean()), "paired_dev_delta_ci95": ci.tolist(),
            "backend": backend, "model": model_path})

    add("original_hgb", baseline, "hgb", "artifacts/original/selector.joblib")
    large_recalls = ROOT / "artifacts/pool300/dev_recalls.npy"
    if not large_recalls.exists():
        payload = joblib.load(ROOT / "artifacts/pool300/validation_candidates.joblib")
        model = joblib.load(ROOT / "artifacts/pool300/selector.joblib")["model"]
        results, queries = payload["results"]["dev"], payload["splits"]["dev"]
        x, _, _, _, offsets = arrays(results, queries, payload["ids"])
        with threadpool_limits(limits=4):
            scores = model.predict_proba(x)[:, 1]
        np.save(large_recalls, query_recalls(scores, results, queries, payload["ids"], offsets))
        del payload, results, queries, x, model
    add("hgb_pool300", np.load(large_recalls), "hgb", "artifacts/pool300/selector.joblib")
    for experiment in EXPERIMENTS:
        directory = ROOT / "artifacts" / experiment
        report = json.loads((directory / "report.json").read_text())
        for kind in report:
            add(f"{experiment}/{kind}", np.load(directory / f"{kind}_dev_recalls.npy"),
                "catboost", f"artifacts/{experiment}/{kind}.joblib")
    add("original_catboost_rrf", np.load(ROOT / "artifacts/original/ensemble_dev_recalls.npy"), "diagnostic", None)
    # Require improvement on both development halves and a nontrivial overall
    # gain. A 0.01 percentage-point tie is not a reason to replace a submission.
    original = records[0]
    eligible = [r for r in records if r["backend"] != "diagnostic" and
                r["dev_recall"] >= original["dev_recall"] + .002 and
                r["confirmation_recall"] >= original["confirmation_recall"] and
                r["tune_recall"] > original["tune_recall"]]
    winner = max(eligible, key=lambda r: r["tune_recall"]) if eligible else original
    summary = {"experiments": records, "recommended": winner,
               "selection_rule": "At least +0.002 dev recall, improved tuning half, no confirmation-half regression; choose highest tuning recall among eligible models.",
               "test_note": "No new reserved-test evaluation. Bootstrap intervals are development diagnostics after model selection."}
    (ROOT / "reports/round2/comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
