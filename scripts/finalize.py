"""Freeze the development winner, evaluate its reserved test, and promote it.

This evaluates the existing model without retraining it. The test result does
not participate in model selection. Run after completing development experiments.
"""
import json
import os
from pathlib import Path
import shutil

import joblib
from threadpoolctl import threadpool_limits

from avito_retrieval.pipeline import evaluate
from avito_retrieval.retrieval import FEATURE_NAMES


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
    reports = {name: json.loads((root / f"reports/{name}/validation.json").read_text())
               for name in ("sparse", "dense")}
    scores = {name: report["dev"][report["selected_method"]]["recall_at_50"]
              for name, report in reports.items()}
    winner = max(scores, key=scores.get)
    print(f"Frozen development winner: {winner} {scores[winner]:.6f}", flush=True)
    directory = root / "artifacts" / winner
    bundle = joblib.load(directory / "selector.joblib")
    cache = joblib.load(directory / "validation_candidates.joblib")
    if bundle["feature_names"] != FEATURE_NAMES or cache["feature_names"] != FEATURE_NAMES:
        raise ValueError("Feature schema differs; regenerate the experiment")
    with threadpool_limits(limits=8):
        test = evaluate(cache["results"]["test"], cache["splits"]["test"], cache["ids"],
                        bundle["method"], bundle["model"], cache["history_texts"])
    report = reports[winner]
    report["test"] = test
    report["selected_experiment"] = winner
    report["development_comparison"] = scores
    report["selection_note"] = "Experiment and model frozen on development scores before reserved-test evaluation."
    (root / "reports/validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    shutil.copy2(directory / "selector.joblib", root / "artifacts/selector.joblib")
    print(json.dumps({"experiment": winner, "development": scores,
                      "test_recall_at_50": test["recall_at_50"],
                      "test_candidate_union_recall": test["candidate_union_recall"]}, indent=2))


if __name__ == "__main__":
    main()
