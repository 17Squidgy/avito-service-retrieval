"""Generate an independently validated CSV from a selected CatBoost experiment."""
import argparse
import json
import hashlib
from pathlib import Path

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .catboost_experiment import frame
from .data import load_data
from .features import FeatureBuilder
from .retrieval import Retriever, topk
from .submission import validate


def load_bundle(path):
    """Portable release uses native CatBoost weights and an inspectable JSON schema."""
    if path.suffix != ".json":
        return joblib.load(path)
    from catboost import CatBoostClassifier, CatBoostRanker
    bundle = json.loads(path.read_text())
    weights = path.parent / bundle["model_file"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != bundle["model_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    cls = CatBoostClassifier if bundle["kind"] == "classifier" else CatBoostRanker
    bundle["model"] = cls().load_model(str(weights))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("answer_round2.csv"))
    args = parser.parse_args()
    bundle = load_bundle(args.model)
    train, queries, items = load_data("data/raw")
    with threadpool_limits(limits=8):
        retriever = Retriever(items, train, bundle["pool"], bundle["dense_model"])
        results = retriever.retrieve(queries)
        if bundle["enhanced"]:
            results = FeatureBuilder(items).enrich(results, queries)
        answers = []
        for result in results:
            features = result["features"]
            if "feature_indices" in bundle:
                features = features[:, bundle["feature_indices"]]
            x = frame(features, bundle["feature_names"])
            kw = {"prediction_type": "RawFormulaVal"} if bundle["kind"] == "classifier" else {}
            scores = bundle["model"].predict(x, thread_count=8, **kw)
            answers.append(" ".join(retriever.ids[result["candidates"][topk(scores, 50)]]))
    answer = pd.DataFrame({"query_id": queries.query_id, "answer": answers})
    report = validate(answer, queries, items)
    answer.to_csv(args.output, index=False)
    validate(pd.read_csv(args.output, dtype=str, keep_default_na=False), queries, items)
    report["model"] = str(args.model)
    report_dir = Path("reports/round2")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "submission_validation.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.output}: {report}")


if __name__ == "__main__":
    main()
