"""Separate unretrieved positives from selector losses on development queries.

This reads training-derived development labels only, never benchmark labels.
Examples are deterministic diagnostics, not rules used during prediction.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from avito_retrieval.retrieval import topk


def main():
    payload = joblib.load('artifacts/pool300/validation_candidates.joblib')
    queries, results = payload['splits']['dev'], payload['results']['dev']
    scores = np.load('artifacts/catboost_pool300/classifier_dev_scores.npy')
    rows, offset = [], 0
    for q, result in zip(queries.itertuples(index=False), results):
        n = len(result['candidates'])
        pool = payload['ids'][result['candidates']]
        selected = set(pool[topk(scores[offset:offset+n], 50)])
        relevant = set(q.relevant)
        absent = relevant - set(pool)
        discarded = (relevant & set(pool)) - selected
        rows.append({'query': q.search_query, 'positives': len(relevant),
                     'retrieval_misses': len(absent), 'selector_misses': len(discarded),
                     'recall': len(relevant & selected)/len(relevant)})
        offset += n
    assert offset == len(scores)
    report = {'partition': 'development only', 'queries': len(rows),
              'recall_at_50': float(np.mean([r['recall'] for r in rows])),
              'queries_with_retrieval_misses': sum(r['retrieval_misses'] > 0 for r in rows),
              'queries_with_selector_misses': sum(r['selector_misses'] > 0 for r in rows),
              'missing_positive_items': {k: sum(r[k] for r in rows) for k in ('retrieval_misses', 'selector_misses')},
              'macro_recall_loss': {k: float(np.mean([r[k]/r['positives'] for r in rows]))
                                    for k in ('retrieval_misses', 'selector_misses')},
              'examples': {k: [r for r in rows if r[k]][:8] for k in ('retrieval_misses', 'selector_misses')}}
    Path('reports/error_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
