# CatBoost, larger candidate pools, and HyDE

This round uses the existing 4,000 fit and 1,000 development query contexts.
Retrieval history excludes evaluation interactions as before. The previously
reported reserved-test result is not reused to choose these models, and no new
model is evaluated on that test. Official submission attempts remain unused.

`comparison.json` records all end-to-end Recall@50 results and the selected
artifact. Candidate-pool recall is an upper bound, not the submission metric.
Development halves (seed 731, 500 queries each) provide a sensitivity diagnostic:
choose tree count on the first half and report the second. Both belong to an
already-used development set; neither is an independent final holdout. Paired
bootstrap intervals use 2,000 development resamples and do not correct for model
selection or multiple comparisons.

## Experiments

- CatBoostClassifier with weighted Logloss; observed positives get weight
  `50 / number_of_relevant_items`, unobserved pairs get weight 1.
- CatBoostRanker with `YetiRank:mode=NDCG;top=50`, and separately `QuerySoftMax`.
  Both optimize surrogate losses. Checkpoints are selected using actual macro
  Recall@50, including relevant items missing from the candidate pool.
- Original 27 features versus 51 features: generator ranks and membership,
  title/filter overlap, title phrase match, description length, rating-filter
  compatibility, and three categorical category/microcategory attributes.
  A 48-feature ablation removes categorical attributes.
- 100 versus 300 results per generator. The latter increases mean pool size to
  1,526 and development pool recall from 0.9187 to 0.9572. The original HGB
  selector on this larger pool obtains 0.8547 Recall@50.
- One fixed equal reciprocal-rank fusion of original HGB and base CatBoost.
- A local HyDE pilot using Qwen2.5-1.5B-Instruct on 300 randomly chosen development
  queries (seed 913). A generated service-ad description is encoded as an E5
  passage, either alone or blended equally with the original query embedding.
  Each variant adds global, location-boosted, and same-location top-100 lists.

HyDE raises pilot union recall from 0.9150 to 0.9217. Against the larger existing
pool, generated descriptions add 0.0033 recall; blended embeddings add zero.
Some descriptions change the service intent or switch into the customer's voice.
This tested configuration is not promoted: the small pool gain has no measured
final-top-50 benefit yet. This does not rule out stronger generators or prompts.

Temporal walk-forward is unavailable because the supplied schema has no event
timestamps or documented chronological row order. Arbitrarily treating row order
as time would create an unsupported validation claim. The existing text/context
holdouts remain the validation protocol.

## Reproduce

Run the initial dense evaluation from the root README first to create its frozen
candidate cache. All commands below run from the repository root. The HyDE pilot
requires MPS or CUDA; CatBoost experiments train on CPU. Keep separate output
folders, and remove/rebuild candidate caches if data, splits, or generators change.

```bash
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m avito_retrieval.catboost_experiment
.venv/bin/python -m avito_retrieval.catboost_experiment --enhanced --output artifacts/catboost_enhanced
.venv/bin/python -m avito_retrieval.catboost_experiment --enhanced --numeric-only --kinds classifier --output artifacts/catboost_numeric
.venv/bin/python -m avito_retrieval.catboost_experiment --enhanced --numeric-only --kinds classifier --iterations 800 --threads 4 --output artifacts/catboost_numeric800
.venv/bin/python -m avito_retrieval.catboost_experiment --kinds ranker --ranking-loss QuerySoftMax --threads 4 --output artifacts/catboost_softmax
.venv/bin/python scripts/build_larger_pool.py
.venv/bin/python -m avito_retrieval.pipeline evaluate --reuse-candidates --artifacts artifacts/pool300 --reports reports/round2/hgb_pool300 --threads 4
.venv/bin/python -m avito_retrieval.catboost_experiment --cache artifacts/pool300/validation_candidates.joblib --enhanced --kinds classifier --threads 4 --output artifacts/catboost_pool300
.venv/bin/python -m avito_retrieval.catboost_experiment --cache artifacts/pool300/validation_candidates.joblib --enhanced --numeric-only --kinds classifier --iterations 800 --threads 4 --output artifacts/catboost_pool300_numeric
.venv/bin/python scripts/download_hyde_model.py
.venv/bin/python scripts/hyde_pilot.py
.venv/bin/python scripts/summarize_round2.py
```

The comparison script also rebuilds original-model and fixed-fusion development
diagnostics when they are absent. Its promotion rule requires at least 0.002
overall development gain, improvement on the tuning half, and no regression on
the confirmation half; among eligible models it chooses the best tuning recall.
This is a practical selection rule, not a statistical significance test.

CatBoost 1.2.10 is Apache-2.0 licensed. The optional HyDE generator is
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
Apache-2.0, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Generation uses
its chat template, greedy decoding, at most 72 new tokens, and a Russian prompt
requesting at most 35 words without invented prices or locations. Downloads occur
only during setup; inference uses local files and no external model API.

## Measured development results

| Experiment | Recall@50 |
|---|---:|
| original_hgb | 0.855333 |
| hgb_pool300 | 0.854700 |
| catboost/classifier | 0.850433 |
| catboost/ranker | 0.817117 |
| catboost_enhanced/classifier | 0.846733 |
| catboost_enhanced/ranker | 0.824417 |
| catboost_numeric/classifier | 0.855433 |
| catboost_numeric800/classifier | 0.853400 |
| catboost_pool300/classifier | 0.858490 |
| catboost_pool300_numeric/classifier | 0.854857 |
| catboost_softmax/ranker | 0.839200 |
| original_catboost_rrf | 0.850600 |

Selected: `artifacts/catboost_pool300/classifier.joblib`. Its CSV is `answer.csv`
(and the identical `answer_catboost.csv`). The original submission is preserved
at `artifacts/original/answer.csv`. The selected improvement is 0.00316 absolute
recall; paired development bootstrap 95% interval for its difference from the
original is approximately [-0.00835, 0.01501], so superiority is uncertain.
