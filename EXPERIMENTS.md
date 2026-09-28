# Avito service candidate generation

Retrieve up to 50 service advertisements for each Russian search query. The target
is **macro Recall@50**: each query contributes equally, irrespective of how many
positive advertisements it has. The ordering of the final 50 does not matter.

The current submission combines sparse retrieval, multilingual E5-small, and a
CatBoost selector over a larger candidate pool. It achieves **0.8585 development
Recall@50**, compared with **0.8553** for the original HGB selector. The gain is
small and uncertain; its paired development confidence interval includes zero.
The original model scored **0.8444 on the reserved local test**; the new model has
not been scored there. These are local scores, not leaderboard results.
The Russian solution description is in [SOLUTION.md](SOLUTION.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps --no-build-isolation
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/inspect_data.py
.venv/bin/python scripts/download_model.py
```

The setup script downloads the organizer's archive and checks its SHA-256 before
extracting the three Parquet files into `data/raw/`. Data, models, and generated
submissions are ignored by Git. Keep all IDs as strings. Never edit the submission
in a spreadsheet application.

## Reproduce the original baseline

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m avito_retrieval.pipeline evaluate \
  --artifacts artifacts/sparse --reports reports/sparse
PYTHONPATH=src .venv/bin/python -m avito_retrieval.pipeline evaluate \
  --dense-model artifacts/models/multilingual-e5-small \
  --artifacts artifacts/dense --reports reports/dense
# Choose using dev only, freeze the saved model, then evaluate reserved test:
.venv/bin/python scripts/finalize.py
PYTHONPATH=src .venv/bin/python -m avito_retrieval.pipeline predict --output answer_original.csv
PYTHONPATH=src .venv/bin/python -m avito_retrieval.submission answer_original.csv
```

Inference runs locally without external APIs. The trained selector is saved to
`artifacts/selector.joblib`; candidate caches and diagnostics are saved separately.

`requirements-lock.txt` records the tested environment (Python 3.14 on macOS ARM).
Use `.venv/bin/pip install -r requirements-lock.txt` to install those exact package
versions, including the optional neural dependencies, before installing this
project with `pip install -e . --no-deps`.

For the original model, run `pipeline predict` and the submission validator.
`artifacts/selector.joblib` still stores that original HGB model.
`artifacts/original/answer.csv` preserves its original submission;
`answer_sparse.csv` is the lexical baseline. No official submission attempts
were used during development.

## Current CatBoost submission

[Round 2 experiments](reports/round2/README.md) document CatBoost losses, richer
features, larger pools, and a local HyDE pilot, with all training commands.
The current model is `artifacts/catboost_pool300/classifier.joblib` (350 trees,
51 features, 300 items per generator). Generate its submission explicitly:

```bash
.venv/bin/python -m avito_retrieval.catboost_predict artifacts/catboost_pool300/classifier.joblib --output answer.csv
.venv/bin/python -m avito_retrieval.submission answer.csv
```

`answer.csv` and `answer_catboost.csv` contain this variant. See
[comparison.json](reports/round2/comparison.json) for model-selection results.
The old reserved test was not rescored during this round. HyDE is experimental
and is not needed to reproduce the selected submission.

| Configuration | Dev Recall@50 | Dev pool recall | Reserved test Recall@50 |
|---|---:|---:|---:|
| Sparse + reciprocal-rank fusion | 0.7072 | 0.8784 | — |
| Sparse + learned selector | 0.8246 | 0.8784 | — |
| Sparse + E5 + learned selector | **0.8553** | **0.9187** | **0.8444** |

The original model's reserved-test pool recall is 0.9253, with an average of 584
candidates before selecting 50. Full metrics and generator contributions are in
`reports/validation.json`; experiment-specific reports are in `reports/sparse/`
and `reports/dense/`. The non-winning experiments were not scored on reserved test; the round 2
CatBoost submission also has no reserved-test score.

On a GPU machine, `.venv/bin/python scripts/encode_items.py` can precompute item
embeddings before the dense experiment. It requires MPS or CUDA. The regular
pipeline also supports CPU encoding. Subsequent runs reuse the content-based cache.

## Retrieval approach

The initial system combines several sources of candidates:

1. **Russian Snowball BM25:** title/service parameters and description are indexed
   separately. Title matches receive higher weight. Descriptions are capped at
   5,000 characters to limit boilerplate and indexing cost.
2. **Character TF-IDF:** 3–5 character n-grams from titles
   tolerate spelling differences and inflection.
3. **Historical query neighbours:** 2–5 character n-grams find similar training
   queries; their selected items vote with similarity and frequency weights.
4. **Context-aware variants:** soft location and category priors learned exclusively
   from retrieval-training interactions complement unrestricted retrieval. Exact
   location and nearby-location BM25 lists add geographic coverage; coordinates
   produce a log-distance feature for the selector.
5. **Contextual popularity:** supplies fallback candidates for sparse queries.

The original baseline uses up to 100 items per generator; the current CatBoost
variant uses 300. The baseline deduplicated union is scored by
a histogram gradient boosting selector using retrieval scores, relative scores,
context compatibility, historical popularity, and item metadata. Reciprocal-rank
fusion and a transparent weighted score serve as comparison baselines. Dev
Recall@50 chooses the final method. Candidate-union recall measures the ceiling
available to the selector.

Unobserved query–item pairs are **not guaranteed negatives**. They are used as
weak negatives for the selector; positives outside the candidate pool are never
injected, so evaluation includes retrieval misses.

## Validation protocol and limitations

Query context includes all five `search_*` attributes. Text is normalized for
case, punctuation, whitespace, and `ё/е`; identical normalized contexts remain
together. A fixed seed splits complete contexts into selector training (4,000),
development (1,000), and reserved test (1,000). For 60% of each partition, complete
query texts are held out, with one context sampled per text. All other contexts
of those cold texts are also excluded from history. This approximates the
benchmark's 63% unseen-text share; ordinary random context splits would overvalue
historical lookup. Every interaction for the selected evaluation contexts
is removed before calculating historical-query, popularity, location, or category
statistics. All of a context's unique positive item IDs form its relevance set.

Validation searches the full union of benchmark and training items, with benchmark
metadata taking precedence. This guarantees that all held-out positives exist in
the corpus without selecting queries based on benchmark membership. Its larger
corpus makes scores non-comparable to the official leaderboard in absolute terms.
Item text from held-out pairs is corpus metadata, not an interaction feature.

There are no timestamps, so a temporal split is unavailable. Seen-text and
unseen-text segments are reported separately. Benchmark overlap and distribution
differences are audited in `reports/data_audit.json`. Do not use official attempts
to reconstruct hidden labels or add rules for specific query IDs.

During final inference, interaction statistics are rebuilt from all training pairs
and search is restricted to `benchmark_items.parquet`.

The initial data audit found:

| Quantity | Value |
|---|---:|
| Training pairs | 497,673 |
| Distinct training items | 344,825 |
| Benchmark queries / items | 2,452 / 189,212 |
| Full local validation corpus | 515,895 |
| Benchmark query texts seen in training (raw text) | 36.99% |
| Benchmark items seen in training | 9.59% |
| Training positives sharing the query location | 83.10% |

The development partition has 34.4% seen query texts and an average of 1.139
positive items per query. Benchmark corpus overlap is not the same as overlap of
its unknown positive labels; do not infer leaderboard recall from it.

## Files

| Path | Purpose |
|---|---|
| `src/avito_retrieval/data.py` | Data contracts and query-level splits |
| `src/avito_retrieval/retrieval.py` | Generators and candidate features |
| `src/avito_retrieval/pipeline.py` | Evaluation, selector training, prediction |
| `src/avito_retrieval/submission.py` | Strict CSV validation |
| `scripts/inspect_data.py` | Dataset audit |
| `reports/validation.json` | Measured generator and selector quality |

## Open-source components

The implementation uses NumPy (BSD), SciPy (BSD), pandas (BSD), PyArrow (Apache
2.0), scikit-learn (BSD), Snowball stemmer (BSD), and tqdm (MPL/MIT). BM25,
character TF-IDF, reciprocal-rank fusion, and gradient boosting are established
public methods; the contest implementation is written in this repository.

Optional neural experiments use PyTorch and Sentence Transformers. Any pretrained
model actually used in a submission must be recorded with its exact revision,
license, input formatting, and local download instructions. No neural model is
required for the sparse baseline.

The included semantic generator uses [multilingual E5-small](https://huggingface.co/intfloat/multilingual-e5-small),
MIT license, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. It encodes `query:`
and `passage:` prefixed inputs, uses normalized embeddings with dot-product search,
and truncates at 96 tokens. Passages contain title, service parameters, and the
start of the description. The local encoder chooses MPS, CUDA, or CPU according to
availability. Item embeddings are cached by text content and model revision.

```bash
.venv/bin/pip install -e '.[dense]'
.venv/bin/python scripts/download_model.py
PYTHONPATH=src .venv/bin/python -m avito_retrieval.pipeline evaluate \
  --dense-model artifacts/models/multilingual-e5-small \
  --artifacts artifacts/dense --reports reports/dense
```

Keep experiment outputs in separate directories. `--reuse-candidates` explicitly
reuses a saved experiment's data and features; omit it after any generator, data,
or split change. Preserve the model directory when moving trained artifacts to
another machine, or update the model path in the artifact accordingly.

Round 2 also uses CatBoost 1.2.10 (Apache-2.0). The optional Qwen HyDE model,
its revision, license, and generation settings are recorded in the round 2 report.
