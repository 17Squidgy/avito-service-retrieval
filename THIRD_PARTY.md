# Open-source components

The contest-specific implementation is in `src/avito_retrieval/`. It combines
established BM25, character TF-IDF, reciprocal-rank fusion, nearest-neighbour
retrieval, and gradient boosting approaches. Exact installed package versions
are recorded in `requirements-lock.txt`.

| Component | Use | License / source |
|---|---|---|
| CatBoost 1.2.10 | Final classifier; experimental ranking losses | Apache-2.0, https://github.com/catboost/catboost |
| multilingual E5-small | Local semantic embeddings | MIT, https://huggingface.co/intfloat/multilingual-e5-small |
| scikit-learn, NumPy, SciPy, pandas | Sparse retrieval, baseline, data and numeric operations | BSD family; respective upstream projects |
| PyArrow | Parquet input | Apache-2.0 |
| Snowball stemmer | Russian stemming | BSD |
| PyTorch, Sentence Transformers, Transformers | Local neural inference | Upstream license notices apply; installed packages retain them |
| tqdm | Progress display | MPL-2.0 / MIT |
| Qwen2.5-1.5B-Instruct | Optional rejected HyDE pilot; not used by final inference | Apache-2.0, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |

E5 revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Inputs use `query:` /
`passage:` prefixes, normalized embeddings, 96-token truncation. Model weights
are downloaded by the setup script, not redistributed in this repository.

Qwen revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its optional download and
pilot are documented in `reports/round2/README.md`. It is unnecessary for the final
submission. The HyDE approach is described in https://arxiv.org/abs/2212.10496.

The supplied `model/selector.cbm` was trained locally on the contest training data.
Organizer data is not redistributed in Git; setup retrieves the organizer's public
archive. No external inference service is used.
