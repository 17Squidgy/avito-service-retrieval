"""Additional selector features computed only from query and corpus metadata.

No labels or held-out interaction counts are used here. Generator membership is
represented by separate ranks rather than only a fused score.
"""
import re
import numpy as np

from .data import normalize
from .retrieval import FEATURE_NAMES, tokens

GENERATORS = ["bm25", "char", "history", "bm25_context", "char_context",
              "history_context", "exact_history_context", "bm25_local", "bm25_nearby",
              "popularity_context", "dense", "dense_context", "dense_local"]
EXTRA_NAMES = [f"rr_{g}" for g in GENERATORS] + [
    "source_count", "title_token_coverage", "title_exact_phrase", "title_words",
    "filter_token_coverage", "description_chars", "minimum_rating", "rating_margin",
    "cat_microcategory", "cat_item_category", "cat_search_category"]


class FeatureBuilder:
    def __init__(self, items):
        self.titles = items.item_title_raw.fillna("").map(normalize).to_numpy()
        self.title_tokens = [set(tokens(t)) for t in self.titles]
        self.params = items.item_infm_params_text.fillna("").to_numpy()
        self.description_lengths = items.item_description_raw.fillna("").str.len().to_numpy()
        self.microcat = items.item_microcat_id.to_numpy()
        self.category = items.item_category_id.to_numpy()
        self.parameter_cache = {}

    def enrich(self, results, queries):
        enriched = []
        for result, q in zip(results, queries.itertuples(index=False)):
            rows = result["candidates"]
            rank_columns = []
            for name in GENERATORS:
                indices = result["generators"].get(name, [])
                ranks = {int(row): 1 / (1 + rank) for rank, row in enumerate(indices)}
                rank_columns.append(np.array([ranks.get(int(row), 0) for row in rows], dtype=np.float32))
            query = normalize(q.search_query)
            query_tokens = set(tokens(query))
            filter_tokens = set(tokens(q.search_infm_params_text))
            cover = lambda a, b: len(a & b) / max(len(a), 1)
            title_coverage = [cover(query_tokens, self.title_tokens[row]) for row in rows]
            param_coverage = []
            for row in rows:
                if filter_tokens:
                    if int(row) not in self.parameter_cache:
                        self.parameter_cache[int(row)] = set(tokens(self.params[row]))
                    param_coverage.append(cover(filter_tokens, self.parameter_cache[int(row)]))
                else:
                    param_coverage.append(0)
            match = re.search(r"рейтинг пользователя\s+(\d+(?:[.,]\d+)?)", str(q.search_infm_params_text).lower())
            minimum_rating = float(match.group(1).replace(",", ".")) if match else 0
            extra = np.column_stack(rank_columns + [
                np.count_nonzero(np.column_stack(rank_columns), axis=1), title_coverage,
                [bool(query) and query in self.titles[row] for row in rows],
                [len(self.title_tokens[row]) for row in rows], param_coverage,
                np.log1p(self.description_lengths[rows]), np.full(len(rows), minimum_rating),
                result["features"][:, FEATURE_NAMES.index("rating")] - minimum_rating,
                self.microcat[rows], self.category[rows], np.full(len(rows), q.search_category),
            ]).astype(np.float32)
            enriched.append({**result, "features": np.column_stack([result["features"], extra])})
        return enriched
