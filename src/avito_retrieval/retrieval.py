"""Complementary sparse generators and an explicit candidate feature contract.

All matrices are local scipy sparse matrices. No API is called at inference.
Location is a soft feature: service providers may serve neighbouring cities.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import re
import snowballstemmer

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from tqdm import tqdm

from .data import identifier, normalize


@lru_cache(maxsize=200000)
def stem(word):
    # Cache only token strings, never entire item descriptions.
    if not hasattr(stem, "impl"):
        stem.impl = snowballstemmer.stemmer("russian")
    return stem.impl.stemWord(word)


def tokens(text):
    return [stem(t) for t in normalize(text).split() if len(t) > 1]


def topk(scores, k):
    """Deterministic descending top-k, including ties at the cutoff."""
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int32)
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k - len(above)]
    chosen = np.concatenate([above, tied])
    return chosen[np.lexsort((chosen, -scores[chosen]))].astype(np.int32)


class BM25:
    def fit(self, texts):
        self.vectorizer = CountVectorizer(tokenizer=tokens, token_pattern=None,
                                          lowercase=False, min_df=2, max_features=250000,
                                          dtype=np.float32)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((len(texts) - document_frequency + .5) / (document_frequency + .5))
        # Robertson BM25 with positive IDF. Stored document weights allow sparse
        # query × document multiplication instead of Python document loops.
        row_penalty = 1.2 * (.25 + .75 * lengths / max(lengths.mean(), 1))
        counts.data = (counts.data * 2.2 /
                       (counts.data + np.repeat(row_penalty, np.diff(counts.indptr))))
        self.matrix = counts.multiply(idf).tocsr().T.tocsr()
        return self

    def transform(self, queries):
        q = self.vectorizer.transform(queries)
        q.data[:] = 1
        return q @ self.matrix


FEATURE_NAMES = [
    "bm25", "char", "history", "filter_bm25", "exact_history",
    "bm25_relative", "char_relative", "history_relative", "filter_relative",
    "same_location", "location_prior", "category_prior", "popularity",
    "rating", "reviews", "price", "phone_hidden", "messages_forbidden",
    "query_words", "filter_words", "delivery", "history_support",
    "rrf", "best_rank",
    "dense", "dense_relative",
    "log_distance_km",
]


class Retriever:
    def __init__(self, items, history, pool_per_generator=100, dense_model=None):
        self.items = items.reset_index(drop=True)
        self.ids = self.items.item_id.to_numpy()
        self.id_to_row = {v: i for i, v in enumerate(self.ids)}
        self.n = len(items)
        self.pool = pool_per_generator
        self.dense = None
        if dense_model:
            from .dense import DenseEncoder
            from pathlib import Path
            self.dense = DenseEncoder(dense_model, Path(dense_model).parent / "e5_cache")
            self.dense_items = self.dense.items(self.items)
        print(f"Indexing {self.n:,} items; {len(history):,} historical pairs", flush=True)
        title = self.items.item_title_raw.fillna("")
        params = self.items.item_infm_params_text.fillna("")
        desc = self.items.item_description_raw.fillna("").str.slice(0, 5000)
        self.title_bm = BM25().fit((title + " " + params).tolist())
        print("Title/parameter BM25 ready", flush=True)
        self.body_bm = BM25().fit(desc.tolist())
        print("Description BM25 ready", flush=True)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
            min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32)
        self.char_matrix = self.char_vectorizer.fit_transform(title.map(normalize)).T.tocsr()
        print("Character title index ready", flush=True)
        self.locations = self.items.item_location_id.map(identifier).to_numpy()
        self.categories = self.items.item_category_id.map(identifier).to_numpy()
        self.popularity = np.zeros(self.n, dtype=np.float32)
        counts = history.item_id.value_counts()
        for item, count in counts.items():
            if item in self.id_to_row:
                self.popularity[self.id_to_row[item]] = count
        self.popularity = np.log1p(self.popularity)

        def transition(search_col, item_col):
            table = defaultdict(Counter)
            for a, b in zip(history[search_col].map(identifier), history[item_col].map(identifier)):
                table[a][b] += 1
            return {a: {b: count / sum(values.values()) for b, count in values.items()}
                    for a, values in table.items()}

        self.location_prior = transition("search_location_id", "item_location_id")
        self.category_prior = transition("search_category", "item_category_id")
        self.location_cache = {}
        self.category_cache = {}
        self.distance_cache = {}
        # Corpus metadata is available for every candidate, including unseen
        # items. Location centroids do not depend on held-out query labels.
        coordinates = pd.DataFrame({"location": self.locations,
            "lat": pd.to_numeric(self.items.get("item_latitude", pd.Series(np.nan, index=self.items.index)), errors="coerce"),
            "lon": pd.to_numeric(self.items.get("item_longitude", pd.Series(np.nan, index=self.items.index)), errors="coerce")})
        self.centroids = coordinates.groupby("location")[["lat", "lon"]].median().to_dict("index")
        self.lat = np.radians(coordinates.lat.to_numpy(dtype=np.float32))
        self.lon = np.radians(coordinates.lon.to_numpy(dtype=np.float32))
        self.numeric = np.column_stack([
            pd.to_numeric(self.items[c], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            for c in ("item_rating", "item_rating_reviews_count", "item_price",
                      "item_is_phone_hidden", "item_is_message_forbidden")])
        self.numeric[:, 1:3] = np.log1p(np.maximum(self.numeric[:, 1:3], 0))

        # Index historical query text. Similar query retrieval transfers observed
        # choices to synonyms/spelling variants without using held-out labels.
        # Neighbours with no retrievable items must not occupy the nearest-query
        # slots. This matters at inference: most training items are absent from
        # the benchmark corpus. Global support statistics still use all history.
        hist = history.loc[history.item_id.isin(self.id_to_row), ["search_query", "item_id"]].copy()
        hist["text"] = hist.search_query.map(normalize)
        self.history_texts = sorted(set(hist.text) - {""})
        self.text_to_row = {s: i for i, s in enumerate(self.history_texts)}
        hvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                              max_features=150000, sublinear_tf=True, dtype=np.float32)
        self.history_vectorizer = hvec
        if self.history_texts:
            self.history_matrix = hvec.fit_transform(self.history_texts).T.tocsr()
        else:
            # Fit a vocabulary but expose zero query rows if history and corpus
            # are disjoint. Retrieval then relies entirely on content sources.
            hvec.fit(["empty history"])
            self.history_matrix = sparse.csr_matrix((len(hvec.vocabulary_), 0), dtype=np.float32)
        pair_counts = hist.groupby(["text", "item_id"]).size()
        rr, cc, vv = [], [], []
        for (text, item), count in pair_counts.items():
            if text in self.text_to_row and item in self.id_to_row:
                rr.append(self.text_to_row[text]); cc.append(self.id_to_row[item]); vv.append(np.log1p(count))
        self.history_clicks = sparse.csr_matrix((np.asarray(vv, dtype=np.float32), (rr, cc)),
                                               shape=(len(self.history_texts), self.n))
        self.history_normalized = (l2_normalize(self.history_clicks, norm="l1", axis=1)
                                   if self.history_texts else self.history_clicks)
        self.support = history.search_query.map(normalize).value_counts().to_dict()

    def priors(self, query):
        location, category = identifier(query.search_location_id), identifier(query.search_category)
        # Bound cache growth by actual location/category cardinality, not queries.
        if location not in self.location_cache:
            if len(self.location_cache) >= 64:
                self.location_cache.pop(next(iter(self.location_cache)))
            prior = self.location_prior.get(location, {})
            self.location_cache[location] = np.array([prior.get(v, 0) for v in self.locations], dtype=np.float32)
        if category not in self.category_cache:
            prior = self.category_prior.get(category, {})
            self.category_cache[category] = np.array([prior.get(v, 0) for v in self.categories], dtype=np.float32)
        return (((self.locations == location) & (location != "-1")).astype(np.float32),
                self.location_cache[location], self.category_cache[category])

    def retrieve(self, queries):
        """Return candidate indices/features and per-generator validation lists."""
        results = []
        dense_queries = self.dense.queries(queries) if self.dense else None
        # Small batches bound sparse score matrix memory even for generic queries.
        for start in tqdm(range(0, len(queries), 64), desc="Retrieval batches"):
            batch = queries.iloc[start:start + 64]
            texts = batch.search_query.fillna("").map(normalize).tolist()
            filters = batch.search_infm_params_text.fillna("").tolist()
            bm = self.title_bm.transform(texts) * 2 + self.body_bm.transform(texts) * .6
            char = self.char_vectorizer.transform(texts) @ self.char_matrix
            filter_scores = self.title_bm.transform(filters)
            similar = self.history_vectorizer.transform(texts) @ self.history_matrix
            dense_scores = (dense_queries[start:start + len(batch)] @ self.dense_items.T
                            if self.dense else None)
            for j, q in enumerate(batch.itertuples(index=False)):
                b = bm[j].toarray().ravel()
                c = char[j].toarray().ravel()
                f = filter_scores[j].toarray().ravel()
                sim = similar[j].toarray().ravel()
                nearest = topk(sim, 20)
                sim[sim < .25] = 0
                h = np.asarray((sparse.csr_matrix((sim[nearest] ** 4)[None, :]) @
                                self.history_normalized[nearest]).toarray()).ravel()
                exact = np.zeros(self.n, dtype=np.float32)
                if texts[j] in self.text_to_row:
                    exact = self.history_clicks[self.text_to_row[texts[j]]].toarray().ravel()
                same, loc, cat = self.priors(q)
                location = identifier(q.search_location_id)
                if location not in self.distance_cache:
                    if len(self.distance_cache) >= 64:
                        self.distance_cache.pop(next(iter(self.distance_cache)))
                    center = self.centroids.get(location, {"lat": np.nan, "lon": np.nan})
                    lat, lon = np.radians(center["lat"]), np.radians(center["lon"])
                    hav = np.sin((self.lat - lat) / 2) ** 2 + np.cos(self.lat) * np.cos(lat) * np.sin((self.lon - lon) / 2) ** 2
                    distance = 12742 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
                    self.distance_cache[location] = np.nan_to_num(distance, nan=20000).astype(np.float32)
                distance = self.distance_cache[location]
                context = .15 + .65 * same + .6 * np.sqrt(loc) + .2 * np.sqrt(cat)
                generators = {"bm25": b, "char": c, "history": h,
                              "bm25_context": b * context, "char_context": c * context,
                              "history_context": h * context,
                              "exact_history_context": exact * context,
                              "bm25_local": b * same,
                              "bm25_nearby": b * (.1 + np.exp(-distance / 30)),
                              "popularity_context": self.popularity * context}
                d = dense_scores[j] if self.dense else np.zeros(self.n, dtype=np.float32)
                if self.dense:
                    generators["dense"] = d
                    generators["dense_context"] = d + .04 * same + .02 * np.sqrt(loc)
                    generators["dense_local"] = d * same
                lists = {}
                for name, score in generators.items():
                    indices = topk(score, self.pool)
                    # Zero-similarity arbitrary IDs must not receive RRF votes.
                    lists[name] = indices[score[indices] > 0]
                candidates = np.unique(np.concatenate(list(lists.values())))
                if len(candidates) < min(50, self.n):
                    candidates = np.unique(np.concatenate([candidates, topk(self.popularity, 50)]))
                rrf = np.zeros(self.n, dtype=np.float32)
                best_rank = np.full(self.n, self.pool + 1, dtype=np.float32)
                for name, indices in lists.items():
                    ranks = np.arange(1, len(indices) + 1)
                    # Global popularity is a fallback generator, not a text match.
                    weight = .2 if name == "popularity_context" else 1
                    rrf[indices] += weight / (60 + ranks)
                    best_rank[indices] = np.minimum(best_rank[indices], ranks)
                arrays = [b, c, h, f, exact, b / max(b.max(), 1e-6),
                          c / max(c.max(), 1e-6), h / max(h.max(), 1e-6),
                          f / max(f.max(), 1e-6), same, loc, cat, self.popularity]
                features = np.column_stack([a[candidates] for a in arrays] +
                    [self.numeric[candidates, col] for col in range(5)] +
                    [np.full(len(candidates), value) for value in
                     (len(texts[j].split()), len(normalize(filters[j]).split()),
                      float(q.search_is_delivery_search or 0), np.log1p(self.support.get(texts[j], 0)))] +
                    [rrf[candidates], best_rank[candidates], d[candidates],
                     d[candidates] / max(d.max(), 1e-6)]).astype(np.float32)
                features = np.column_stack([features, np.log1p(distance[candidates])]).astype(np.float32)
                assert features.shape[1] == len(FEATURE_NAMES)
                results.append({"candidates": candidates, "features": features,
                                "generators": lists})
        return results
