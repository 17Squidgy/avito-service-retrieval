"""Regression checks for costly contest mistakes and held-out label leakage."""
import unittest

import numpy as np
import pandas as pd

from avito_retrieval.data import context_keys, split_queries
from avito_retrieval.retrieval import FEATURE_NAMES, Retriever, topk
from avito_retrieval.submission import validate


class Contracts(unittest.TestCase):
    def test_submission_preserves_ids_and_rejects_duplicates(self):
        queries = pd.DataFrame({"query_id": ["00WuFMaXSFZBxSzT"]})
        items = pd.DataFrame({"item_id": ["000000000000000a"]})
        answer = pd.DataFrame({"query_id": queries.query_id, "answer": [items.item_id[0]]})
        self.assertTrue(validate(answer, queries, items)["valid"])
        for bad in ["000000000000000A", "000000000000000a 000000000000000a", "000000000000000b"]:
            answer["answer"] = bad
            with self.assertRaises(ValueError):
                validate(answer, queries, items)

    def test_split_excludes_every_pair_of_heldout_context(self):
        frame = pd.DataFrame({"search_query": [f"услуга {i}" for i in range(20)] * 2,
            "search_location_id": 1, "search_is_delivery_search": 0,
            "search_infm_params_text": "", "search_category": 1,
            "item_id": [f"{i:016x}" for i in range(40)]})
        frame["context_id"] = context_keys(frame)
        history, splits = split_queries(frame, n_fit=4, n_dev=3, n_test=3)
        heldout = set().union(*(set(q.query_id) for q in splits.values()))
        self.assertFalse(heldout & set(history.context_id))
        self.assertTrue(all(len(r) == 2 for q in splits.values() for r in q.relevant))

    def test_topk_ties_are_deterministic(self):
        self.assertEqual(topk(np.array([1., 2., 2., 2.]), 2).tolist(), [1, 2])

    def test_retrieval_smoke(self):
        items = pd.DataFrame({"item_id": [f"{i:016x}" for i in range(8)],
            "item_title_raw": ["ремонт телевизоров"] * 4 + ["монтаж дверей"] * 4,
            "item_description_raw": ["услуги ремонт телевизоров"] * 4 + ["услуги монтаж дверей"] * 4,
            "item_infm_params_text": "вид услуги ремонт", "item_location_id": [1] * 4 + [2] * 4,
            "item_category_id": 1, "item_rating": 4.5, "item_rating_reviews_count": 5,
            "item_price": 100, "item_is_phone_hidden": 0, "item_is_message_forbidden": 0})
        history = items.copy()
        history["search_query"] = items.item_title_raw
        history["search_location_id"] = items.item_location_id
        history["search_category"] = 1
        queries = pd.DataFrame({"search_query": ["ремонт телевизора"],
            "search_location_id": [1], "search_category": [1],
            "search_infm_params_text": [""], "search_is_delivery_search": [0]})
        retriever = Retriever(items, history, pool_per_generator=3)
        result = retriever.retrieve(queries)[0]
        self.assertEqual(result["features"].shape[1], len(FEATURE_NAMES))
        self.assertTrue(np.isfinite(result["features"]).all())
        self.assertLess(result["generators"]["bm25"][0], 4)
        # Completely disjoint historical items are a valid cold-start case.
        history["item_id"] = [f"{i + 100:016x}" for i in range(8)]
        cold = Retriever(items, history, pool_per_generator=3).retrieve(queries)[0]
        self.assertEqual(len(cold["generators"]["history"]), 0)
        self.assertTrue(np.isfinite(cold["features"]).all())


if __name__ == "__main__":
    unittest.main()
