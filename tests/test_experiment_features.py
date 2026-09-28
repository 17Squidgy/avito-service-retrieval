"""Ensure expanded features are label-free and preserve the base feature prefix."""
import unittest
import importlib.util
import numpy as np
import pandas as pd
from avito_retrieval.features import EXTRA_NAMES, FeatureBuilder, GENERATORS
from avito_retrieval.retrieval import FEATURE_NAMES


class ExtraFeatures(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("catboost"), "optional CatBoost dependency")
    def test_macro_recall_counts_retrieval_misses(self):
        from avito_retrieval.catboost_experiment import query_recalls
        results = [{"candidates": np.array([0])}, {"candidates": np.array([1])}]
        queries = pd.DataFrame({"relevant": [["a", "missing"], ["missing"]]})
        values = query_recalls(np.array([1., 1.]), results, queries,
                               np.array(["a", "b"]), np.array([0, 1, 2]))
        np.testing.assert_array_equal(values, [.5, 0.])

    def test_title_filter_and_generator_features(self):
        items = pd.DataFrame({"item_title_raw": ["Ремонт телевизоров", "Монтаж дверей"],
            "item_infm_params_text": ["ремонт техники", "установка"],
            "item_description_raw": ["Описание", "Описание"], "item_microcat_id": [10, 20],
            "item_category_id": [114, 114]})
        queries = pd.DataFrame({"search_query": ["ремонт телевизоров"],
            "search_infm_params_text": ["Рейтинг пользователя 4 звезды и выше"], "search_category": [114]})
        features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
        features[:, FEATURE_NAMES.index("rating")] = [4.5, 3]
        result = {"candidates": np.array([0, 1]), "features": features,
                  "generators": {"bm25": np.array([0, 1]), "char": np.array([0])}}
        output = FeatureBuilder(items).enrich([result], queries)[0]["features"]
        np.testing.assert_array_equal(output[:, :len(FEATURE_NAMES)], features)
        names = FEATURE_NAMES + EXTRA_NAMES
        self.assertEqual(output.shape[1], len(names))
        self.assertEqual(output[0, names.index("title_token_coverage")], 1)
        self.assertEqual(output[1, names.index("title_token_coverage")], 0)
        self.assertEqual(output[0, names.index("source_count")], 2)
        self.assertEqual(output[1, names.index("rr_bm25")], .5)
        self.assertEqual(output[0, names.index("rating_margin")], .5)
        self.assertEqual(output[1, names.index("rating_margin")], -1)


if __name__ == "__main__":
    unittest.main()
