from __future__ import annotations

import unittest

try:
    from tasks.task15_farmtech_ml_cloud.comparison import REQUIRED_MODELS, run_comparison
except ImportError:
    from comparison import REQUIRED_MODELS, run_comparison


class GroupComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparison = run_comparison(random_state=42)

    def test_five_required_regressors_on_official_dataset(self) -> None:
        self.assertEqual(set(self.comparison.metrics["model"]), set(REQUIRED_MODELS))
        self.assertTrue(self.comparison.source_path.name == "crop_yield.csv")

    def test_best_model_has_lowest_rmse(self) -> None:
        best_row = self.comparison.metrics.iloc[0]
        self.assertEqual(self.comparison.best_model_name, best_row["model"])
        self.assertTrue(self.comparison.metrics["rmse"].min() == best_row["rmse"])

    def test_test_set_stays_unfiltered(self) -> None:
        self.assertEqual(self.comparison.outlier_summary["test_rows_untouched"], 39)
        self.assertGreater(self.comparison.outlier_summary["train_outliers_removed"], 0)


if __name__ == "__main__":
    unittest.main()
