from __future__ import annotations

import unittest

from sklearn.linear_model import LinearRegression

try:
    from tasks.task15_farmtech_ml_cloud.higor_eda import (
        OFFICIAL_CONTRACT,
        load_dataset,
        run_eda,
    )
except ImportError:
    from higor_eda import OFFICIAL_CONTRACT, load_dataset, run_eda


class HigorEdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.source = load_dataset()
        cls.analysis = run_eda(random_state=42)

    def test_uses_official_fiap_dataset_without_synthetic_target(self) -> None:
        self.assertEqual(self.analysis.contract, OFFICIAL_CONTRACT)
        self.assertEqual(len(self.frame), 156)
        self.assertEqual(OFFICIAL_CONTRACT.target, "Yield")
        self.assertEqual(self.source.name, "crop_yield.csv")
        self.assertIn("Yield", self.frame.columns)
        self.assertNotIn("rendimento_esperado_kg_ha", self.frame.columns)

    def test_eda_tables_are_populated(self) -> None:
        self.assertEqual(len(self.analysis.column_dictionary), 6)
        self.assertFalse(self.analysis.describe_stats.empty)
        self.assertFalse(self.analysis.crop_counts.empty)
        self.assertEqual(
            set(self.analysis.correlation_matrix.columns),
            set(OFFICIAL_CONTRACT.numeric_features + (OFFICIAL_CONTRACT.target,)),
        )

    def test_linear_regression_metrics_are_reported(self) -> None:
        self.assertIsInstance(self.analysis.fitted_model.named_steps["model"], LinearRegression)
        self.assertGreaterEqual(self.analysis.metrics["mae"], 0)
        self.assertGreaterEqual(self.analysis.metrics["rmse"], 0)
        self.assertLessEqual(self.analysis.metrics["r2"], 1)
        self.assertIn("mape", self.analysis.metrics)

    def test_test_set_is_not_filtered_by_outlier_detection(self) -> None:
        self.assertGreater(self.analysis.outlier_summary["train_outliers_removed"], 0)
        self.assertEqual(
            len(self.analysis.predictions),
            self.analysis.outlier_summary["test_rows_untouched"],
        )

    def test_analysis_is_reproducible(self) -> None:
        repeated = run_eda(random_state=42)
        self.assertEqual(
            {key: round(value, 10) for key, value in self.analysis.metrics.items()},
            {key: round(value, 10) for key, value in repeated.metrics.items()},
        )


if __name__ == "__main__":
    unittest.main()
