from __future__ import annotations

import unittest

try:
    from tasks.task15_farmtech_ml_cloud.vinicius_analysis import (
        OFFICIAL_CONTRACT,
        load_dataset,
        run_analysis,
    )
except ImportError:
    from vinicius_analysis import OFFICIAL_CONTRACT, load_dataset, run_analysis


class ViniciusAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.source, cls.contract = load_dataset()
        cls.analysis = run_analysis(random_state=42)

    def test_uses_official_fiap_dataset_without_synthetic_target(self) -> None:
        self.assertEqual(self.contract, OFFICIAL_CONTRACT)
        self.assertEqual(len(self.frame), 156)
        self.assertEqual(self.contract.target, "Yield")
        self.assertEqual(self.source.name, "crop_yield.csv")

    def test_cluster_and_outliers_are_reported(self) -> None:
        self.assertGreaterEqual(self.analysis.outlier_summary["selected_k"], 2)
        self.assertGreater(self.analysis.outlier_summary["silhouette"], 0)
        self.assertGreater(self.analysis.outlier_summary["outlier_count"], 0)
        self.assertEqual(
            len(self.analysis.analysis_frame),
            int(self.analysis.cluster_summary["registros"].sum()),
        )

    def test_two_required_models_share_metrics(self) -> None:
        self.assertEqual(set(self.analysis.metrics["model"]), {"Random Forest", "Gradient Boosting"})
        self.assertTrue(self.analysis.metrics["mae"].ge(0).all())
        self.assertTrue(self.analysis.metrics["rmse"].ge(0).all())
        self.assertTrue(self.analysis.metrics["r2"].le(1).all())

    def test_test_set_is_not_filtered_by_outlier_detection(self) -> None:
        self.assertGreater(self.analysis.outlier_summary["train_outliers_removed"], 0)
        self.assertEqual(
            len(self.analysis.predictions),
            self.analysis.outlier_summary["test_rows_untouched"],
        )

    def test_analysis_is_reproducible(self) -> None:
        repeated = run_analysis(random_state=42)
        self.assertEqual(self.analysis.best_model_name, repeated.best_model_name)
        self.assertEqual(
            self.analysis.metrics.round(10).to_dict(orient="records"),
            repeated.metrics.round(10).to_dict(orient="records"),
        )


if __name__ == "__main__":
    unittest.main()
