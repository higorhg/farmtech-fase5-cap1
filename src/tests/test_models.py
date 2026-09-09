from __future__ import annotations

import unittest

try:
    from tasks.task15_farmtech_ml_cloud.models import (
        REQUIRED_MODELS,
        run_models,
        save_outputs,
    )
    from tasks.task15_farmtech_ml_cloud.vinicius_analysis import OFFICIAL_CONTRACT, load_dataset
except ImportError:
    from models import REQUIRED_MODELS, run_models, save_outputs
    from vinicius_analysis import OFFICIAL_CONTRACT, load_dataset


class Task15ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.source, cls.contract = load_dataset()
        cls.analysis = run_models(random_state=42)

    def test_uses_official_crop_yield_dataset(self) -> None:
        self.assertEqual(self.contract, OFFICIAL_CONTRACT)
        self.assertEqual(len(self.frame), 156)
        self.assertEqual(self.contract.target, "Yield")
        self.assertEqual(self.source.name, "crop_yield.csv")

    def test_required_models_and_metrics(self) -> None:
        self.assertEqual(
            set(self.analysis.metrics["model"]),
            {"Dummy baseline", "Decision Tree", "KNN Regressor"},
        )
        for column in ("mae", "rmse", "r2", "mape"):
            self.assertIn(column, self.analysis.metrics.columns)
        self.assertTrue(self.analysis.metrics["mae"].ge(0).all())
        self.assertTrue(self.analysis.metrics["rmse"].ge(0).all())

    def test_best_model_is_one_of_required(self) -> None:
        self.assertIn(self.analysis.best_model_name, REQUIRED_MODELS)
        self.assertIn(self.analysis.best_model_name, self.analysis.fitted_models)

    def test_outlier_filter_applies_only_to_train(self) -> None:
        self.assertGreater(self.analysis.outlier_summary["train_outliers_removed"], 0)
        self.assertEqual(
            len(self.analysis.predictions),
            self.analysis.outlier_summary["test_rows_untouched"],
        )

    def test_analysis_is_reproducible(self) -> None:
        repeated = run_models(random_state=42)
        self.assertEqual(self.analysis.best_model_name, repeated.best_model_name)
        self.assertEqual(
            self.analysis.metrics.round(10).to_dict(orient="records"),
            repeated.metrics.round(10).to_dict(orient="records"),
        )

    def test_save_outputs_writes_humberto_artifacts(self) -> None:
        from pathlib import Path

        task_root = Path(__file__).resolve().parents[1]
        report = save_outputs(self.analysis)
        self.assertIn("metrics", report)
        self.assertEqual(report["dataset_contract"], OFFICIAL_CONTRACT.name)
        self.assertTrue((task_root / "outputs" / "humberto" / "model_metrics.csv").exists())
        self.assertTrue((task_root / "outputs" / "humberto" / "analysis_report.json").exists())


if __name__ == "__main__":
    unittest.main()
