"""Comparação das 5 regressões no crop_yield.csv oficial."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


TASK_ROOT = Path(__file__).resolve().parent
REQUIRED_MODELS = (
    "Linear Regression",
    "Decision Tree",
    "KNN Regressor",
    "Random Forest",
    "Gradient Boosting",
)


def _load_module(name: str):
    try:
        return importlib.import_module(f"tasks.task15_farmtech_ml_cloud.{name}")
    except ImportError:
        return importlib.import_module(name)


@dataclass
class GroupComparison:
    source_path: Path
    metrics: pd.DataFrame
    best_model_name: str
    cluster_summary: pd.DataFrame
    outlier_summary: dict[str, Any]


def run_comparison(
    data_path: str | Path | None = None,
    *,
    random_state: int = 42,
) -> GroupComparison:
    higor = _load_module("higor_eda")
    vinicius = _load_module("vinicius_analysis")
    humberto = _load_module("models")

    eda = higor.run_eda(data_path, random_state=random_state)
    cluster = vinicius.run_analysis(data_path, random_state=random_state)
    trees = humberto.run_models(data_path, random_state=random_state, include_dummy=False)

    linear_row = {"model": "Linear Regression", **eda.metrics}
    rows = [linear_row, *cluster.metrics.to_dict(orient="records"), *trees.metrics.to_dict(orient="records")]
    metrics = pd.DataFrame(rows)
    metrics = metrics[metrics["model"].isin(REQUIRED_MODELS)].copy()
    metrics = metrics.sort_values(["rmse", "mae"]).reset_index(drop=True)
    missing = set(REQUIRED_MODELS) - set(metrics["model"])
    if missing:
        raise ValueError(f"comparação incompleta, faltam modelos: {sorted(missing)}")

    return GroupComparison(
        source_path=eda.source_path,
        metrics=metrics,
        best_model_name=str(metrics.loc[0, "model"]),
        cluster_summary=cluster.cluster_summary,
        outlier_summary=cluster.outlier_summary,
    )


def save_outputs(comparison: GroupComparison, output_dir: Path | None = None) -> dict[str, Any]:
    root = output_dir or TASK_ROOT / "outputs" / "comparison"
    figures_dir = TASK_ROOT / "figures" / "comparison"
    root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison.metrics.to_csv(root / "model_metrics.csv", index=False)
    comparison.cluster_summary.to_csv(root / "cluster_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_frame = comparison.metrics.sort_values("rmse", ascending=False)
    ax.barh(plot_frame["model"], plot_frame["rmse"], color="#2E7D32")
    ax.set_xlabel("RMSE (teste)")
    ax.set_title("Cinco regressões no crop_yield.csv — menor RMSE é melhor")
    fig.tight_layout()
    figure_path = figures_dir / "rmse_comparison.png"
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    report = {
        "source_path": comparison.source_path.name,
        "best_model": comparison.best_model_name,
        "metrics": comparison.metrics.to_dict(orient="records"),
        "outlier_summary": comparison.outlier_summary,
        "figure": str(figure_path.relative_to(TASK_ROOT)),
    }
    (root / "comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = save_outputs(run_comparison())
    print(f"Melhor modelo: {result['best_model']}")
    print(pd.DataFrame(result["metrics"])[["model", "mae", "rmse", "r2"]].round(4).to_string(index=False))
