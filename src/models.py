"""Decision Tree, KNN e baseline Dummy — Parte 3 (Humberto)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from tasks.task15_farmtech_ml_cloud.vinicius_analysis import (
        OFFICIAL_CONTRACT,
        DatasetContract,
        load_dataset,
    )
except ImportError:
    from vinicius_analysis import (  # type: ignore
        OFFICIAL_CONTRACT,
        DatasetContract,
        load_dataset,
    )


TASK_ROOT = Path(__file__).resolve().parent

REQUIRED_MODELS = ("Decision Tree", "KNN Regressor")
OPTIONAL_MODELS = ("Dummy baseline",)


@dataclass
class HumbertoAnalysis:
    source_path: Path
    contract: DatasetContract
    metrics: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    predictions: pd.DataFrame
    best_model_name: str
    outlier_summary: dict[str, Any]


def _preprocessor(contract: DatasetContract) -> ColumnTransformer:
    """Mesmo pré-processador da Parte 2: StandardScaler + OneHot para Crop."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, contract.numeric_features),
            ("categorical", categorical, contract.categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _build_models(contract: DatasetContract, random_state: int) -> dict[str, Pipeline]:
    preprocessor = _preprocessor(contract)
    return {
        "Dummy baseline": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", _preprocessor(contract)),
                (
                    "model",
                    DecisionTreeRegressor(
                        random_state=random_state,
                        max_depth=6,
                        min_samples_leaf=4,
                    ),
                ),
            ]
        ),
        "KNN Regressor": Pipeline(
            [
                ("preprocessor", _preprocessor(contract)),
                (
                    "model",
                    KNeighborsRegressor(n_neighbors=7, weights="distance"),
                ),
            ]
        ),
    }


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    nonzero = np.abs(y_true.to_numpy()) > 1e-9
    mape = (
        float(
            np.mean(
                np.abs(
                    (y_true.to_numpy()[nonzero] - predictions[nonzero])
                    / y_true.to_numpy()[nonzero]
                )
            )
        )
        if nonzero.any()
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "mape": mape,
    }


def run_models(
    data_path: str | Path | None = None,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
    include_dummy: bool = True,
) -> HumbertoAnalysis:
    """Treina DT e KNN no crop_yield.csv com o protocolo da Parte 2."""
    frame, source, contract = load_dataset(data_path)
    if contract != OFFICIAL_CONTRACT:
        raise ValueError(
            "Parte 3 exige crop_yield.csv oficial; recebido contrato "
            f"{contract.name!r}."
        )

    features = contract.numeric_features + contract.categorical_features
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features],
        frame[contract.target],
        test_size=test_size,
        random_state=random_state,
    )

    # IsolationForest aprende só no treino; teste permanece intacto.
    outlier_pre = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    train_matrix = outlier_pre.fit_transform(X_train[contract.cluster_features])
    train_detector = IsolationForest(
        contamination=0.03, random_state=random_state, n_estimators=300
    )
    inlier_mask = train_detector.fit_predict(train_matrix) == 1
    X_train_filtered = X_train.loc[inlier_mask]
    y_train_filtered = y_train.loc[inlier_mask]

    model_catalog = _build_models(contract, random_state)
    if not include_dummy:
        model_catalog = {
            name: pipeline
            for name, pipeline in model_catalog.items()
            if name not in OPTIONAL_MODELS
        }

    fitted: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_frame = pd.DataFrame({"actual": y_test.reset_index(drop=True)})

    for name, pipeline in model_catalog.items():
        pipeline.fit(X_train_filtered, y_train_filtered)
        predictions = pipeline.predict(X_test)
        fitted[name] = pipeline
        metric_rows.append({"model": name, **_metrics(y_test, predictions)})
        prediction_frame[name] = predictions

    metrics = pd.DataFrame(metric_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    required_metrics = metrics[metrics["model"].isin(REQUIRED_MODELS)]
    best = str(required_metrics.loc[required_metrics["rmse"].idxmin(), "model"])

    return HumbertoAnalysis(
        source_path=source,
        contract=contract,
        metrics=metrics,
        fitted_models=fitted,
        predictions=prediction_frame,
        best_model_name=best,
        outlier_summary={
            "outlier_method": "IsolationForest",
            "contamination": 0.03,
            "train_rows": int(len(X_train)),
            "train_inliers": int(inlier_mask.sum()),
            "train_outliers_removed": int((~inlier_mask).sum()),
            "test_rows_untouched": int(len(X_test)),
        },
    )


def predict_preview(
    analysis: HumbertoAnalysis,
    *,
    model_name: str | None = None,
    n_rows: int = 8,
) -> pd.DataFrame:
    name = model_name or analysis.best_model_name
    model = analysis.fitted_models[name]
    test_features = analysis.predictions[["actual"]].copy()
    test_features["predicted"] = analysis.predictions[name].head(n_rows).to_numpy()
    test_features["error"] = (
        test_features["predicted"] - test_features["actual"]
    ).round(3)
    return test_features.head(n_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_outputs(analysis: HumbertoAnalysis, output_dir: Path | None = None) -> dict[str, Any]:
    root = output_dir or TASK_ROOT / "outputs" / "humberto"
    models_dir = TASK_ROOT / "artifacts" / "humberto"
    root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    analysis.metrics.to_csv(root / "model_metrics.csv", index=False)
    analysis.predictions.to_csv(root / "predictions.csv", index=False)

    artifact_paths: dict[str, str] = {}
    for name, model in analysis.fitted_models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        path = models_dir / filename
        joblib.dump(
            {
                "model": model,
                "dataset_contract": analysis.contract,
                "target": analysis.contract.target,
                "model_name": name,
                "random_state": 42,
            },
            path,
        )
        artifact_paths[name] = path.relative_to(TASK_ROOT).as_posix()

    report: dict[str, Any] = {
        "responsible": "Humberto - Parte 3",
        "source_path": analysis.source_path.name,
        "source_sha256": _sha256(analysis.source_path),
        "dataset_contract": analysis.contract.name,
        "rows": int(len(analysis.predictions) + analysis.outlier_summary["train_rows"]),
        "target": analysis.contract.target,
        "outlier_filter": analysis.outlier_summary,
        "best_model": analysis.best_model_name,
        "required_models": list(REQUIRED_MODELS),
        "metrics": analysis.metrics.to_dict(orient="records"),
        "artifacts": artifact_paths,
    }
    (root / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(save_outputs(run_models()), ensure_ascii=False, indent=2))
