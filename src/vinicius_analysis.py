"""Clusterização, outliers, Random Forest e Gradient Boosting da Parte 2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK_ROOT = Path(__file__).resolve().parent
OFFICIAL_DATA_PATH = TASK_ROOT / "data" / "crop_yield.csv"


@dataclass(frozen=True)
class DatasetContract:
    name: str
    target: str
    numeric_features: list[str]
    categorical_features: list[str]
    cluster_features: list[str]


@dataclass
class ViniciusAnalysis:
    source_path: Path
    contract: DatasetContract
    analysis_frame: pd.DataFrame
    cluster_summary: pd.DataFrame
    outlier_summary: dict[str, Any]
    metrics: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    best_model_name: str


FARMTECH_CONTRACT = DatasetContract(
    name="fase4_farmtech_sensores_manejo",
    target="rendimento_esperado_kg_ha",
    numeric_features=[
        "area_ha",
        "umidade_solo",
        "ph",
        "nitrogenio_mg_kg",
        "fosforo_mg_kg",
        "potassio_mg_kg",
        "temperatura_c",
        "chuva_mm",
        "radiacao_w_m2",
        "irrigacao_mm",
        "fertilizante_kg_ha",
    ],
    categorical_features=["cultura"],
    cluster_features=[
        "umidade_solo",
        "ph",
        "nitrogenio_mg_kg",
        "fosforo_mg_kg",
        "potassio_mg_kg",
        "temperatura_c",
        "chuva_mm",
        "irrigacao_mm",
    ],
)

OFFICIAL_CONTRACT = DatasetContract(
    name="crop_yield_fiap",
    target="Yield",
    numeric_features=[
        "Precipitation (mm day-1)",
        "Specific Humidity at 2 Meters (g/kg)",
        "Relative Humidity at 2 Meters (%)",
        "Temperature at 2 Meters (C)",
    ],
    categorical_features=["Crop"],
    cluster_features=[
        "Precipitation (mm day-1)",
        "Specific Humidity at 2 Meters (g/kg)",
        "Relative Humidity at 2 Meters (%)",
        "Temperature at 2 Meters (C)",
    ],
)


def _normalise_official_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {column.strip().lower(): column for column in frame.columns}
    expected = OFFICIAL_CONTRACT.categorical_features + OFFICIAL_CONTRACT.numeric_features + [
        OFFICIAL_CONTRACT.target
    ]
    rename: dict[str, str] = {}
    for canonical in expected:
        source = aliases.get(canonical.lower())
        if source:
            rename[source] = canonical
    return frame.rename(columns=rename)


def load_dataset(data_path: str | Path | None = None) -> tuple[pd.DataFrame, Path, DatasetContract]:
    candidate = Path(data_path) if data_path else OFFICIAL_DATA_PATH
    if not candidate.exists():
        raise FileNotFoundError(
            "dataset oficial não encontrado; adicione data/crop_yield.csv ao repositório"
        )

    frame = pd.read_csv(candidate)
    frame = _normalise_official_columns(frame)
    if set(OFFICIAL_CONTRACT.numeric_features + OFFICIAL_CONTRACT.categorical_features + [OFFICIAL_CONTRACT.target]).issubset(frame.columns):
        contract = OFFICIAL_CONTRACT
    elif set(FARMTECH_CONTRACT.numeric_features + FARMTECH_CONTRACT.categorical_features + [FARMTECH_CONTRACT.target]).issubset(frame.columns):
        contract = FARMTECH_CONTRACT
    else:
        raise ValueError(
            "schema incompatível: esperado crop_yield.csv oficial ou farmtech_sensores_manejo.csv da Fase 4"
        )

    required = contract.numeric_features + contract.categorical_features + [contract.target]
    clean = frame[required].copy()
    for column in contract.numeric_features + [contract.target]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[contract.target].isna().all():
        raise ValueError(f"target {contract.target} não contém valores numéricos")
    return clean, candidate, contract


def _cluster_and_outliers(
    frame: pd.DataFrame,
    contract: DatasetContract,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    cluster_matrix = scaler.fit_transform(imputer.fit_transform(frame[contract.cluster_features]))

    candidates: list[dict[str, float | int]] = []
    max_k = min(6, len(frame) - 1)
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(cluster_matrix)
        candidates.append({"k": k, "silhouette": float(silhouette_score(cluster_matrix, labels))})
    best_k = int(max(candidates, key=lambda row: row["silhouette"])["k"])
    final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=30)
    labels = final_kmeans.fit_predict(cluster_matrix)

    detector = IsolationForest(contamination=0.03, random_state=random_state, n_estimators=300)
    outlier_labels = detector.fit_predict(cluster_matrix)
    enriched = frame.copy()
    enriched["cluster"] = labels
    enriched["is_outlier_eda"] = outlier_labels == -1

    summary = (
        enriched.groupby("cluster", as_index=False)
        .agg(
            registros=(contract.target, "size"),
            rendimento_medio=(contract.target, "mean"),
            rendimento_mediano=(contract.target, "median"),
            outliers=("is_outlier_eda", "sum"),
        )
        .sort_values("rendimento_medio", ascending=False)
    )
    best_silhouette = max(float(row["silhouette"]) for row in candidates)
    details = {
        "selected_k": best_k,
        "silhouette": round(best_silhouette, 6),
        "candidate_scores": candidates,
        "outlier_method": "IsolationForest",
        "contamination": 0.03,
        "outlier_count": int(enriched["is_outlier_eda"].sum()),
        "outlier_rate": round(float(enriched["is_outlier_eda"].mean()), 6),
    }
    return enriched, summary, details


def _preprocessor(contract: DatasetContract) -> ColumnTransformer:
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
        [("numeric", numeric, contract.numeric_features), ("categorical", categorical, contract.categorical_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _models(contract: DatasetContract, random_state: int) -> dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline(
            [
                ("preprocessor", _preprocessor(contract)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=2,
                        max_features=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", _preprocessor(contract)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.04,
                        max_depth=3,
                        min_samples_leaf=3,
                        loss="huber",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    nonzero = np.abs(y_true.to_numpy()) > 1e-9
    mape = float(np.mean(np.abs((y_true.to_numpy()[nonzero] - predictions[nonzero]) / y_true.to_numpy()[nonzero]))) if nonzero.any() else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "mape": mape,
    }


def _importance(models: dict[str, Pipeline]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, pipeline in models.items():
        features = pipeline.named_steps["preprocessor"].get_feature_names_out()
        values = pipeline.named_steps["model"].feature_importances_
        rows.extend(
            {"model": name, "feature": str(feature), "importance": float(value)}
            for feature, value in zip(features, values, strict=True)
        )
    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])


def run_analysis(
    data_path: str | Path | None = None,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
) -> ViniciusAnalysis:
    frame, source, contract = load_dataset(data_path)
    enriched, cluster_summary, outlier_summary = _cluster_and_outliers(
        frame, contract, random_state=random_state
    )

    features = contract.numeric_features + contract.categorical_features
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features],
        frame[contract.target],
        test_size=test_size,
        random_state=random_state,
    )

    # O filtro aprende somente no treino; o teste permanece intacto para avaliação honesta.
    outlier_pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    train_matrix = outlier_pre.fit_transform(X_train[contract.cluster_features])
    train_detector = IsolationForest(contamination=0.03, random_state=random_state, n_estimators=300)
    inlier_mask = train_detector.fit_predict(train_matrix) == 1
    X_train_filtered = X_train.loc[inlier_mask]
    y_train_filtered = y_train.loc[inlier_mask]

    fitted: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_frame = pd.DataFrame({"actual": y_test.reset_index(drop=True)})
    for name, pipeline in _models(contract, random_state).items():
        pipeline.fit(X_train_filtered, y_train_filtered)
        predictions = pipeline.predict(X_test)
        fitted[name] = pipeline
        metric_rows.append({"model": name, **_metrics(y_test, predictions)})
        prediction_frame[name] = predictions

    metrics = pd.DataFrame(metric_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    best = str(metrics.loc[0, "model"])
    return ViniciusAnalysis(
        source_path=source,
        contract=contract,
        analysis_frame=enriched,
        cluster_summary=cluster_summary,
        outlier_summary={
            **outlier_summary,
            "train_rows": int(len(X_train)),
            "train_inliers": int(inlier_mask.sum()),
            "train_outliers_removed": int((~inlier_mask).sum()),
            "test_rows_untouched": int(len(X_test)),
        },
        metrics=metrics,
        fitted_models=fitted,
        predictions=prediction_frame,
        feature_importance=_importance(fitted),
        best_model_name=best,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_outputs(analysis: ViniciusAnalysis, output_dir: Path | None = None) -> dict[str, Any]:
    root = output_dir or TASK_ROOT / "outputs" / "vinicius"
    models_dir = TASK_ROOT / "artifacts" / "vinicius"
    root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    analysis.cluster_summary.to_csv(root / "cluster_summary.csv", index=False)
    analysis.metrics.to_csv(root / "model_metrics.csv", index=False)
    analysis.predictions.to_csv(root / "predictions.csv", index=False)
    analysis.feature_importance.to_csv(root / "feature_importance.csv", index=False)

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
        "responsible": "Vinicius Alves Lopes dos Anjos - RM572814",
        "source_path": analysis.source_path.name,
        "source_sha256": _sha256(analysis.source_path),
        "dataset_contract": analysis.contract.name,
        "rows": int(len(analysis.analysis_frame)),
        "target": analysis.contract.target,
        "cluster": analysis.outlier_summary,
        "best_model": analysis.best_model_name,
        "metrics": analysis.metrics.to_dict(orient="records"),
        "artifacts": artifact_paths,
    }
    (root / "analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(save_outputs(run_analysis()), ensure_ascii=False, indent=2))
