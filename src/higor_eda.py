"""Dicionário, limpeza, EDA e regressão linear da Parte 1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK_ROOT = Path(__file__).resolve().parent
OFFICIAL_DATA_PATH = TASK_ROOT / "data" / "crop_yield.csv"

NUMERIC_FEATURES = [
    "Precipitation (mm day-1)",
    "Specific Humidity at 2 Meters (g/kg)",
    "Relative Humidity at 2 Meters (%)",
    "Temperature at 2 Meters (C)",
]
CATEGORICAL_FEATURES = ["Crop"]
TARGET = "Yield"
REQUIRED_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]

COLUMN_DICTIONARY: list[dict[str, str]] = [
    {
        "column": "Crop",
        "dtype": "categorical",
        "description": "Cultura agrícola registrada no observatório climático.",
    },
    {
        "column": "Precipitation (mm day-1)",
        "dtype": "numeric",
        "description": "Precipitação média diária em milímetros.",
    },
    {
        "column": "Specific Humidity at 2 Meters (g/kg)",
        "dtype": "numeric",
        "description": "Umidade específica do ar a 2 metros (g/kg).",
    },
    {
        "column": "Relative Humidity at 2 Meters (%)",
        "dtype": "numeric",
        "description": "Umidade relativa do ar a 2 metros (%).",
    },
    {
        "column": "Temperature at 2 Meters (C)",
        "dtype": "numeric",
        "description": "Temperatura média do ar a 2 metros (°C).",
    },
    {
        "column": "Yield",
        "dtype": "numeric",
        "description": "Produtividade observada da cultura (alvo real do dataset).",
    },
]


@dataclass(frozen=True)
class OfficialDatasetContract:
    name: str = "crop_yield_fiap"
    target: str = TARGET
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)


OFFICIAL_CONTRACT = OfficialDatasetContract()


@dataclass
class HigorEdaAnalysis:
    source_path: Path
    contract: OfficialDatasetContract
    raw_frame: pd.DataFrame
    clean_frame: pd.DataFrame
    column_dictionary: pd.DataFrame
    describe_stats: pd.DataFrame
    null_counts: pd.DataFrame
    crop_counts: pd.DataFrame
    correlation_matrix: pd.DataFrame
    outlier_summary: dict[str, Any]
    metrics: dict[str, float]
    fitted_model: Pipeline
    predictions: pd.DataFrame


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {column.strip().lower(): column for column in frame.columns}
    rename: dict[str, str] = {}
    for canonical in REQUIRED_COLUMNS:
        source = aliases.get(canonical.lower())
        if source:
            rename[source] = canonical
    return frame.rename(columns=rename)


def load_dataset(data_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    candidate = Path(data_path) if data_path else OFFICIAL_DATA_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"dataset oficial não encontrado em {candidate}")

    frame = pd.read_csv(candidate)
    frame = _normalise_columns(frame)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"schema incompatível; colunas ausentes: {sorted(missing)}")

    return frame[REQUIRED_COLUMNS].copy(), candidate


def build_column_dictionary() -> pd.DataFrame:
    return pd.DataFrame(COLUMN_DICTIONARY)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean[CATEGORICAL_FEATURES[0]] = clean[CATEGORICAL_FEATURES[0]].astype(str).str.strip()
    clean = clean.dropna(subset=[TARGET])
    clean = clean.drop_duplicates().reset_index(drop=True)
    return clean


def _preprocessor() -> ColumnTransformer:
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
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    nonzero = np.abs(y_true.to_numpy()) > 1e-9
    mape = (
        float(np.mean(np.abs((y_true.to_numpy()[nonzero] - predictions[nonzero]) / y_true.to_numpy()[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "mape": mape,
    }


def _eda_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    describe_stats = frame[NUMERIC_FEATURES + [TARGET]].describe().T.reset_index().rename(columns={"index": "column"})
    null_counts = frame.isna().sum().reset_index()
    null_counts.columns = ["column", "null_count"]
    crop_counts = (
        frame[CATEGORICAL_FEATURES[0]]
        .value_counts()
        .rename_axis("crop")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    correlation_matrix = frame[NUMERIC_FEATURES + [TARGET]].corr()
    return describe_stats, null_counts, crop_counts, correlation_matrix


def _fit_linear_regression(
    frame: pd.DataFrame,
    *,
    random_state: int,
    test_size: float,
) -> tuple[Pipeline, dict[str, float], pd.DataFrame, dict[str, Any]]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, X_test, y_train, y_test = train_test_split(
        frame[features],
        frame[TARGET],
        test_size=test_size,
        random_state=random_state,
    )

    outlier_pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    train_matrix = outlier_pre.fit_transform(X_train[NUMERIC_FEATURES])
    train_detector = IsolationForest(contamination=0.03, random_state=random_state, n_estimators=300)
    inlier_mask = train_detector.fit_predict(train_matrix) == 1
    X_train_filtered = X_train.loc[inlier_mask]
    y_train_filtered = y_train.loc[inlier_mask]

    model = Pipeline([("preprocessor", _preprocessor()), ("model", LinearRegression())])
    model.fit(X_train_filtered, y_train_filtered)
    predictions = model.predict(X_test)

    prediction_frame = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": predictions,
            "residual": y_test.reset_index(drop=True).to_numpy() - predictions,
        }
    )
    outlier_summary = {
        "outlier_method": "IsolationForest",
        "contamination": 0.03,
        "train_rows": int(len(X_train)),
        "train_inliers": int(inlier_mask.sum()),
        "train_outliers_removed": int((~inlier_mask).sum()),
        "test_rows_untouched": int(len(X_test)),
    }
    return model, _metrics(y_test, predictions), prediction_frame, outlier_summary


def run_eda(
    data_path: str | Path | None = None,
    *,
    random_state: int = 42,
    test_size: float = 0.25,
) -> HigorEdaAnalysis:
    raw_frame, source = load_dataset(data_path)
    clean_frame = clean_dataset(raw_frame)
    describe_stats, null_counts, crop_counts, correlation_matrix = _eda_tables(clean_frame)
    model, metrics, predictions, outlier_summary = _fit_linear_regression(
        clean_frame,
        random_state=random_state,
        test_size=test_size,
    )
    return HigorEdaAnalysis(
        source_path=source,
        contract=OFFICIAL_CONTRACT,
        raw_frame=raw_frame,
        clean_frame=clean_frame,
        column_dictionary=build_column_dictionary(),
        describe_stats=describe_stats,
        null_counts=null_counts,
        crop_counts=crop_counts,
        correlation_matrix=correlation_matrix,
        outlier_summary=outlier_summary,
        metrics=metrics,
        fitted_model=model,
        predictions=predictions,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figures(analysis: HigorEdaAnalysis, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    sns.set_theme(style="whitegrid")

    columns = NUMERIC_FEATURES + [TARGET]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for axis, column in zip(axes[: len(columns)], columns, strict=True):
        sns.histplot(analysis.clean_frame[column], kde=True, ax=axis, color="#2a9d8f")
        axis.set_title(column, fontsize=9)
    for axis in axes[len(columns) :]:
        axis.axis("off")
    fig.suptitle("Distribuições das variáveis climáticas e Yield", fontsize=12)
    fig.tight_layout()
    distributions_path = figures_dir / "distributions.png"
    fig.savefig(distributions_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(distributions_path.relative_to(TASK_ROOT).as_posix())

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(analysis.correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axis)
    axis.set_title("Matriz de correlação (variáveis numéricas + Yield)")
    fig.tight_layout()
    correlation_path = figures_dir / "correlation.png"
    fig.savefig(correlation_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(correlation_path.relative_to(TASK_ROOT).as_posix())

    precip_col = "Precipitation (mm day-1)"
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=analysis.clean_frame,
        x=precip_col,
        y=TARGET,
        hue=CATEGORICAL_FEATURES[0],
        alpha=0.75,
        ax=axis,
    )
    axis.set_title("Yield vs precipitação por cultura")
    fig.tight_layout()
    yield_precip_path = figures_dir / "yield_vs_precipitation.png"
    fig.savefig(yield_precip_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(yield_precip_path.relative_to(TASK_ROOT).as_posix())

    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=analysis.predictions,
        x="predicted",
        y="residual",
        alpha=0.8,
        ax=axis,
        color="#e76f51",
    )
    axis.axhline(0, color="black", linewidth=1, linestyle="--")
    axis.set_title("Resíduos da Regressão Linear (conjunto de teste)")
    axis.set_xlabel("Previsto")
    axis.set_ylabel("Resíduo")
    fig.tight_layout()
    residuals_path = figures_dir / "linear_regression_residuals.png"
    fig.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(residuals_path.relative_to(TASK_ROOT).as_posix())

    return saved


def save_outputs(analysis: HigorEdaAnalysis, output_dir: Path | None = None) -> dict[str, Any]:
    root = output_dir or TASK_ROOT / "outputs" / "higor"
    figures_dir = TASK_ROOT / "figures" / "eda"
    root.mkdir(parents=True, exist_ok=True)

    analysis.column_dictionary.to_csv(root / "column_dictionary.csv", index=False)
    analysis.describe_stats.to_csv(root / "describe_stats.csv", index=False)
    analysis.null_counts.to_csv(root / "null_counts.csv", index=False)
    analysis.crop_counts.to_csv(root / "crop_counts.csv", index=False)
    analysis.correlation_matrix.to_csv(root / "correlation_matrix.csv")
    analysis.predictions.to_csv(root / "linear_regression_predictions.csv", index=False)

    metrics_frame = pd.DataFrame([{"model": "Linear Regression", **analysis.metrics}])
    metrics_frame.to_csv(root / "linear_regression_metrics.csv", index=False)

    figure_paths = _save_figures(analysis, figures_dir)

    report: dict[str, Any] = {
        "responsible": "Higor Henrique Garcia - RM571820",
        "source_path": analysis.source_path.name,
        "source_sha256": _sha256(analysis.source_path),
        "dataset_contract": analysis.contract.name,
        "rows_raw": int(len(analysis.raw_frame)),
        "rows_clean": int(len(analysis.clean_frame)),
        "target": analysis.contract.target,
        "outlier_summary": analysis.outlier_summary,
        "metrics": analysis.metrics,
        "figures": figure_paths,
    }
    (root / "eda_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(save_outputs(run_eda()), ensure_ascii=False, indent=2))
