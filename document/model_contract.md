# Model Contract — Parte 3 (Humberto)

## Dataset

- Arquivo: `data/crop_yield.csv` (156 linhas)
- Alvo: `Yield`
- Features numéricas: precipitação, umidade específica, umidade relativa, temperatura
- Feature categórica: `Crop`
- Contrato compartilhado: `OFFICIAL_CONTRACT` em `vinicius_analysis.py`

## Protocolo (igual à Parte 2)

| Parâmetro | Valor |
|-----------|-------|
| `random_state` | 42 |
| `test_size` | 0.25 |
| Outliers | `IsolationForest(contamination=0.03)` **somente no treino** |
| Pré-processamento | `StandardScaler` (numéricas) + `OneHotEncoder` (`Crop`) |

## Modelos

| Modelo | Obrigatório | Algoritmo |
|--------|-------------|-----------|
| Decision Tree | Sim | `DecisionTreeRegressor` |
| KNN Regressor | Sim | `KNeighborsRegressor` |
| Dummy baseline | Referência | `DummyRegressor(strategy="median")` |

## Métricas de saída

- MAE, RMSE, R², MAPE
- Tabela de previsões no conjunto de teste
- Artefatos em `outputs/humberto/` e `artifacts/humberto/`

## Regras

- Sem target sintético nem fallback para `produtos_agricolas.csv`
- Mesmo split e filtro de outliers que `vinicius_analysis.run_analysis`
- Semente fixa garante reprodutibilidade entre partes
