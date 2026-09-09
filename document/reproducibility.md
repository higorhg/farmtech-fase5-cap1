# Reproducibility

## Parâmetros globais

- `random_state=42`
- `test_size=0.25`
- Dataset oficial: `data/crop_yield.csv` via `load_dataset()` de `vinicius_analysis.py`

## Parte 2 (Vinícius) e Parte 3 (Humberto)

Ambas compartilham:

1. Mesmo carregamento e contrato (`OFFICIAL_CONTRACT`)
2. Mesmo split treino/teste
3. `IsolationForest(contamination=0.03, n_estimators=300)` fitado **apenas** nas linhas de treino
4. Pré-processador idêntico: imputação mediana + `StandardScaler` nas numéricas; imputação moda + `OneHotEncoder` em `Crop`

## Comandos

Na pasta `src/` do repositório de entrega (ou na pasta da task, com `PYTHONPATH` da raiz do clone):

```bash
python -m unittest discover -s tests -q
```

## Saídas esperadas

- `outputs/humberto/model_metrics.csv`
- `outputs/humberto/predictions.csv`
- `outputs/humberto/analysis_report.json`
- `artifacts/humberto/*.joblib`

## Checklist antes do merge final

- [ ] Métricas DT e KNN no `crop_yield.csv` (sem `yield_index`)
- [ ] Testes `test_models` e `test_vinicius_analysis` passando
- [ ] Notebook regenerado a partir do script de build
