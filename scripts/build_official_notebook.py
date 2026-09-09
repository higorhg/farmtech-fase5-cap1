"""Gera o notebook oficial da entrega, já com saídas preenchidas."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd

from tasks.task15_farmtech_ml_cloud.comparison import run_comparison, save_outputs
from tasks.task15_farmtech_ml_cloud.higor_eda import run_eda, save_outputs as save_eda
from tasks.task15_farmtech_ml_cloud.vinicius_analysis import run_analysis


OUTPUT = Path(__file__).resolve().parent / "HigorHenriqueGarcia_rm571820_pbl_fase4.ipynb"


def _md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def _code(source: str, stdout: str = ""):
    cell = nbf.v4.new_code_cell(dedent(source).strip() + "\n")
    if stdout:
        cell.outputs = [
            nbf.v4.new_output("stream", name="stdout", text=stdout if stdout.endswith("\n") else stdout + "\n")
        ]
        cell.execution_count = 1
    return cell


def build_notebook() -> Path:
    eda = run_eda(random_state=42)
    save_eda(eda)
    cluster = run_analysis(random_state=42)
    comparison = run_comparison(random_state=42)
    save_outputs(comparison)

    metrics_txt = comparison.metrics.round(4).to_string(index=False)
    crops_txt = eda.crop_counts.to_string(index=False)
    describe_txt = eda.describe_stats.round(2).to_string()
    cluster_txt = cluster.cluster_summary.round(3).to_string(index=False)
    corr_txt = eda.correlation_matrix.round(3).to_string()

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    notebook["cells"] = [
        _md(
            """
            # FarmTech na Era da Cloud Computing
            ## Entrega 1 — Machine Learning (`crop_yield.csv`)

            **Notebook oficial:** `HigorHenriqueGarcia_rm571820_pbl_fase4.ipynb`  
            **Dataset:** `data/crop_yield.csv` (156 registros, alvo real `Yield`)  
            **Protocolo comum:** `random_state=42`, teste 25%, Isolation Forest (3%) só no treino.

            | Integrante | RM | Parte |
            |---|---|---|
            | Higor Henrique Garcia | 571820 | Dicionário, limpeza, EDA e regressão linear |
            | Vinicius Alves Lopes dos Anjos | 572814 | Clusterização, outliers, Random Forest e Gradient Boosting |
            | Humberto | 570536 | Decision Tree e KNN |
            | Igor | 572822 | Comparação AWS (Entrega 2 no README) |
            """
        ),
        _md(
            """
            ## 1. Dicionário e qualidade dos dados

            Colunas oficiais: cultura, precipitação (mm/dia), umidade específica (g/kg),
            umidade relativa (%), temperatura (°C) e rendimento. Não há target sintético.
            Após a limpeza restam 156 linhas; nulos numéricos são imputados por mediana
            apenas no pipeline de modelo.
            """
        ),
        _code(
            """
            from pathlib import Path
            import pandas as pd
            from higor_eda import run_eda, save_outputs

            eda = run_eda(random_state=42)
            save_outputs(eda)
            print(eda.column_dictionary.to_string(index=False))
            print()
            print(eda.crop_counts.to_string(index=False))
            print()
            print(eda.null_counts.to_string(index=False))
            """,
            stdout=eda.column_dictionary.to_string(index=False)
            + "\n\n"
            + crops_txt
            + "\n\n"
            + eda.null_counts.to_string(index=False),
        ),
        _md(
            f"""
            ## 2. EDA — achados

            O rendimento varia por cultura (escalas diferentes de cacau, café, etc.),
            então `Crop` entra como one-hot. A precipitação e as umidades são as
            variáveis climáticas mais associadas ao `Yield`.

            ```
            {describe_txt}
            ```

            Correlação (numéricas + alvo):

            ```
            {corr_txt}
            ```

            Figuras: `figures/eda/distributions.png`, `figures/eda/correlation.png`,
            `figures/eda/yield_vs_precipitation.png`.
            """
        ),
        _md(
            f"""
            ## 3. Clusterização e outliers

            K-Means escolhe `k` pelo maior silhouette entre 2 e 6
            (**k={cluster.outlier_summary['selected_k']}**, silhouette
            {cluster.outlier_summary['silhouette']}). Isolation Forest marca
            {cluster.outlier_summary['outlier_count']} cenários discrepantes na EDA.
            Na regressão, o filtro aprende **só no treino**
            ({cluster.outlier_summary['train_outliers_removed']} removidos de
            {cluster.outlier_summary['train_rows']}); o teste fica com
            {cluster.outlier_summary['test_rows_untouched']} linhas intactas.

            ```
            {cluster_txt}
            ```
            """
        ),
        _code(
            """
            from vinicius_analysis import run_analysis

            cluster = run_analysis(random_state=42)
            print(cluster.cluster_summary.round(3).to_string(index=False))
            print(cluster.outlier_summary)
            """,
            stdout=cluster_txt + "\n" + str(cluster.outlier_summary),
        ),
        _md(
            """
            ## 4. Cinco regressões

            Mesmo split, mesmo pré-processamento (StandardScaler + OneHot de `Crop`)
            e o mesmo filtro de outlier no treino.

            | # | Algoritmo | Parte |
            |---|---|---|
            | 1 | Linear Regression | Higor |
            | 2 | Decision Tree | Humberto |
            | 3 | KNN Regressor | Humberto |
            | 4 | Random Forest | Vinícius |
            | 5 | Gradient Boosting | Vinícius |
            """
        ),
        _code(
            """
            from comparison import run_comparison, save_outputs

            comparison = run_comparison(random_state=42)
            report = save_outputs(comparison)
            print(comparison.metrics.round(4).to_string(index=False))
            print("Melhor modelo (RMSE):", comparison.best_model_name)
            """,
            stdout=metrics_txt + f"\nMelhor modelo (RMSE): {comparison.best_model_name}",
        ),
        _md(
            f"""
            ## 5. Conclusões e limites

            - Melhor RMSE no teste: **{comparison.best_model_name}**.
            - KNN sofre com a escala e a cardinalidade de culturas (156 linhas).
            - Árvores e boosting capturam interações clima × cultura; a regressão
              linear também se beneficia do one-hot de `Crop` neste conjunto pequeno.
            - Limite: amostra oficial é curta; métricas não devem ser lidas como
              produção agrícola nacional. Não removemos outlier do teste para não
              maquiar o erro.

            Figura comparativa: `figures/comparison/rmse_comparison.png`.

            ## 6. Entrega 2 — Cloud (resumo)

            On-Demand 100%, Linux `t3.micro` (2 vCPU, 1 GiB, até 5 Gbit) + 50 GiB gp3.
            **N. Virginia é mais barata** (~US$ 11,59/mês vs ~US$ 19,86 em São Paulo).
            Com restrição de dado no exterior e acesso rápido, a região escolhida é
            **São Paulo (`sa-east-1`)**: LGPD / residência no Brasil e menor latência.
            Figuras e passo a passo da calculadora: README e `document/`.
            """
        ),
    ]
    OUTPUT.write_text(nbf.writes(notebook), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    path = build_notebook()
    print(f"Notebook oficial: {path}")
