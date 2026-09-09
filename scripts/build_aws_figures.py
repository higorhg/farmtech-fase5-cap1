"""Gera gráficos PNG da comparação AWS On-Demand (Parte 4 — Cloud)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COST_JSON = ROOT / "outputs" / "aws" / "cost_estimate.json"
FIG_DIR = ROOT / "figures" / "aws"


def load_costs() -> dict:
    with COST_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def _region_labels(data: dict) -> tuple[list[str], dict[str, str]]:
    labels: list[str] = []
    codes: list[str] = []
    for code in ("sa-east-1", "us-east-1"):
        region = data["regions"][code]
        labels.append(f"{region['display_name']}\n({code})")
        codes.append(code)
    return labels, dict(zip(labels, codes, strict=True))


def plot_monthly_total_bar(data: dict) -> Path:
    labels, _ = _region_labels(data)
    totals = [data["regions"][code]["total_monthly_usd"] for code in ("sa-east-1", "us-east-1")]
    colors = ["#FF9900", "#232F3E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, totals, color=colors, width=0.55)
    ax.set_ylabel("USD / mês (On-Demand)")
    ax.set_title("Custo mensal estimado — EC2 t3.micro + EBS gp3 50 GiB")
    ax.set_ylim(0, max(totals) * 1.25)

    for bar, value in zip(bars, totals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"${value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    cheaper = data["comparison"]["cheaper_region_name"]
    ax.annotate(
        f"Mais barata: {cheaper}",
        xy=(1, totals[1]),
        xytext=(0.35, totals[1] + 2.5),
        arrowprops={"arrowstyle": "->", "color": "#555"},
        fontsize=9,
        color="#333",
    )

    fig.text(
        0.01,
        0.01,
        f"Fonte: AWS Price List API · cotação {data['meta']['quote_date']}",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    output = FIG_DIR / "aws_monthly_total_bar.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_cost_breakdown_stacked(data: dict) -> Path:
    labels, _ = _region_labels(data)
    ec2 = [data["regions"][code]["ec2"]["monthly"] for code in ("sa-east-1", "us-east-1")]
    ebs = [data["regions"][code]["ebs_gp3"]["monthly"] for code in ("sa-east-1", "us-east-1")]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.55
    x = range(len(labels))

    ec2_bars = ax.bar(x, ec2, width, label="EC2 t3.micro", color="#FF9900")
    ebs_bars = ax.bar(x, ebs, width, bottom=ec2, label="EBS gp3 (50 GiB)", color="#232F3E")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("USD / mês")
    ax.set_title("Composição do custo — instância vs disco")
    ax.legend(loc="upper right")

    for idx, (ec2_val, ebs_val) in enumerate(zip(ec2, ebs, strict=True)):
        total = ec2_val + ebs_val
        ax.text(idx, ec2_val / 2, f"${ec2_val:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(idx, ec2_val + ebs_val / 2, f"${ebs_val:.2f}", ha="center", va="center", color="white", fontsize=10)
        ax.text(idx, total + 0.35, f"Total ${total:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.text(
        0.01,
        0.01,
        "Baseline gp3: 3.000 IOPS e 125 MB/s incluídos (sem custo extra)",
        fontsize=8,
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    output = FIG_DIR / "aws_cost_breakdown_stacked.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_sa_east_premium_pie(data: dict) -> Path:
    sp = data["regions"]["sa-east-1"]
    va = data["regions"]["us-east-1"]
    extra = sp["total_monthly_usd"] - va["total_monthly_usd"]
    base = va["total_monthly_usd"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(
        [base, extra],
        labels=[
            f"Base (N. Virginia)\n${base:.2f}/mês",
            f"Prêmio São Paulo\n+${extra:.2f}/mês",
        ],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#232F3E", "#FF9900"],
        textprops={"fontsize": 10},
    )
    ax.set_title(
        "Quanto a mais custa São Paulo vs N. Virginia\n(mesmo workload On-Demand)",
        fontsize=11,
    )

    output = FIG_DIR / "aws_sa_east_premium_pie.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_costs()

    generated = [
        plot_monthly_total_bar(data),
        plot_cost_breakdown_stacked(data),
        plot_sa_east_premium_pie(data),
    ]
    for path in generated:
        print(f"Gerado: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
