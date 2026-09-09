# Comparação AWS On-Demand — São Paulo vs N. Virginia

**Task:** FarmTech Cap 1 — Parte 4 (Cloud) · assign 625979  
**Responsável:** Igor Zeviani Nogueira — RM 572822  
**Data da cotação:** 2026-09-08  
**Modelo de cobrança:** On-Demand 100% (sem Reserved, Spot ou Savings Plans)

## Hipótese de workload

| Item | Valor |
|------|-------|
| Sistema operacional | Linux/Unix |
| Tipo de instância | **t3.micro** (General purpose) |
| vCPU | 2 |
| Memória | 1 GiB |
| Rede | Até 5 Gbit/s (burst) |
| Disco | **EBS gp3**, 50 GiB |
| IOPS / throughput EBS | 3.000 IOPS e 125 MB/s (baseline incluído, sem provisionamento extra) |
| Uptime mensal | 730 h (padrão da calculadora AWS para estimativa mensal contínua) |

> **Por que t3.micro?** Atende exatamente ao enunciado (2 vCPU, 1 GiB, rede até 5 Gbit) e é a opção General purpose Linux mais usada em exercícios acadêmicos de baixo custo.

## Tabela de custos mensais (USD)

| Componente | São Paulo (`sa-east-1`) | N. Virginia (`us-east-1`) |
|------------|-------------------------:|---------------------------:|
| EC2 t3.micro On-Demand | $0,0168/h × 730 h = **$12,26** | $0,0104/h × 730 h = **$7,59** |
| EBS gp3 50 GiB | $0,152/GB-mês × 50 = **$7,60** | $0,08/GB-mês × 50 = **$4,00** |
| **Total mensal** | **$19,86** | **$11,59** |

### Qual região é mais barata?

**N. Virginia (`us-east-1`)** — economia de **$8,27/mês** (~**71%** mais barato que São Paulo para o mesmo cenário).

| Métrica | Valor |
|---------|------:|
| Diferença absoluta | $8,27/mês |
| São Paulo vs Virginia | +71,4% |

## Fontes da cotação

| Fonte | URL | Uso |
|-------|-----|-----|
| AWS Price List API (EC2 + EBS por região) | [us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json) · [sa-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/sa-east-1/index.json) | Preços unitários On-Demand e gp3 |
| Página EC2 T3 | https://aws.amazon.com/ec2/instance-types/t3/ | Confirmação de specs (2 vCPU, 1 GiB, 5 Gbit) |
| Página EBS | https://aws.amazon.com/ebs/pricing/ | Baseline gp3 (3.000 IOPS / 125 MB/s incluídos) |
| AWS Pricing Calculator | https://calculator.aws/ | Repetição manual do cenário (passo a passo abaixo) |

Números consolidados em `outputs/aws/cost_estimate.json`. Gráficos gerados por `build_aws_figures.py` em `figures/aws/`.

## Figuras geradas

| Arquivo | Conteúdo |
|---------|----------|
| `figures/aws/aws_monthly_total_bar.png` | Barras — total mensal SP vs VA |
| `figures/aws/aws_cost_breakdown_stacked.png` | Empilhado — EC2 vs EBS por região |
| `figures/aws/aws_sa_east_premium_pie.png` | Pizza — prêmio de São Paulo sobre base Virginia |

---

## Passo a passo para repetir na AWS Pricing Calculator

Use este roteiro para o **segundo vídeo** (gravação pelo Higor). Não é necessário print falso — grave a tela real da calculadora seguindo os passos.

### 1. Abrir a calculadora

1. Acesse https://calculator.aws/
2. Clique em **Create estimate** (ou **Criar estimativa**).

### 2. Adicionar EC2 (primeira região — São Paulo)

1. Busque **Amazon EC2** e clique em **Configure**.
2. **Region:** `South America (Sao Paulo)`.
3. **Pricing strategy:** `On-Demand`.
4. **Operating system:** `Linux`.
5. **Instance type:** `t3.micro` (2 vCPU, 1 GiB).
6. **Pricing model:** `On-Demand Instances` — 100% utilization, **730 hours/month**.
7. **Storage (EBS):**
   - Tipo: **General Purpose SSD (gp3)**
   - Tamanho: **50 GiB**
   - IOPS: **3000** (default incluído)
   - Throughput: **125 MB/s** (default incluído)
8. Anote o subtotal EC2 + EBS (~**US$ 19,86/mês** na cotação de 2026-09-08).

### 3. Duplicar ou criar segunda estimativa (N. Virginia)

1. Repita o mesmo fluxo com **Region:** `US East (N. Virginia)`.
2. Mesmos valores: t3.micro Linux, On-Demand 730 h, gp3 50 GiB.
3. Anote o subtotal (~**US$ 11,59/mês**).

### 4. Comparar e comentar no vídeo

1. Mostre os dois totais lado a lado.
2. Diga qual é **mais barata** (N. Virginia).
3. Explique que, **mesmo sendo mais cara**, para FarmTech a escolha recomendada é **São Paulo** — ver `docs/aws_justificativa.md` (LGPD + latência).
4. Opcional: inclua no README as PNGs de `figures/aws/` geradas por este repositório.

### 5. Exportar (opcional)

- Use **Export** / **Share** na calculadora para link PDF/JSON da estimativa oficial.
- Cole o link no README da entrega (não substitui as figuras locais).

---

## Observações

- Valores **não incluem**: transferência de dados para internet, Elastic IP ocioso, snapshots EBS, impostos ou conversão BRL.
- Instâncias T3 em Unlimited Mode podem gerar cobrança extra de CPU credits se a baseline for excedida continuamente; para demo/acadêmico assume-se uso dentro da baseline.
- Preços AWS mudam; sempre registrar **data da cotação** ao entregar.
