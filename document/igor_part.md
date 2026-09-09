# Parte 4 — Igor Zeviani Nogueira (RM 572822)

## Escopo da Entrega 2 (Cloud)

- Estimativa **On-Demand 100%** na calculadora AWS: Linux, 2 vCPU, 1 GiB, até 5 Gbit, 50 GB.
- Máquina de referência: `t3.micro` + EBS **gp3** 50 GiB, 730 h/mês.
- Comparação **São Paulo (`sa-east-1`)** vs **N. Virginia (`us-east-1`)**.
- Justificativa com restrição de dado no exterior (LGPD) e latência para o Brasil.
- Figuras no README + passo a passo para o segundo vídeo.

## Conclusão

| Região | Total / mês (08/09/2026) |
|---|---:|
| N. Virginia | US$ 11,59 (**mais barata**) |
| São Paulo | US$ 19,86 |

**Região escolhida: São Paulo.** O custo maior é aceito para manter dado e processamento no Brasil e reduzir latência.

## Arquivos

- [`aws_compare.md`](aws_compare.md) — tabela, fontes, roteiro da calculadora
- [`aws_justificativa.md`](aws_justificativa.md) — LGPD + latência
- Figuras: `figures/aws/` (no repo de entrega: `assets/aws/`)
- JSON: `outputs/aws/cost_estimate.json`
