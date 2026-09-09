# Justificativa de região AWS — FarmTech (LGPD + latência)

**Task:** FarmTech Cap 1 — Parte 4 (Cloud) · assign 625979  
**Cenário:** API/workload Linux leve (t3.micro + 50 GiB gp3) para apoio ao pipeline ML agrícola FarmTech.

## Pergunta do enunciado

> Com restrição legal de dado no exterior e necessidade de acesso rápido: **qual região escolher e por quê?**

## Resposta executiva

**Região escolhida: South America (São Paulo) — `sa-east-1`**

Mesmo custando **~US$ 19,86/mês** contra **~US$ 11,59/mês** em N. Virginia (+71%), São Paulo é a opção correta para FarmTech porque:

1. **Mantém dados e processamento no Brasil**, alinhado à LGPD e à expectativa de residência de dados sensíveis do agronegócio.
2. **Oferece latência muito menor** para usuários e sensores no território nacional.
3. **Reduz risco jurídico e operacional** de transferência internacional desnecessária.

O custo extra é o “prêmio de soberania + performance local” — aceitável para um MVP acadêmico de baixo volume.

---

## 1. Restrição legal — LGPD e dado no exterior

### Contexto FarmTech

O projeto trata de **dados agrícolas operacionais** (solo, clima, produtividade, localização de propriedades). Embora nem todo dado seja “dado pessoal”, na prática o sistema pode agregar:

- identificação de produtores, técnicos ou operadores;
- coordenadas e limites de propriedades rurais;
- histórico de produção vinculado a pessoas jurídicas ou físicas.

### LGPD (Lei nº 13.709/2018) — pontos relevantes

| Princípio / artigo | Implicação para cloud |
|--------------------|------------------------|
| **Art. 6º — necessidade e adequação** | Só transferir/processar fora do Brasil se houver base legal e proporcionalidade. |
| **Art. 33 — transferência internacional** | Dados pessoais só podem ir ao exterior com hipóteses legais (consentimento específico, cláusulas contratuais, adequação reconhecida, etc.). |
| **Residência de dados (expectativa do setor)** | Cooperativas e clientes agrícolas frequentemente exigem que dados **permaneçam no Brasil**, mesmo quando a lei permita transferência com salvaguardas. |

### Por que **não** escolher N. Virginia como região primária

| Risco em `us-east-1` | Mitigação em `sa-east-1` |
|----------------------|---------------------------|
| Dados em datacenter nos EUA | Datacenter AWS em **São Paulo, Brasil** |
| Transferência internacional por default | Tráfego e persistência **intra-Brasil** |
| Due diligence contratual extra (SCC, DPA, avaliação ANPD) | Contrato AWS Brasil + região local simplifica narrativa de conformidade |
| Percepção de “dado saiu do país” para o cliente rural | Mensagem clara: **“hospedado no Brasil”** |

> **Nota acadêmica:** LGPD não proíbe automaticamente cloud nos EUA, mas impõe **condições**. Para FarmTech, a restrição do enunciado (“dado no exterior”) torna **sa-east-1** a escolha defensável sem depender de exceções legais complexas.

---

## 2. Acesso rápido — latência para usuário brasileiro

### Latência típica (ordem de grandeza)

Valores variam por ISP, horário e rota; são **estimativas de referência** para o vídeo/README:

| Origem do usuário | Região AWS | RTT típico |
|-------------------|------------|------------|
| São Paulo / SP | `sa-east-1` | **15–40 ms** |
| Interior BR | `sa-east-1` | **30–80 ms** |
| São Paulo / SP | `us-east-1` | **120–180 ms** |
| Interior BR | `us-east-1` | **150–220 ms** |

### Impacto no FarmTech

- **Dashboard e API de inferência ML:** cada interação (consulta de cluster, previsão de yield) faz ida e volta HTTP; 100 ms a mais por request degradam UX em conexões móveis no campo.
- **Integração com sensores / edge:** pipelines near-real-time toleram menos latência; manter compute no Brasil evita “salto” transatlântico.
- **Disaster recovery:** backups e réplicas podem existir em outra região, mas o **endpoint primário** deve servir usuários BR com menor RTT.

**Conclusão de performance:** para “acesso rápido” ao usuário brasileiro, **São Paulo vence N. Virginia** de forma consistente.

---

## 3. Decisão multicritério

| Critério | Peso (FarmTech) | `sa-east-1` | `us-east-1` |
|----------|:---------------:|:-----------:|:-----------:|
| Custo On-Demand | Médio | ❌ +71% | ✅ mais barato |
| LGPD / dado no Brasil | **Alto** | ✅ | ❌ |
| Latência usuário BR | **Alto** | ✅ | ❌ |
| Maturidade / variedade de serviços | Baixo (MVP) | ✅ suficiente | ✅ marginalmente maior |
| **Resultado** | — | **✅ Escolhida** | Custo menor, risco/latência maiores |

---

## 4. Arquitetura recomendada (MVP)

```
[Usuário / app FarmTech no Brasil]
           │
           ▼  (baixa latência, dados no BR)
   [ALB ou API Gateway — sa-east-1]
           │
           ▼
   [EC2 t3.micro + EBS gp3 50 GiB — sa-east-1]
           │
           ├── Modelo ML (artefato read-only)
           └── Logs/métricas (sem PII desnecessária)
```

**Boas práticas complementares (mencionar no vídeo):**

- Criptografia EBS em repouso (default KMS).
- Security groups restritivos (443/22 apenas onde necessário).
- Não replicar dados pessoais para `us-east-1` sem base legal.
- Tag `DataClassification=agricultural` + região fixa em IaC (Terraform/CloudFormation futuro).

---

## 5. Frase pronta para o README / vídeo

> “Comparamos On-Demand 100% entre **São Paulo** e **N. Virginia**: Virginia é **~US$ 8,27/mês mais barata**, mas escolhemos **sa-east-1** porque o FarmTech precisa manter **dados no Brasil (LGPD)** e oferecer **resposta rápida** aos usuários no território nacional. O custo adicional compra conformidade e experiência de uso.”

---

## Referências

- Lei nº 13.709/2018 (LGPD): https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm
- AWS — Região South America (São Paulo): https://aws.amazon.com/about-aws/global-infrastructure/regions_az/
- Comparativo de custos: `docs/aws_compare.md` e `outputs/aws/cost_estimate.json`
