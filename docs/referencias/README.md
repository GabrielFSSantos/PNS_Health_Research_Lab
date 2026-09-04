# Referências bibliográficas — PNS Health Research Lab

Índice de artigos usados para ancorar decisões em [trajetoria.md](../trajetoria.md). BibTeX: [bibliografia.bib](bibliografia.bib). Notas: [notas/](notas/). Monografia (TCC): [../research/monografia.pdf](../research/monografia.pdf) · [nota](notas/monografia_tcc.md).

## Referências âncora (versionadas em `pdfs/`)

| # | Referência | Uso no lab | PDF local |
|---|------------|------------|-----------|
| 1 | **Santos (2025)** — TCC UFOP | Baseline metodológico, métricas oficiais, pipeline R+Python | [../research/monografia.pdf](../research/monografia.pdf) · [nota](notas/monografia_tcc.md) |
| 2 | **Ferreira et al.** — ERICA tendências uso serviços | Contexto PNS/ERICA, variável-alvo | [pdfs/erica_trends_health_services.pdf](pdfs/erica_trends_health_services.pdf) · [nota](notas/erica_trends_health_services.md) |
| 3 | **Ferreira et al.** — ERICA prevalência DCNT | Comorbidades, perfis de saúde | [pdfs/erica_prevalence_ncd.pdf](pdfs/erica_prevalence_ncd.pdf) · [nota](notas/erica_prevalence_ncd.md) |
| 4 | **Mani & Zhang (2003)** / undersampling | NearMiss-1 no overlay legado | [pdfs/undersampling_strategies.pdf](pdfs/undersampling_strategies.pdf) · [nota](notas/undersampling_strategies.md) |
| 5 | **Prevalências DCNT na PNS** | Epidemiologia, comorbidades na base | [pdfs/prevalencia_dcnt_pns.pdf](pdfs/prevalencia_dcnt_pns.pdf) · [nota](notas/prevalencia_dcnt_pns.md) |
| — | **Imputação e ausentes PNS** | PRE-01/PRE-02 (trajetória §2.1) | [nota](notas/imputacao_pns.md) |

## Referências de suporte (PDF local em `pdfs/_local/`, não versionadas)

| Arquivo local | Tema | Nota |
|---------------|------|------|
| `1 - Obesity and health service utilization.pdf` | Obesidade × uso de serviços | [nota](notas/obesity_health_service.md) |
| `2 - Multimorbidity is associated with TV-viewing.pdf` | Multimorbidade × comportamento | [nota](notas/multimorbidity_tv.md) |
| `3 - Correlates of screen-based behaviors.pdf` | Comportamentos sedentários | [nota](notas/screen_behaviors.md) |
| `4 - Use of integrative and complementary.pdf` | Medicina integrativa/complementar | [nota](notas/integrative_complementary.md) |
| `5 - Multimorbidity and tooth loss.pdf` | Multimorbidade × saúde bucal | [nota](notas/multimorbidity_tooth_loss.md) |
| `6 - Prevalence, awareness, treatment and control of hypertension.pdf` | Hipertensão na PNS | [nota](notas/hypertension_pns.md) |
| `7 - Prevalence of functional dependence.pdf` | Dependência funcional | [nota](notas/functional_dependence.md) |
| `8 - Long working hours and self-rated health.pdf` | Trabalho × autoavaliação de saúde | [nota](notas/long_working_hours.md) |
| `Custos diretos da dor lombar em hospitais.pdf` | Custos hospitalares, dor lombar | [nota](notas/back_pain_costs.md) |
| `Medical costs and out-of-pocket expenditures.pdf` | Gastos de saúde | [nota](notas/medical_costs.md) |
| `The care pathway of individuals with.pdf` | Trajetória de cuidado | [nota](notas/care_pathway.md) |

## Matriz decisão → referência → fase

| Decisão metodológica | Referência | Fase |
|---------------------|------------|------|
| Variável-alvo: uso de serviços de saúde | Monografia; ERICA trends | F0, F1 |
| 15 comorbidades + limitação de atividades | Monografia; prevalência DCNT | F0, F1 |
| PCA (8 componentes) antes de K-Means | Monografia §4.3.2 | F0, F1 |
| k=6 clusters, silhouette ~0,51 | Monografia; showcase | F1 |
| NearMiss + target encoding antes do split | Mani & Zhang; monografia §4.2.7–4.2.9 | F0 (legado) |
| NA / códigos especiais → 0 em comorbidades | IBGE PNS; [nota imputação](notas/imputacao_pns.md) | F0, F1 — ver [trajetoria §2.1](../trajetoria.md#21-decisões-legacy-vs-pns-justificativas) |
| Peso amostral PNS (`V00291`) | Barros & Victora; IBGE PNS | F2 (default) |
| RF, MLP, XGBoost, SVC | Monografia cap. 5 | F0, F1 |
| Desbalanceamento de classes | Undersampling strategies | F0, F2 |

## Como citar na trajetória

1. Ler a [nota](notas/) antes de citar numa fase.
2. Indicar **o que tomamos** e **o que é diferente** no software atual.
3. Decisão metodológica: tabela Decisão | Alternativa | Fundamento (artigo + `run_id` ou showcase).

## Manutenção

- Novo artigo → entrada em `bibliografia.bib` + nota em `notas/` + linha neste README.
- PDF âncora novo → `pdfs/` com nome canônico; demais → `pdfs/_local/` (gitignored).
