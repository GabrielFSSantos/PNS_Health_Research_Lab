# Documentação — PNS Health Research Lab

Entrada rápida na raiz: [README.md](../README.md).

## Documentos principais

| Arquivo | Para quem | Conteúdo |
|---------|-----------|----------|
| **[trajetoria.md](trajetoria.md)** | Pesquisa e apresentações | **Documento único da pesquisa** — enquadramento, fases F0–F3, decisões legacy vs pns (§2.1), **resultados (Parte 4)**, lacunas |
| [documentacao.md](documentacao.md) | Desenvolvimento e revisão técnica | Módulos, fórmulas, YAML, scripts, contrato de artefatos |
| [../configs/README.md](../configs/README.md) | Quem configura runs | Perfis `configs/profiles/`, overrides e `_internal/` |
| [referencias/](referencias/) | Ancoragem bibliográfica | BibTeX, notas por artigo, PDFs âncora |
| [research/monografia.pdf](research/monografia.pdf) | TCC original | Baseline metodológico (UFOP, 2025) |

## Ordem de leitura sugerida

1. [README.md](../README.md) — visão em 2 minutos e instalação
2. [configs/README.md](../configs/README.md) — como rodar com `--profile pns` ou `legacy`
3. [trajetoria.md](trajetoria.md) Partes 0–2 — problema, metodologia, protocolo e **§2.1 decisões**
4. [trajetoria.md Parte 4](trajetoria.md#parte-4--resultados-consolidados) — resultados modulares (4.0) + showcase TCC (4.1+)
5. [documentacao.md §1.5](documentacao.md#15-conceitos-em-linguagem-acessível) — conceitos e pipeline

## Manutenção

| Evento | Atualizar |
|--------|-----------|
| Nova run modular | `trajetoria.md` Parte 4.0 + `data/bases/` |
| Mudança em configs | `configs/README.md` + presets em `src/.../persistence/` |
| Novo módulo ou estágio | `documentacao.md` §3 |
| Novo artigo | `referencias/bibliografia.bib` + `referencias/notas/` |
| Mudança de tolerâncias TCC | `configs/_internal/tcc_overlay.yaml` + Parte 2 da trajetória |
| Nova decisão metodológica | `trajetoria.md` Parte 2.1 e Parte 5 |

Artefatos numéricos em `data/bases/` (principal) e `artifacts/showcase/` (baseline TCC congelada).
