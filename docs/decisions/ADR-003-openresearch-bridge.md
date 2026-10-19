# ADR-003: Usar OpenResearch como camada não-live de pesquisa

## Status

Accepted

## Data

2026-09-14

## Contexto

O [OpenResearch](https://github.com/alphaXiv/OpenResearch) organiza agentes,
worktrees, árvore de experimentos, logs e backends de execução. Isso cobre a
descoberta de literatura e a exploração paralela, mas não conhece o contrato
CSV do Biohub, o holdout por dataset/tempo, as invariantes de topologia ou a
reconciliação de submissions Kaggle.

## Decisão

Usar o OpenResearch apenas para propor e documentar hipóteses. O adaptador
`biohub_lab.openresearch` aceita um snapshot JSON, exige um comando fixo sem
shell, rejeita mudanças de comando/ambiente, separa nós já respondidos ou em
execução e emite propostas com `enabled=false`, `live_eligible=false` e
`metric_probe=false`. Opções que o runner Biohub ainda não implementa ficam
registradas como leads de reimplementação, sem serem executadas.

`lab.md`, `state/submissions.json`, o validador e o runner Biohub continuam
sendo as fontes de verdade. Nenhum processo `orx`, branch ou upload é
iniciado pelo adaptador.

## Consequências

- A pesquisa pode ser paralelizada sem permitir que uma hipótese externa pule os
  gates de holdout, cobertura temporal, topologia e tempo.
- O contrato fixo de execução fica explícito em `fixed_run_command` e
  `fixed_run_argv`.
- A fila preserva a distinção entre uma ideia reproduzível, uma configuração
  materializável e uma candidata live.
- O OpenResearch não é uma dependência do runtime Biohub e não é usado no
  ambiente local enquanto os recursos Modal estiverem bloqueados.
