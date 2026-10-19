# ADR-001: Controle leve separado dos candidatos de visão

## Status

Accepted

## Data

2026-08-04

## Contexto

O desafio usa imagens 3D+tempo, OME-Zarr, grafos de tracking e modelos que exigem ambientes diferentes. A máquina de desenvolvimento atual não possui NumPy, PyTorch, Zarr, `tracksdata` ou Kaggle CLI instalados.

## Decisão

Manter o controlador, o validador CSV, o ranking e o orçamento em Python padrão. Cada candidato de ML será um comando externo declarado em `config/experiments.json`, com artefatos e métricas padronizados.

## Alternativas consideradas

### Embutir PyTorch/Zarr no controlador

Rejeitado nesta fase: torna o smoke test pesado, mistura dependências e dificulta reproduzir candidatos com ambientes distintos.

### Automatizar diretamente notebooks Kaggle

Rejeitado como caminho principal: notebooks têm estado implícito e tornam o loop local menos auditável. O pipeline pode chamar um notebook/código externo no futuro, mas o contrato local continua sendo um artefato CSV + `metrics.json`.

### Enviar automaticamente após qualquer score local

Rejeitado: o score pode conter leakage ou otimização excessiva no public leaderboard. A reserva de slots é explícita, limitada e baseada em ranking local com diversidade.

## Consequências

- O primeiro teste roda sem GPU nem rede.
- A integração real exige adaptar o comando e o ambiente, mas não reescrever o controlador.
- Métricas científicas continuam responsabilidade do candidato/avaliador oficial; o controlador verifica existência, formato e proveniência.
- A separação torna possível registrar candidatos promissores e não promissores sem apagar evidências.
