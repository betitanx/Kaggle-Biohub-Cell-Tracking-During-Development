# ADR-002: Serializar IDs de nós no limite de submissão

## Status

Accepted

## Data

2026-08-05

## Contexto

Os algoritmos do laboratório usam grafos internos zero-based. Os cinco envios por Notebook (`55257733`, `55258616`, `55270179`, `55270868` e `55272073`) produziram arquivos que passavam as verificações estruturais locais, mas o Kaggle retornou `Submission Scoring Error`. A comparação com outputs públicos que receberam score confirmou que o avaliador aceita IDs zero-based, não contíguos e quebras `CRLF`; portanto IDs positivos e `LF` não são requisitos comprovados. O canário `55301126` cobriu `t=0..99` nos quatro datasets e também falhou. Os controles `55308277` e `55308442` mostraram que nem um output público byte a byte nem o `sample_submission.csv` embutido no Notebook foram aceitos; a investigação passou a tratar a reexecução do Notebook como parte do contrato, sem atribuir o erro aos IDs.

## Decisão

Manter IDs zero-based dentro dos algoritmos e remapeá-los para `1..N` por dataset somente em `write_submission_csv`, como uma convenção conservadora e auditável do laboratório — não como uma exigência atribuída ao avaliador. O validador de produção exige as dez colunas na ordem exata, IDs de linha contíguos de `0` a `N-1`, IDs ativos não negativos, referências válidas e, para candidatos de produção, nós em todos os timepoints esperados. `write_submission_csv`, `normalize_submission_ids` e a publicação no Dataset serializam/normalizam com quebras `LF`; `CRLF` permanece apenas como opção de diagnóstico controlado. A função `normalize_submission_ids` repara artefatos legados preservando a topologia. O cliente de Notebook baixa e valida o arquivo produzido pelo próprio Notebook antes de chamar `competition_submit_code`, mas a aceitação final depende da reexecução do código no avaliador.

## Alternativas consideradas

### Alterar todos os algoritmos para serem one-based

Rejeitado: espalharia uma convenção de serialização pela detecção, linking, divisão, benchmarks sintéticos e adaptadores, aumentando o risco de erros de referência.

### Exigir que a serialização de produção seja positiva

Aceito como convenção interna, mas não como regra do Kaggle: outputs públicos pontuados demonstraram que IDs zero-based e não contíguos também são aceitos. O validador local permite esses IDs para não confundir uma preferência de serialização com o contrato oficial.

### Remapear IDs globalmente entre datasets

Rejeitado: o exemplo oficial reinicia os IDs por dataset, e as arestas são referências locais ao grafo daquele dataset.

## Consequências

- Os algoritmos preservam a indexação natural de Python e `tracksdata`.
- Todos os candidatos que usam `write_submission_csv` passam a compartilhar o mesmo contrato de serialização.
- Arquivos antigos precisam ser reparados antes de novo envio. Os refs `55270179`, `55270868` e `55272073` falharam mesmo após o remapeamento e, no último caso, após a reserialização em `LF`.
- A esteira bloqueia candidatos que não cobrem todos os timepoints esperados e a variante clássica pode inserir uma detecção fallback determinística em frames vazios; essa correção melhora o preflight estrutural, mas não garante aceitação no scoring.
- A verificação pós-Notebook evita submeter um arquivo diferente ou inválido em relação ao artefato validado localmente.
- O primeiro runner autocontido revelou que o kernel Kaggle pode não ter `zarr`; a esteira inclui um leitor mínimo de Zarr v3 com NumPy + Blosc2 e valida a equivalência byte a byte no smoke local antes de repetir o envio. O retry `55311152` executou o kernel até `COMPLETE` e foi aceito pelo endpoint, deixando o score pendente.
