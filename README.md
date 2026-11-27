# T09 — Pruning proativo de regras no UnimNeuron Safe

**Data Stream Mining (M5) · Grupo 09** · Beatriz Partidário, João Silva, Ricardo Rafael

> Pergunta de investigação: o pruning proativo baseado em atividade consegue remover regras obsoletas com pouca perda preditiva? E qual dos três sinais — inatividade fixa (S1), inatividade adaptativa (S2) ou utilização EMA (S3) — equilibra melhor compacidade e estabilidade preditiva sob concept drift?

Protocolo congelado a **20/11/2026**. Qualquer alteração posterior fica registada na secção [Registo de alterações](#11-registo-de-alterações) (o quê, quando, porquê).

---

## 1. Estado atual

| Componente | Estado |
|---|---|
| B0 — `ENF_UnimNeuron_Safe.py` do professor | intacto, não modificado; nas experiências usa-se `B0Counted` (o mesmo modelo + contadores) |
| S1 — inatividade fixa (`S1Fixed`) | implementado e testado |
| S2 — inatividade adaptativa (`S2Adaptive`) | implementado e testado |
| S3 — utilização EMA (`S3EMA`) | implementado e testado |
| Salvaguardas (última regra, cobertura de classe) | implementadas, iguais nas três variantes |
| Contadores (`pruned_now`, `created_now`, `recreated_now`, `capacity_deletions`) | implementados e testados |
| `run_experiment.py` | implementado e testado (CSV + `.json`) |
| Streams | A–B–A implementado em `t09/streams.py`; RandomRBFDrift, ConceptDrift_Agrawal e Elec2 [a preencher] |
| MVE (B0, S1, S2, S3 no A–B–A, 1 seed) | feito — ver [secção 8](#8-medições-já-feitas) |
| Ficheiro de testes (`tests/test_pruning.py`) | por fazer (os testes abaixo foram feitos à mão) |
| `metrics.py` | Implementado e testado; calcula métricas preditivas, estruturais e de recursos e gera gráficos comparativos. |
| `stats.py` | Implementado; análise estatística emparelhada com Wilcoxon, correção de Holm e Friedman. Aguarda múltiplas seeds para validação experimental. |
| Grelhas de hiperparâmetros | propostas para A–B–A; falta diagnóstico em RBF e Agrawal |
| Diagnóstico (`t09/diagnostics.py`) | feito no A–B–A (seed 100); por fazer em RBF e Agrawal |

Testes já feitos às variantes:
- o modelo nunca fica sem regras;
- S2 com `k = 0` produz exatamente a mesma trajetória que S1 com `H = H₀`;
- a média online dos intervalos (S2) dá o valor correto num caso calculado à mão;
- o estado EMA (S3) fica consistente quando regras são criadas, fundidas ou removidas;
- S1 com `H` enorme dá exatamente o mesmo resultado que B0;
- `B0Counted` dá exatamente as mesmas previsões e o mesmo nº de regras que o modelo original;
- a salvaguarda de classe guarda a última regra de cada classe (teste com regras construídas à mão);
- o contador `capacity_deletions` fica a 0 com `max_rules = 1000` e dispara quando o limite é baixo (20);
- a regra recriada conta só dentro de W, só perto de uma regra removida e só uma vez por regra removida;
- a soma das colunas `_now` do CSV é igual aos totais do `.json`;
- o pruning só usa `x`: garantido pela estrutura (a monitorização é feita antes de o modelo ver `y_t`); falta o teste escrito.

---

## 2. Esquema da experiência

```
 1. IMPLEMENTAR        B0 (original) + S1, S2, S3 como subclasses
        │              (só mudam a monitorização e o pruning)
        ▼
 2. DIAGNOSTICAR       correr B0 nos streams de desenvolvimento e medir
        │              intervalos entre ativações e ema/média → escalas
        ▼
 3. DEFINIR GRELHAS    valores de H, H₀, k, α, θ a partir dessas escalas
        │              (6–9 combinações por variante)
        ▼
 4. CONSTRUIR STREAMS  A–B–A, RBF, Agrawal, Elec2 · horizonte · warm-up
        │              ordem cronológica · seeds de tuning e de teste
        ▼
 5. TUNING             todas as combinações × seeds de desenvolvimento
        │              → 1 configuração escolhida por variante e stream
        ▼
 6. AVALIAÇÃO          B0, S1*, S2*, S3* × seeds de avaliação (uma vez só)
        │
        ▼
 7. ANÁLISE            métricas, curvas, recuperação após drift,
                       testes estatísticos, interpretação
```

### Porquê esta ordem

1. **Variantes primeiro, como subclasses de B0.** Herdam tudo do Safe (criação de regras, merge, consequentes, feature weights). Assim a única diferença entre B0 e S1–S3 é o pruning, que é o que a pergunta de investigação compara.
2. **Diagnóstico antes das grelhas.** Os parâmetros têm unidades com significado (observações, fração da média). Escolhê-los sem medir leva a grelhas onde a variante nunca atua — aconteceu com H₀ = 100 no S2, que tornava o termo adaptativo inativo.
3. **Grelhas fixadas antes de qualquer tuning,** com tamanho parecido entre variantes, para nenhuma ter mais oportunidades de "acertar".
4. **Seeds de tuning separadas das de avaliação.** Os hiperparâmetros escolhem-se só com as seeds de desenvolvimento; as de avaliação usam-se uma única vez. Isto evita otimizar sobre os dados com que se reportam os resultados.
5. **Comparação emparelhada.** Todas as variantes veem exatamente o mesmo stream (mesma seed, mesma ordem), por isso as diferenças vêm do método e não dos dados.

---

## 3. Decisões tomadas

### 3.1 Modelo e baseline

| Decisão | Valor | Motivo |
|---|---|---|
| Baseline B0 | `EvolvingNeuroFuzzyUnimSafe` original, sem pruning proativo | ponto de comparação da proposta |
| Ficheiro do professor | não é modificado | garante que B0 é o código fornecido |
| Variantes | subclasses de B0 (S2 herda de S1) | isola o efeito do pruning; S2 com k = 0 ≡ S1 |
| `max_rules` | **1000**, igual para todas as variantes | evita remoções por capacidade; na corrida de 20 000 obs. o B0 chegou a 139 regras |
| Remoções por capacidade | uma corrida com `capacity_deletions > 0` é inválida | isola o pruning proativo |
| Medição do uso | antes de aprender o label atual | o ponto atual não infla a ativação das regras; coerente com test-then-train |
| Salvaguarda base | nunca remover a última regra | o modelo nunca fica vazio |
| B0 nas experiências | `B0Counted` = `EvolvingNeuroFuzzyUnimSafe` sem alterações + contadores (`n_created`, `capacity_deletions`) | o contador de capacidade tem de existir também no B0; verificado que as previsões e o nº de regras são idênticos ao original |
| Contagem sem mexer no original | `CountingMixin` reescreve `_create_rule` só para contar e chama o original; se o nº de regras não muda após uma criação, houve uma remoção por capacidade | o ficheiro do professor fica intacto |

**Notas sobre o modelo fornecido** (registadas como pede o protocolo):
- o merge (`_maybe_merge_last_rule`) nunca fundiu regras nas corridas feitas (0 merges em 20 000 obs. no A–B–A): na prática, as regras só são criadas, nunca fundidas;
- alguns argumentos do construtor são aceites mas não usados: `lambda_sim`, `buffer_size_similarity` e outros que caem em `**kwargs`; o `random_state` cria um gerador que nunca é usado, por isso o modelo é determinístico;
- o modelo só cria regras novas quando o ponto cai fora do intervalo de distâncias à média global (ou longe da regra mais próxima), e não quando uma zona fica sem cobertura — relevante para interpretar o pruning (ver secção 8).

### 3.2 Variantes

**Sinal de atividade relevante (S1, S2):** a regra r está ativa na observação t se
`a_r ≥ β·a_max`, `a_max ≥ 0,30` e `d_r ≤ 3σ_r` (as três condições ao mesmo tempo).

| Variante | Estado por regra | Remove quando |
|---|---|---|
| S1 | `inactive` (obs. desde a última ativação relevante ou desde a criação) | `inactive > H` |
| S2 | `inactive`, `n_act`, `mean_gap` (média online dos intervalos) | `inactive > H_r`, com `H_r = H₀` se `n_act < n_min`, senão `max(H₀, k·mean_gap)` |
| S3 | `ema`, `born_t` | `ema < θ·média(ema)` **e** `t − born_t ≥ G` |

Detalhes de implementação:
- S2: o intervalo é `inactive + 1`, lido antes de repor o contador; a média divide pelo número de intervalos (`n_act − 1`), não pelo número de ativações. `n_min = 6`.
- S3: `u_r = a_r / Σa` (o mesmo peso que a previsão usa); `ema ← (1−α)·ema + α·u_r`. Uma regra nova começa com `ema = média(ema)` atual. O limite `θ·média(ema)` é calculado uma vez por inspeção, antes de remover.
- S3 **não** usa as condições β / 0,30 / 3σ.

Parâmetros fixos (não entram na grelha):

| Parâmetro | Valor |
|---|---|
| β | 0,90 |
| `a_max` mínimo | 0,30 |
| fator de distância | 3σ |
| `n_min` (S2) | **6** — a média dos intervalos só é usada a partir de 6 ativações relevantes (5 intervalos); antes disso H_r = H₀. Valor fixo, não afinado |
| N — intervalo de inspeção | **1** — o critério é verificado a cada observação, igual nas três variantes |
| Salvaguarda de cobertura de classe | **sim**, igual nas três variantes (ver abaixo) |

**Implementação do pruning.** Cada variante só define quem é candidata (`_select_candidates`) e que regra vale mais a pena manter (`_keep_priority`: S1/S2 a mais recentemente ativa, S3 a de maior `ema`). O `_prune` é comum às três e aplica as salvaguardas, por isso estas são garantidamente iguais em S1, S2 e S3.

**Salvaguarda de cobertura de classe.** A classe dominante de uma regra é a classe com mais peso nos seus `class_counts`. Uma regra nunca é removida se for a última regra cuja classe dominante é c — caso contrário o modelo deixaria de conseguir prever c até criar uma nova regra para essa classe. Um contador regista quantas vezes a salvaguarda bloqueou uma remoção; se ficar a 0, a salvaguarda não alterou os resultados e isso é reportado.

### 3.3 Datasets

| Stream | Papel | Configuração |
|---|---|---|
| A–B–A (nosso) | drift recorrente; testa remoção prematura e recriação | A: x ~ U[0; 0,6]², y = 1[x₀ > 0,3]; B: x ~ U[0,4; 1]², y = 1[x₁ > 0,7]; A regressa; mudanças a 1/3 e 2/3 |
| RandomRBFDrift_synth | drift contínuo em P(x) e P(y\|x) | `change_speed` > 0 [a preencher], nº de centróides [a preencher] |
| ConceptDrift_Agrawal | controlo negativo: só P(y\|x) muda | funções [a preencher], posição e largura do drift [a preencher] |
| Elec2_real | dados reais; teste de transferência | prefixo usado [a preencher] |

O pruning por atividade só pode atuar quando P(x) muda; o Agrawal serve para mostrar o que acontece quando não muda.

### 3.4 Protocolo de dados

| Decisão | Valor |
|---|---|
| Ordem | cronológica, nunca baralhada; o mesmo stream para todas as variantes |
| Avaliação | prequential (test-then-train): prever x_t, depois aprender (x_t, y_t) |
| Warm-up | [a preencher] observações: o modelo aprende com elas, mas ficam fora das métricas |
| Horizonte A–B–A | [a preencher] (referência: 5 000) |
| Horizonte RBF | [a preencher] (referência: 20 000) |
| Horizonte Agrawal | [a preencher] (referência: 20 000) |
| Horizonte Elec2 | [a preencher] (referência: 10 000) |
| Pré-processamento | min-max para [0, 1], igual ao baseline e igual para todas as variantes |
| Limitação | o min/max global usa informação do futuro (leakage leve, igual em todas as variantes) |
| Seeds de desenvolvimento (tuning) | [a preencher] — conjunto separado das de avaliação |
| Seeds de avaliação (teste) | [a preencher] — usadas uma única vez |
| Repetições | [a preencher] por variante em cada stream sintético, uma por seed de avaliação, emparelhadas → 4 variantes × R repetições × 3 streams = [a preencher] corridas |
| Elec2 | 1 corrida por variante (ordem fixa e modelo determinístico); não usado para tuning |
| Determinismo | o modelo Safe é determinístico (`random_state` não é usado); a variabilidade vem só do gerador do stream |

### 3.5 Margem de degradação e critério de tuning

**O que é 1 p.p.** "p.p." quer dizer *ponto percentual*: é a diferença **absoluta** entre duas accuracies escritas em percentagem. Se o B0 tem 76,5 % de accuracy, uma variante com perda ≤ 1 p.p. tem de ter **pelo menos 75,5 %**. Não é o mesmo que "1 %" relativo: 1 % de 76,5 % seria só 0,77 p.p. (limite de 75,7 %). Usamos a diferença absoluta porque é mais fácil de ler e não depende do nível de accuracy de cada stream.

**Margem de degradação aceitável.** Uma variante "não degrada" se a sua accuracy prequential média ficar no máximo 1 p.p. abaixo da do B0 **no mesmo stream e com as mesmas seeds**. É a resposta prática à pergunta "o pruning custa demasiado?": abaixo de 1 p.p., a perda considera-se pequena face à redução de regras. Na avaliação final, esta margem é lida em conjunto com os testes estatísticos da [secção 7](#7-análise-estatística) — um teste pode dar diferença significativa mas menor do que 1 p.p. (real mas pequena), ou não significativa mas maior (pouca evidência).

**Critério de tuning.** Para cada variante e cada stream, nas seeds de desenvolvimento:
1. corre-se B0 e cada combinação da grelha;
2. calcula-se, para cada combinação, a accuracy média e o número médio de regras sobre as seeds de desenvolvimento;
3. ficam elegíveis as combinações com `accuracy ≥ accuracy(B0) − 1 p.p.`;
4. entre as elegíveis, escolhe-se a com **menos regras em média** (maior compressão);
5. se nenhuma for elegível: [a preencher — p.ex. escolher a de menor perda de accuracy e registar].

Assim o tuning não escolhe simplesmente a configuração mais precisa nem a mais compacta, mas a mais compacta entre as que respeitam a margem.

| Decisão | Valor |
|---|---|
| Margem | 1 p.p. de accuracy absoluta face a B0 |
| Tamanho das grelhas | 6 a 9 combinações por variante |
| Regra para os limites | ver [secção 5](#5-grelhas-de-hiperparâmetros) |

---

## 4. Como correr

O projeto usa **uv** para gerir o ambiente (`pyproject.toml` + `uv.lock`). Os comandos correm sempre através do `uv run` e **a partir da raiz do repositório**, com `python -m` (como módulo). Correr o ficheiro diretamente (`python t09/run_experiment.py` ou o botão ▶ do VS Code) dá `ModuleNotFoundError: No module named 't09'`, porque o Python passa a procurar imports dentro de `t09/`.

```bash
# instalar / sincronizar o ambiente (uma vez, ou quando o uv.lock mudar)
uv sync

# uma corrida: stream, variante, seed, nº de observações e hiperparâmetros
uv run python -m t09.run_experiment --stream aba --variant S2 --seed 100 --n 5000 --warmup 50 --param H=30 --param k=5

# MVE: as quatro variantes no A-B-A
uv run python -m t09.run_experiment --stream aba --variant B0 --seed 100 --n 5000 --warmup 50
uv run python -m t09.run_experiment --stream aba --variant S1 --seed 100 --n 5000 --warmup 50 --param H=100
uv run python -m t09.run_experiment --stream aba --variant S2 --seed 100 --n 5000 --warmup 50 --param H=30 --param k=5
uv run python -m t09.run_experiment --stream aba --variant S3 --seed 100 --n 5000 --warmup 50 --param alpha=0.01 --param theta=0.3

# diagnóstico para as grelhas (B0 sem pruning)
uv run python -m t09.diagnostics --n 5000 --seed 100

# adicionar uma dependência (ex.: river para os streams)
uv add river
```

Argumentos do `run_experiment`:

| Argumento | Significado |
|---|---|
| `--stream`, `--variant`, `--seed`, `--n` | obrigatórios; variantes: `B0`, `S1`, `S2`, `S3` |
| `--param nome=valor` | um por hiperparâmetro (repetível); os não indicados usam o valor por defeito |
| `--max-rules` | por defeito 1000 |
| `--warmup` | só para o resumo do `.json`; o CSV guarda todas as observações |
| `--out` | pasta de resultados (por defeito `results`) |
| `--overwrite` | permite substituir um resultado já existente (sem ele, o script recusa) |

Estrutura de pastas:

```
UnimNeuron-EvolvingNF/
├── ENF_UnimNeuron_Safe.py      # B0 — não mexer
├── t09/
│   ├── __init__.py
│   ├── prunning.py             # CountingMixin, B0Counted, S1Fixed, S2Adaptive, S3EMA
│   ├── streams.py              # geradores de streams + registo STREAMS
│   ├── run_experiment.py       # uma corrida prequential → CSV + .json
│   ├── diagnostics.py          # diagnóstico das grelhas (B0 sem pruning)
│   ├── metrics.py              # métricas, compactação e gráficos
│   └── stats.py                # Wilcoxon, Holm e Friedman
├── results/<stream>/<variante>[_<params>]_seed<k>.csv   (+ .json)
├── results/                    # Resultados das experiências
├── evaluation/                 # Métricas, tabelas e gráficos
└── README.md                   # Documentação do projeto
```

### Ficheiros

**`streams.py`** — cada stream é uma função `(n, seed) → (X, y, info)`, em ordem cronológica. `X` já pré-processado (min-max para [0, 1]), `y` com classes 0, 1, …, e `info` com pelo menos `n_features`, `n_classes` e `drift_points` (lista vazia se não houver). Para acrescentar um stream, escreve-se a função e regista-se no dicionário `STREAMS`; o runner não muda.

**`run_experiment.py`** — corre uma experiência test-then-train: para cada observação, prevê (`predict`) e só depois aprende (`partial_fit`). Os contadores do modelo são acumulados; as colunas `_now` são a diferença antes/depois de cada observação.
- o nome do ficheiro inclui os hiperparâmetros (ex.: `S2_H30_k5_seed100`), para corridas de tuning diferentes não se sobreporem;
- o `.json` guarda: stream e parâmetros do gerador, seed, n, variante, `max_rules`, os parâmetros passados (`params`) **e os efetivamente usados**, incluindo valores por defeito (`params_effective`), um resumo rápido, o commit do git, a data e as versões de Python e numpy;
- `valid = false` se houve remoções por capacidade (corrida inválida para a comparação principal).

**`prunning.py`** — as variantes (ver [secção 3.2](#32-variantes)).

**`diagnostics.py`** — ver [secção 5](#5-grelhas-de-hiperparâmetros).

**`metrics.py`** — analisa os CSVs gerados pelo `run_experiment.py` e calcula métricas preditivas (accuracy, macro-F1), estruturais (número médio, máximo e final de regras, compression ratio, pruning e recriações) e computacionais (runtime e tempo por observação). Exclui o warm-up, verifica a validade das corridas e gera `summary_runs.csv` e gráficos de evolução das regras e accuracy em janela móvel.

**`stats.py`** — realiza comparações estatísticas emparelhadas entre B0, S1, S2 e S3, utilizando os resultados de `summary_runs.csv`. Implementa Wilcoxon signed-rank, correção de Holm e Friedman para comparar accuracy e número médio de regras. Quando existem dados suficientes, guarda os resultados em `paired_tests.csv`.

Formato de resultados (uma linha por observação):

```csv
t,y_true,y_pred,n_rules,pruned_now,created_now,recreated_now,capacity_deletions,time_s
```

`time_s` é o tempo do `predict` + `partial_fit` dessa observação.

---

## 5. Grelhas de hiperparâmetros

Regra geral: cada grelha vai de "quase nunca remove" a "remove agressivamente", com valores espaçados em escala multiplicativa. Se a melhor configuração ficar num extremo, a grelha é alargada e a alteração registada.

### Grelha comum em unidades relativas

Os parâmetros H, H₀ e 1/α são medidos em observações, e a escala de tempo das regras muda de stream para stream: um H = 100 pode ser "muito tempo" num stream e "pouco" noutro. Para usar **a mesma grelha em todos os streams**, estes parâmetros são definidos como múltiplos de uma escala de referência **s**, medida em cada stream.

**O que é o s.** Um *gap* é o número de observações entre duas ativações relevantes consecutivas da mesma regra — o tempo que uma regra em uso passa "sem ser vista" entre duas utilizações. **s é o percentil 99 de todos os gaps**, medido no diagnóstico (B0 sem pruning, seeds de desenvolvimento, antes de qualquer tuning). Ou seja: 99 % das vezes, uma regra em uso volta a ser ativada em s observações ou menos. É, portanto, a maior ausência "normal" de uma regra viva, e por isso uma unidade natural para medir inatividade: H = 3 × s quer dizer "tolerar três vezes a ausência normal mais longa". Usa-se o p99 e não a média porque o que interessa evitar é remover regras vivas, e isso decide-se pelas ausências longas, não pelas típicas.

Os parâmetros que já são relativos (k, θ) ficam iguais em todos os streams.

| Parâmetro | Regra relativa | Porquê | No A–B–A (s = 33) | Valores finais |
|---|---|---|---|---|
| H (S1) | c × s, c ∈ {1,5; 3; 6; 12; 24; 48} | acima de s (não remover regras vivas) até bem acima, com teto em 48 × s | ≈ {50, 100, 200, 400, 800, 1600} | [a preencher] |
| H₀ (S2) | c × s, c ∈ {0,6; 0,9; 1,5} | gama em que o termo adaptativo atua numa parte das regras (`k·mean_gap > H₀`) | ≈ {20, 30, 50} | [a preencher] |
| k (S2) | {3, 5, 10} (já relativo) | "quantas vezes a ausência típica da regra se tolera" | {3, 5, 10} | [a preencher] |
| 1/α (S3) | c × s, c ∈ {1,5; 3; 6} | memória da EMA mais longa do que as ausências normais e muito mais curta do que a distância entre drifts | α ≈ {0,02; 0,01; 0,005} | [a preencher] |
| θ (S3) | {0,25; 0,3; 0,35} (já relativo) | abaixo do p5 de `ema/média` das regras em uso e acima da maioria das regras obsoletas | {0,25; 0,3; 0,35} | [a preencher] |
| G (S3) | 6 × s (= maior 1/α da grelha) | a EMA de uma regra nova precisa de ≈ 1/α observações para refletir o uso real | ≈ 200 | [a preencher] |

Tamanho das grelhas: S1 6 · S2 3 × 3 = 9 · S3 3 × 3 = 9. O S1 tem só um parâmetro, por isso leva mais valores de H para ter um número de combinações parecido com S2 e S3.

Porque é que esta opção serve a pergunta de investigação: cada variante é afinada com a mesma regra em todos os streams, aplicada à escala de cada um. Se um sinal ganhar em vários streams, isso deve-se ao sinal e não a uma grelha mais favorável. A comparação entre streams é sobre padrões (qual variante ganha, quanta compressão, onde há degradação), não sobre os valores absolutos dos hiperparâmetros.

Limitações:
- o limite superior de H ("abaixo da distância entre drifts") não é relativo a s, e não existe no RBF (drift contínuo); 48 × s funciona como teto;
- o θ foi calibrado só no A–B–A; o diagnóstico do RBF e do Agrawal tem de confirmar que o p5 de `ema/média` das regras em uso continua acima de 0,35 — se não, a grelha do θ é revista e a alteração registada.

Diagnóstico por stream (B0 sem pruning, `t09/diagnostics.py`):

| Stream | Seed | Gaps p50 / p90 / p99 (= s) | Intervalo médio por regra p99 | Inatividade das regras obsoletas (p10–p50) | `ema/média` regras em uso p5 | `ema/média` regras obsoletas p30 (α = 0,01) | Distância entre drifts |
|---|---|---|---|---|---|---|---|
| A–B–A | 100 (provisória) | 3 / 11 / 33 | 23 | 136–1006 | ≈ 0,35 | 0,19 | 1 667 |
| RBF | [ ] | [ ] | [ ] | — | [ ] | — | — |
| Agrawal | [ ] | [ ] | [ ] | — | [ ] | — | [ ] |
| Elec2 | não usado para tuning | | | | | | |

Fração de regras em que o termo adaptativo do S2 atua (A–B–A, seed 100):

| H₀ | k = 3 | k = 5 | k = 10 |
|---|---|---|---|
| 20 | 39 % | 85 % | 100 % |
| 30 | 18 % | 45 % | 96 % |
| 50 | 2 % | 18 % | 66 % |
| 100 | 0 % | 2 % | 18 % |

Como ler o diagnóstico do S3: no A–B–A sabemos quais regras estão obsoletas na fase B (centro na zona que só o conceito A usa). As regras em uso quase nunca descem abaixo de ≈ 0,35 da média; as obsoletas descem muito abaixo (p30 entre 0,1 e 0,26). O θ fica entre essas duas zonas. Nos streams sem verdade conhecida (RBF, Agrawal) usa-se só a distribuição global de `ema/média`.

Notas do MVE preliminar (stream do demo, 3 000 obs., não conclusivo):
- No stream do demo, P(x) nunca muda (os pontos caem sempre em todo o quadrado), por isso nenhuma regra fica obsoleta e os `ema` ficam todos entre 0,73× e 1,2× da média; com θ ≤ 0,3 o S3 não remove nada. Isto **não** acontece no A–B–A do protocolo, onde P(x) muda. É o comportamento esperado no controlo negativo (Agrawal).
- S2 com H₀ = 100 e k = 3 é idêntico a S1: os intervalos são curtos (mediana ≈ 7), por isso k·Δ nunca ultrapassa H₀.

---

## 6. Métricas

**Preditivas** (excluem o warm-up): accuracy prequential (principal), macro-F1 e accuracy em janela móvel de 500 observações, registada a cada 100 observações.

**Estruturais:** número de regras (média, máximo, final, curva), compression ratio = regras médias da variante / regras médias de B0, eventos de pruning, regras recriadas.

**Regra recriada:** regra nova cujo centro fica a menos de σ (distância de Chebyshev) de uma regra removida nas últimas W = [a preencher] observações. Implementação (no modelo, porque o CSV não guarda centros): usa-se o **σ da regra removida** ("a nova regra caiu na zona que a regra apagada cobria?"); cada regra removida só pode ser recriada **uma vez**; o `prune_log` guarda `(t, centro, σ)` de cada remoção. A confirmar com a Pessoa 3.

**Problema com W = 500 no A–B–A:** as regras da zona A são removidas no início da fase B e o conceito A só regressa ≈ 1 600 observações depois, fora da janela — exatamente a recriação que o A–B–A devia medir não é contada (no MVE, 0–3 recriações na fase A2). Proposta: W relativo como as grelhas (ex.: W = 48 × s, ≈ 1 600 no A–B–A) ou W ≥ distância entre drifts. [a decidir]

**Recursos:** tempo total, tempo por observação, tamanho aproximado do modelo (parâmetros por regra × nº de regras + estado de monitorização: S1 1 valor, S2 3 valores, S3 2 valores por regra).

**Recuperação após drift** (A–B–A e Agrawal): accuracy antes do drift (janela anterior), queda (mínimo depois) e tempo de recuperação até a accuracy em janela voltar a [a preencher] % do valor anterior.

---

## 7. Análise estatística

- Cada par (stream, seed) é um bloco; todas as comparações são emparelhadas.
- Friedman + Nemenyi entre B0, S1, S2 e S3, para accuracy e para número médio de regras.
- Wilcoxon signed-rank de cada S contra B0, com correção de Holm, α = 0,05, lido em conjunto com a margem de 1 p.p. (ver [secção 3.5](#35-margem-de-degradação-e-critério-de-tuning)).
- Tabela final por stream: média ± desvio padrão sobre as seeds de avaliação (Elec2: valor único).
- Um teste não significativo não prova igualdade.

---

## 8. Medições já feitas

**B0, 20 000 obs., A–B–A do protocolo, seed 100 (escolhida só para esta medição, não faz parte das seeds da experiência)** (VM Linux no portátil, 4 núcleos):

| Medida | Valor |
|---|---|
| Tempo total | 77 s (3,9 ms/obs.) |
| Tempo por bloco de 5 000 | 8,7 / 18,7 / 23,7 / 26,3 s (cresce com o nº de regras) |
| Regras máx. / média / final | 139 / 101 / 139 |
| Regras criadas | 139 |
| Merges | 0 — o merge nunca fundiu regras |
| Remoções por capacidade | 0 |

**MVE: B0, S1, S2 e S3 no A–B–A** (5 000 obs., seed 100, `max_rules` = 1000, resumo com warm-up 50 — valores iniciais, **sem tuning**):

| Variante | Hiperparâmetros | Accuracy | Δ face a B0 | Regras médias | Removidas | Criadas | Recriadas (W = 500) | Bloqueios de classe | ms/obs. | Válida |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | — | **0,742** | — | 54,1 | 0 | 80 | 0 | — | 2,1 | sim |
| S1 | H = 100 | 0,729 | −1,3 p.p. | 29,5 | 68 | 114 | 11 | 0 | 1,8 | sim |
| S2 | H₀ = 30, k = 5 | 0,721 | −2,1 p.p. | 25,3 | 94 | 130 | 41 | 0 | 1,4 | sim |
| S3 | α = 0,01, θ = 0,3, G = 200 | 0,666 | −7,6 p.p. | 41,6 | 27 | 80 | 0 | 0 | 2,4 | sim |

(ms/obs. medidos no container de desenvolvimento; os valores no portátil são diferentes. O B0 teve a melhor accuracy também nas corridas feitas no portátil.)

O que o MVE valida (critérios da proposta):
1. B0 reproduz o modelo fornecido — sim (`B0Counted` idêntico ao original);
2. S1, S2 e S3 correm no mesmo stream e removem regras sem deixar o modelo vazio — sim;
3. o termo adaptativo do S2 atua (k·Δ > H₀ em parte das regras) — sim, com H₀ = 30 e k = 5 em ≈ 45 % das regras;
4. o estado EMA do S3 fica consistente com criações, merges e remoções — sim.

Primeiras observações (uma seed, sem tuning — **não são conclusões**):
- o B0 teve a melhor accuracy; nenhuma variante ficou dentro da margem de 1 p.p. com estes valores iniciais. Com o tuning, o critério escolhe a configuração mais compacta **dentro** da margem — se nenhuma combinação da grelha a respeitar, aplica-se a regra da secção 3.5;
- S1 e S2 reduzem as regras para cerca de metade com perdas de 1–2 p.p., e são mais rápidos por observação do que o B0;
- S1 e S2 criam mais regras do que o B0 (114 e 130 contra 80): parte das regras removidas é substituída por novas — o fenómeno de recriação;
- o S3 removeu 27 regras e não criou nenhuma a mais do que o B0: as zonas onde removeu regras ficam sem cobertura. Hipótese para a queda de 7,6 p.p.: o Safe só cria regras novas quando o ponto cai fora do intervalo de distâncias à média global, não quando uma zona fica descoberta;
- o S3 é mais lento por observação apesar de ter menos regras, porque atualiza o `ema` de todas as regras em cada observação.

Estimativa de custo: cada corrida de 20 000 obs. demora ≈ 77 s no pior caso (B0), por isso o tempo total ≈ nº de corridas × 77 s, dividido por até 4 se correrem em paralelo. O nº de corridas depende das seeds e repetições ainda por fixar. O Agrawal (9 features) vai ser mais lento — [a medir].

### 8.2. Avaliação preliminar com `metrics.py`

Foi realizada uma avaliação exploratória dos quatro modelos no stream A–B–A, com 5.000 observações, seed 100 e warm-up de 50 observações.

| Variante | Accuracy | Macro-F1 | Regras médias | Redução de regras |
|---|---:|---:|---:|---:|
| B0 | 74,24% | 73,68% | 54,60 | — |
| S1 | 69,56% | 69,54% | 29,15 | 46,61% |
| S2 | 68,91% | 68,91% | 28,20 | 48,36% |
| S3 | 65,47% | 65,47% | 39,57 | 27,52% |

Todas as corridas foram consideradas válidas, sem eliminações por capacidade.

Nesta configuração exploratória, S1 e S2 reduziram aproximadamente metade do número médio de regras, mas apresentaram perdas de accuracy de 4,69 e 5,33 pontos percentuais relativamente a B0. S3 obteve a maior perda preditiva, de 8,77 pontos percentuais.

Os gráficos de accuracy em janela móvel e evolução das regras mostram diferenças mais acentuadas após as mudanças de conceito, particularmente no regresso ao conceito A.

Estes resultados são preliminares, baseados numa única seed de desenvolvimento e em hiperparâmetros não otimizados. Não permitem estabelecer superioridade estatística entre variantes.

---

## 9. Resultados

### 9.1 Tuning — configuração escolhida

| Stream | S1 | S2 | S3 |
|---|---|---|---|
| A–B–A | [ ] | [ ] | [ ] |
| RBF | [ ] | [ ] | [ ] |
| Agrawal | [ ] | [ ] | [ ] |

### 9.2 Avaliação (seeds de avaliação)

| Stream | Variante | Accuracy | Macro-F1 | Regras médias | Compression ratio | Pruning | Recriadas | ms/obs. |
|---|---|---|---|---|---|---|---|---|
| A–B–A | B0 | | | | 1,00 | — | — | |
| | S1 | | | | | | | |
| | S2 | | | | | | | |
| | S3 | | | | | | | |
| RBF | … | | | | | | | |
| Agrawal | … | | | | | | | |
| Elec2 | … | | | | | | | |

### 9.3 Recuperação após drift

No stream A–B–A, o B0 apresentou melhor recuperação da accuracy após as mudanças de conceito. As variantes com pruning, especialmente S3, tiveram maiores perdas de desempenho.
O tempo de recuperação ainda não foi calculado. Futuramente, será medido até a accuracy recuperar 95% do valor anterior ao drift, utilizando várias seeds.

### 9.4 Testes estatísticos

O stats.py está implementado e inclui os testes Wilcoxon, Friedman e a correção de Holm para comparar B0, S1, S2 e S3. Atualmente, existe apenas uma seed (100), pelo que ainda
não há dados suficientes para realizar os testes.
A análise estatística será feita após as experiências com várias seeds. O teste de Nemenyi ainda falta implementar.

---

## 10. Interpretação e conclusões

[a preencher: qual sinal equilibra melhor compacidade e estabilidade; o que acontece no controlo negativo (Agrawal); remoção prematura no A–B–A; transferência para o Elec2; limitações]

---

## 11. Registo de alterações

| Data | Alteração | Motivo |
|---|---|---|
| 20/11/2026 | protocolo congelado | — |
| 20/11/2026 | N = 1; n_min = 6; salvaguarda de cobertura de classe ativa (com contador); grelha comum em unidades relativas (múltiplos de s = p99 dos gaps) | decisões de grupo no congelamento |
| 20/11/2026 | B0 nas experiências passa a ser `B0Counted` (mesmo modelo + contadores); salvaguarda de classe implementada no `_prune` comum; regra recriada calculada no modelo com o σ da regra removida | contadores exigidos pelo protocolo sem mexer no ficheiro do professor |
| 20/11/2026 | valores por defeito do S3: θ = 0,3, G = 200 (centro da grelha) | alinhar com a grelha relativa |
| 20/11/2026 | grelhas propostas a partir do diagnóstico no A–B–A: S1 H ∈ {50…1600}; S2 H₀ ∈ {20, 30, 50} × k ∈ {3, 5, 10}; S3 α ∈ {0,005; 0,01; 0,02} × θ ∈ {0,25; 0,3; 0,35}, G = 200 | as grelhas do plano inicial tinham valores fora da escala medida (H₀ = 100 desativava o termo adaptativo; θ = 0,5 removeria regras em uso) |
| 27/11/2026 | Implementados `metrics.py` e `stats.py` para avaliação preditiva, estrutural e estatística | Automatizar a análise e comparação dos modelos B0, S1, S2 e S3 |
| 27/11/2026 | Geradas métricas e gráficos da experiência A–B–A (5.000 observações, seed 100) | Obter resultados preliminares e analisar o impacto do pruning na accuracy e no número de regras |
| 27/11/2026 | Identificadas limitações na análise estatística e recuperação após drift | Definir o trabalho pendente para a avaliação final |

---

## 12. Divisão de trabalho

| Pessoa | Responsável por |
|---|---|
| João Silva — Modelos | código das variantes e runner; grelhas |
| Ricardo  — Dados e protocolo | streams e `PROTOCOL.md` |
|  Beatriz — Avaliação | métricas, checkpoints e estatística |

## Referências

- P. Angelov e X. Zhou, "On line learning fuzzy rule-based system structure from data streams", FUZZ-IEEE, 2008.
- J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy e A. Bouchachia, "A survey on concept drift adaptation", ACM Computing Surveys 46(4), 2014.
