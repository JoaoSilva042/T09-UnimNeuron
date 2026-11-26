# PROTOCOL.md — Protocolo experimental (T09, Grupo 09)
## 1. Datasets

Código dos streams: `t09/streams.py`. Cada loader recebe `(n, seed)` e devolve
`(X, y, info)` por ordem cronológica, com o X já normalizado.

| Stream | Papel | Configuração fixada | N |
| --- | --- | --- | --- |
| `aba` (A–B–A) | drift recorrente; false pruning e recriação | A: x ~ U[0; 0,6]², y = 1[x₀ > 0,3]; B: x ~ U[0,4; 1]², y = 1[x₁ > 0,7]; A regressa; drifts a 1/3 e 2/3 do stream | 5000 (provisório) |
| `rbf` (RandomRBFDrift) | drift contínuo em P(x) e P(y\|x) | 2 classes, 10 features, 50 centróides, `n_drift_centroids` = 50, `change_speed` = 0,001; `seed_model` = `seed_sample` = seed da corrida | 20000 (provisório) |
| `agrawal` (ConceptDrift_Agrawal) | controlo negativo: só P(y\|x) muda | `Agrawal(classification_function=0)` → `Agrawal(classification_function=4)`; drift em `n // 2` (observação 10000 para N = 20000); largura 1000; mesma seed nos dois geradores | 20000 (provisório) |
| `elec2` | dados reais (P(x) muda) | prefixo cronológico das primeiras N observações; sem seed | 10000 (provisório) |

### Verificações feitas

- Versão da River: 0.27.0.
- `synth.RandomRBFDrift()` tem por omissão `change_speed = 0.0` (sem drift) e
  seeds `None` (sem reprodutibilidade). Aqui usamos `change_speed = 0,001` e
  seeds explícitas.
- O `ConceptDrift_Agrawal` do baseline (`exp_sota_comparison.py`) usa
  `Agrawal(seed=1)` e `Agrawal(seed=2)` sem `classification_function`, ou seja,
  a função 0 nos dois lados: não tem drift real, só muda a seed. Aqui as funções
  são explícitas (0 → 4). Verificado: com o mesmo x, cerca de 47 % das labels
  diferem entre as duas funções.
- Elec2: `river.datasets.Elec2`, descarregado de
  `maxhalford.github.io/files/datasets/electricity.zip`; 45312 observações no
  total; 8 features (`date`, `day`, `period`, `nswprice`, `nswdemand`,
  `vicprice`, `vicdemand`, `transfer`); classificação binária (a label vem como
  `True`/`False`). Uma observação é o registo de um período de 30 minutos do
  mercado de eletricidade de Nova Gales do Sul (a feature `date` cresce ao
  longo do stream, logo P(x) muda). É usado o prefixo das primeiras N
  observações, em ordem cronológica.
- Horizontes: valores provisórios (referência do enunciado do M6). Serão
  confirmados com o tempo e o número máximo de regras da corrida de B0 com
  20 000 observações: `[PREENCHER]`.