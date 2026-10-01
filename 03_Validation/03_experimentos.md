# 3. Experimentos: ablação de atributos e abordagem ordinal

Script: `experiments.py` (resultados em `results/ablation.csv` e
`results/ordinal.csv`). Mesmo protocolo: CV 10×3 no conjunto de
desenvolvimento, SVC RBF com C=1 e gamma='scale'.

## Ablação dos atributos criados

Referência: SVC com todos os 6 atributos criados = **0,6677**; sem nenhum =
0,6551.

| configuração | acc CV | Δ contra "todos" |
|---|---|---|
| sem `shucked_ratio` | 0,6619 | **−0,0059** |
| sem `shell_ratio` | 0,6647 | −0,0031 |
| sem `viscera_ratio` | 0,6649 | −0,0028 |
| sem `density` | 0,6692 | +0,0015 |
| sem `log_whole_weight` | 0,6680 | +0,0003 |
| sem `log_shell_weight` | 0,6680 | +0,0003 |
| todos, mas **sem `sex`** | 0,6615 | −0,0063 |

Adicionando **um único** atributo aos originais:

| atributo adicionado | acc CV |
|---|---|
| `shucked_ratio` | 0,6611 |
| `shell_ratio` | 0,6603 |
| `viscera_ratio` | 0,6583 |
| `density` | 0,6573 |
| `log_whole_weight` | 0,6565 |
| `log_shell_weight` | 0,6556 |

**Conclusões**

- O ganho vem das **três razões entre pesos**, principalmente a fração de carne
  (`shucked_ratio`), o que confirma a hipótese da análise exploratória:
  proporções carregam informação de idade que o tamanho absoluto não carrega.
- `density` e os logs não acrescentam nada (Δ ≈ 0, bem dentro do erro padrão
  de ~0,004). Os logs são quase redundantes para um kernel RBF, que já é não
  linear.
- `sex` continua útil mesmo com as razões (−0,6 p.p. ao remover).
- A partir daqui, o conjunto de atributos virou um hiperparâmetro com três
  opções: `originais`, `razoes` (originais + 3 razões) e `todas` (originais +
  6 criados).

## Abordagem ordinal (Frank & Hall, 2001)

As classes são faixas de idade ordenadas (1 < 2 < 3). Em vez de um
classificador multiclasse, treinamos dois classificadores binários,
P(y > 1) e P(y > 2), e combinamos:
P(1) = 1 − P(y>1), P(2) = P(y>1) − P(y>2), P(3) = P(y>2).
A ideia é que o modelo nunca "pule" da classe 1 para a 3 sem passar pela 2.

| modelo base | multiclasse | ordinal | Δ |
|---|---|---|---|
| Regressão Logística | 0,6556 | 0,6591 | +0,0035 |
| SVC (RBF) | 0,6677 | 0,6639 | −0,0039 |

**Conclusão:** nenhum efeito relevante (±0,4 p.p., dentro do ruído). A matriz
de confusão final (`05_avaliacao_final.md`) mostra que confusões 1↔3 já são
raras no multiclasse direto: o modelo aprende a ordem sozinho, porque o
tamanho cresce com a classe. A abordagem foi descartada por não compensar a
complexidade extra.
