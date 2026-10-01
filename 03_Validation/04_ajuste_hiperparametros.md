# 4. Ajuste de hiperparâmetros

Scripts: `tune.py` (validação cruzada aninhada) e `svc_surface.py`
(superfície do SVC e escolha final). Resultados em `results/nested_cv.csv`,
`results/best_params.json`, `results/svc_surface.csv` e
`results/final_config.json`.

## 4.1 Validação cruzada aninhada

Problema: se ajustamos os hiperparâmetros olhando a CV e reportamos a melhor
acurácia dessa mesma CV, o número sai **otimista**, porque escolhemos o máximo
de várias estimativas ruidosas. A CV aninhada separa as duas coisas:

- **Laço interno** (StratifiedKFold, 5 folds): `GridSearchCV` escolhe os
  hiperparâmetros usando só o treino do fold externo.
- **Laço externo** (5 folds × 2 repetições = 10 estimativas): mede o
  desempenho do *procedimento inteiro* ("rodar o grid e usar o melhor") em
  dados que o grid nunca viu.

O conjunto de atributos (`originais` / `razoes` / `todas`) também entra no grid.

| família | grid |
|---|---|
| SVC (RBF) | C ∈ {0,3; 1; 3; 10; 30}, gamma ∈ {0,01; 0,03; 0,1; 0,3} |
| Regressão Logística | C ∈ {0,01; 0,1; 1; 10; 100} |
| kNN | k ∈ {15; 25; 35; 50; 75}, pesos uniformes ou por distância |
| MLP | camadas ∈ {(16), (64), (64, 32)}, alpha (L2) ∈ {0,001; 0,01; 0,1; 1} |
| ExtraTrees | min_samples_leaf ∈ {5; 10; 20}, max_features ∈ {sqrt; 0,5; 1,0} |
| HistGradientBoosting | max_depth ∈ {2; 3}, max_iter ∈ {100; 300}, l2 ∈ {0; 1}, min_samples_leaf ∈ {20; 50} (lr = 0,03) |

### Resultados

| família | **acc aninhada** | desvio | melhor score interno | hiperparâmetros escolhidos (em todo o dev) |
|---|---|---|---|---|
| SVC (RBF) | **0,6677** | 0,017 | 0,6727 | C=10, gamma=0,03, todas |
| HistGradientBoosting | 0,6573 | 0,018 | 0,6615 | depth=2, 300 iter, l2=1, leaf=20, todas |
| ExtraTrees | 0,6565 | 0,018 | 0,6651 | leaf=20, max_features=1,0, razoes |
| Regressão Logística | 0,6559 | 0,013 | 0,6599 | C=100, razoes |
| kNN | 0,6495 | 0,017 | 0,6619 | k=35, distância, razoes |
| MLP | 0,6485 | 0,022 | 0,6595 | (64, 32), alpha=0,1, razoes |

Teste t corrigido (folds externos) contra o SVC: Regressão Logística
(p = 0,02) e MLP (p = 0,05) ficam significativamente abaixo; HistGB (p = 0,12),
ExtraTrees (p = 0,19) e kNN (p = 0,07) não, mas todos ficam 1 a 2 p.p. atrás.

**Observações**

- O "melhor score interno" é sempre maior que a acurácia aninhada (0,5 a
  1,2 p.p.). Esse é exatamente o otimismo de seleção que a CV aninhada
  elimina. Para kNN e ExtraTrees ele é maior, porque o grid tem mais
  configurações equivalentes.
- A regularização resolveu o sobreajuste das árvores (ExtraTrees com
  `min_samples_leaf=20`, HistGB com profundidade 2), mas elas não passaram o
  SVC.
- Quase todas as famílias preferiram `razoes` ou `todas`, o que confirma o
  ganho da engenharia de atributos.

### Ensemble

Soft e hard voting de SVC + Regressão Logística + MLP + ExtraTrees
(ajustados): **0,6639** e **0,6633** na CV 10×3, iguais ao SVC sozinho
(p = 0,96). Os modelos erram nos mesmos exemplos (a sobreposição entre as
classes 2 e 3 é intrínseca), então combiná-los não traz diversidade. O
ensemble foi descartado: mais complexo, mais lento e sem ganho.

## 4.2 Escolha final dos hiperparâmetros do SVC

Um sinal de alerta apareceu no `tune.py`: o SVC com o grid ajustado (C=10,
gamma=0,03) teve **0,6636** na CV 10×3, *abaixo* do SVC com os valores padrão
(C=1, gamma='scale' ≈ 0,07), que teve 0,6677 nos mesmos folds. O GridSearch
interno de 5 folds é ruidoso o bastante para escolher um ponto pior. Por isso,
em vez de confiar no argmax de uma CV, mapeamos a superfície inteira com CV
10×3 (8 valores de C × 7 de gamma × 3 conjuntos de atributos = 168
configurações).

Acurácia CV 10×3 com `features = razoes`:

| C \ gamma | 0,01 | 0,02 | 0,03 | 0,05 | 0,07 | 0,1 | 0,2 |
|---|---|---|---|---|---|---|---|
| 0,1 | 0,632 | 0,639 | 0,644 | 0,650 | 0,652 | 0,656 | 0,658 |
| 0,3 | 0,648 | 0,653 | 0,654 | 0,661 | 0,663 | 0,664 | 0,666 |
| 1 | 0,655 | 0,659 | 0,661 | 0,663 | 0,666 | 0,669 | 0,667 |
| 2 | 0,659 | 0,662 | 0,663 | 0,666 | 0,668 | 0,668 | 0,663 |
| **3** | 0,658 | 0,664 | 0,665 | 0,667 | **0,668** | 0,665 | 0,663 |
| 5 | 0,661 | 0,664 | 0,666 | 0,665 | 0,667 | 0,665 | 0,661 |
| 10 | 0,663 | 0,667 | 0,667 | 0,666 | 0,665 | 0,662 | 0,654 |
| 30 | 0,667 | 0,665 | 0,665 | 0,665 | 0,663 | 0,659 | 0,645 |

(`originais` fica sistematicamente cerca de 1 p.p. abaixo; `todas` fica
praticamente igual a `razoes`. As tabelas completas estão em
`results/svc_surface.log`.)

A superfície é um **platô diagonal**: C maior pede gamma menor, e vice-versa
(os dois controlam a complexidade da fronteira). Ao longo do platô, a acurácia
varia menos que o erro padrão da CV (~0,004). **28 configurações ficam dentro
de 1 erro padrão do melhor ponto** (0,6692; `todas`, C=1, gamma=0,1).

### Critério de escolha e o que foi descartado

1. *Argmax puro*: escolheria um ponto isolado cujo valor é em parte ruído.
2. *Regra de 1 erro padrão, ordenando por C e depois por gamma*: foi a
   primeira tentativa. Escolheu C=0,3 e gamma=0,2, na borda do grid. Isso é
   inconsistente, porque gamma alto também é complexidade.
3. *Regra de 1 erro padrão priorizando menos atributos*: escolheu
   `originais`, que fica abaixo em **toda** a superfície e contraria a
   ablação. Descartada.
4. **Critério adotado**: a maior média na **vizinhança 3×3** do grid (C ×
   gamma). Isso escolhe o centro do platô, onde pequenas mudanças de
   hiperparâmetro (ou de dados) mudam pouco o resultado.

Como as tentativas 2 e 3 foram feitas olhando os resultados, registramos
isso aqui. Mesmo assim, a escolha usou apenas o conjunto de desenvolvimento, e
o holdout continua intocado para medir o resultado sem viés.

**Configuração final:** SVC RBF, **C = 3, gamma = 0,07, features = `razoes`**
(originais + one-hot de `sex` + 3 razões de peso, padronizados).
CV 10×3 = 0,6679 ± 0,022; média na vizinhança = 0,6666. Sete das oito
melhores vizinhanças usam `razoes`, então a escolha do conjunto de atributos
é estável.

## Por que o SVC com kernel RBF (explicação do algoritmo)

- **SVM** procura a fronteira de decisão com **margem máxima** entre as
  classes. Só os exemplos perto da fronteira (os *vetores de suporte*)
  definem o modelo.
- **Margem suave (C)**: com classes sobrepostas não há separação perfeita.
  C é o custo de cada violação da margem: C pequeno gera uma margem larga e
  tolerante (mais viés), C grande tenta acertar todo exemplo de treino (mais
  variância).
- **Kernel RBF**: K(x, x') = exp(−gamma · ‖x − x'‖²). Equivale a projetar os
  dados num espaço de dimensão infinita onde uma fronteira linear corresponde
  a uma fronteira suave e não linear no espaço original. gamma define o
  "alcance" de cada exemplo: gamma pequeno gera fronteiras suaves, gamma
  grande gera fronteiras muito locais.
- **Multiclasse**: o `SVC` do scikit-learn usa *one-vs-one*: treina 3
  classificadores binários (1×2, 1×3, 2×3) e decide por votação.
- **Por que funciona bem aqui**: as classes formam regiões contínuas e
  ordenadas no espaço de atributos padronizados, com uma fronteira 2↔3 difusa.
  A margem suave lida bem com a sobreposição sem memorizar ruído (gap
  treino-CV de ~2 p.p.), ao contrário das árvores sem regularização.
- **Cuidados**: a padronização é obrigatória (a distância euclidiana está
  dentro do kernel), e o custo de treino cresce entre O(n²) e O(n³), o que não
  é problema com ~3 mil exemplos.
