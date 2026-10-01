# 2. Protocolo de validação e comparação de modelos

Script: `compare_models.py` (resultados em `results/compare_models.csv` e
`results/compare_models_folds.csv`).

## Protocolo de validação (vale para todas as etapas)

```
abalone_dataset.csv (3132)
├── desenvolvimento: 80% = 2505 exemplos   → toda comparação, ajuste e escolha
└── holdout:        20% =  627 exemplos    → trancado; usado UMA vez no final
```

- **Divisão estratificada** por `type`, com semente fixa (`SEED = 42` em
  `common.py`), então todos os scripts veem exatamente a mesma divisão.
- **Holdout trancado**: nenhuma decisão (modelo, atributos, hiperparâmetros)
  usa o holdout. Ele só serve para confirmar, no fim, que a estimativa da CV
  não ficou otimista pela quantidade de comparações feitas.
- **Validação cruzada estratificada repetida** (10 folds × 3 repetições = 30
  estimativas) no conjunto de desenvolvimento. Repetir reduz a variância da
  estimativa, que com ~250 exemplos por fold é considerável (desvio de ~2 p.p.
  entre folds).
- **Sem vazamento**: padronização, one-hot e criação de atributos ficam dentro
  de um `Pipeline` do scikit-learn, então o `fit` do pré-processamento acontece
  só nos folds de treino.
- **Comparação estatística**: teste t pareado **corrigido** de Nadeau & Bengio
  (2003). Os folds de uma CV compartilham exemplos de treino, então o teste t
  comum subestima a variância e acha "diferenças significativas" demais. A
  correção multiplica a variância por (1/k + n_teste/n_treino).
- **Mesmos folds para todos os modelos** (mesma `random_state`), o que permite
  o teste pareado.

## Modelos comparados

Hiperparâmetros "razoáveis", sem ajuste, para escolher quais famílias merecem
ser ajustadas:

- Baseline: `DummyClassifier` (classe majoritária)
- Probabilísticos / lineares: GaussianNB, LDA, QDA, Regressão Logística
- Baseados em instância: kNN (k=15)
- Árvores: árvore de decisão (profundidade 5), RandomForest, ExtraTrees,
  HistGradientBoosting
- Baseados em margem / redes: SVC com kernel RBF, MLP (64 neurônios)

Cada um foi rodado com os atributos originais e com os originais mais atributos
criados (FE):

| atributo | definição | intuição |
|---|---|---|
| `shell_ratio` | shell / whole | concha proporcionalmente mais pesada com a idade |
| `shucked_ratio` | shucked / whole | fração de carne |
| `viscera_ratio` | viscera / whole | fração de vísceras |
| `density` | whole / (length·diameter·height) | densidade aparente |
| `log_whole_weight`, `log_shell_weight` | log(1+x) | pesos crescem ~cubicamente com o tamanho |

## Resultados (CV 10×3 no conjunto de desenvolvimento)

| modelo | atributos | acc treino | **acc CV** | desvio | gap treino-CV |
|---|---|---|---|---|---|
| SVC (RBF) | + FE | 0,686 | **0,668** | 0,020 | 0,019 |
| Regressão Logística | + FE | 0,664 | 0,656 | 0,021 | 0,009 |
| ExtraTrees | + FE | 0,897 | 0,655 | 0,024 | 0,242 |
| SVC (RBF) | originais | 0,673 | 0,655 | 0,026 | 0,018 |
| MLP (64) | + FE | 0,665 | 0,652 | 0,026 | 0,012 |
| LDA | + FE | 0,654 | 0,650 | 0,027 | 0,004 |
| RandomForest | + FE | 0,945 | 0,650 | 0,024 | 0,295 |
| ExtraTrees | originais | 0,833 | 0,649 | 0,020 | 0,184 |
| Regressão Logística | originais | 0,654 | 0,648 | 0,028 | 0,006 |
| kNN (k=15) | + FE | 0,697 | 0,646 | 0,024 | 0,052 |
| RandomForest | originais | 0,929 | 0,644 | 0,021 | 0,285 |
| Árvore (prof. 5) | originais | 0,674 | 0,636 | 0,025 | 0,038 |
| HistGradientBoosting | + FE | 1,000 | 0,633 | 0,022 | 0,367 |
| LDA | originais | 0,637 | 0,632 | 0,026 | 0,004 |
| MLP (64) | originais | 0,640 | 0,631 | 0,037 | 0,009 |
| HistGradientBoosting | originais | 0,986 | 0,628 | 0,027 | 0,358 |
| kNN (k=15) | originais | 0,681 | 0,627 | 0,028 | 0,054 |
| QDA | originais | 0,630 | 0,622 | 0,027 | 0,009 |
| GaussianNB | + FE | 0,602 | 0,598 | 0,038 | 0,004 |
| GaussianNB | originais | 0,568 | 0,567 | 0,027 | 0,001 |
| Dummy | – | 0,344 | 0,344 | 0,001 | 0,000 |

Teste t corrigido contra o melhor (SVC + FE): a diferença é significativa
(p < 0,05) contra quase todos, **exceto** ExtraTrees + FE (p = 0,07) e
MLP + FE (p = 0,11).

## Conclusões

1. **Todos os modelos razoáveis ficam entre 0,63 e 0,67.** O teto parece vir
   dos dados (sobreposição entre as classes 2 e 3), não do algoritmo.
2. **A engenharia de atributos ajuda de forma consistente**, com ganho de +0,5
   a +3 p.p. em 9 dos 11 modelos não triviais. As exceções foram a árvore de
   decisão (−0,2 p.p., dentro do ruído) e a QDA (−2,3 p.p., provavelmente
   pela colinearidade extra na estimativa das covariâncias por classe).
3. **Os modelos de árvore sobreajustam muito com os padrões**: RandomForest e
   HistGradientBoosting chegam a 0,94–1,00 no treino e 0,63–0,65 na CV. Vale
   ajustar a regularização deles (`min_samples_leaf`, `max_depth`) antes de
   descartá-los.
4. **SVC RBF**: melhor média, gap pequeno (bem regularizado) e rápido. É o
   principal candidato.

Próximos passos: entender quais atributos criados importam e testar a
estrutura ordinal (`03_experimentos.md`), depois ajustar os hiperparâmetros das
famílias mais promissoras (`04_ajuste_hiperparametros.md`).
