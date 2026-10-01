# 1. Análise exploratória

Script: `eda.py` (saída completa em `results/eda.log`).

## Visão geral

| | treino (`abalone_dataset.csv`) | aplicação (`abalone_app.csv`) |
|---|---|---|
| Linhas | 3132 | 1045 |
| Colunas | 8 atributos + `type` | 8 atributos |
| Valores faltantes | 0 | 0 |
| Duplicatas | 0 | – |

Diferente da atividade 01, não há imputação a fazer. O foco da atividade é a
**validação**.

## Classes

| type | n | % |
|---|---|---|
| 1 | 1078 | 34,4 |
| 2 | 1003 | 32,0 |
| 3 | 1051 | 33,6 |

As classes são balanceadas, então **acurácia** é uma métrica adequada (e é a
métrica do servidor). O baseline de "sempre prever a classe majoritária" fica
em ~34%.

## O que é `type`?

As médias por classe crescem monotonicamente em todos os atributos:

| type | length | diameter | height | whole_weight | shell_weight |
|---|---|---|---|---|---|
| 1 | 0,419 | 0,320 | 0,105 | 0,430 | 0,121 |
| 2 | 0,560 | 0,437 | 0,148 | 0,923 | 0,257 |
| 3 | 0,590 | 0,464 | 0,163 | 1,118 | 0,333 |

Isso bate com a versão clássica em 3 classes do dataset Abalone da UCI, em que o
número de anéis (idade) é discretizado em faixas (jovem / adulto / velho). Duas
consequências:

1. **As classes são ordinais.** Testamos uma abordagem ordinal em
   `03_experimentos.md`.
2. **As classes 2 e 3 se sobrepõem muito.** Depois que o abalone para de
   crescer, a idade quase não altera as medidas. Por exemplo, os quantis de
   `shell_weight`:

   | type | 5% | 25% | 50% | 75% | 95% |
   |---|---|---|---|---|---|
   | 1 | 0,017 | 0,058 | 0,105 | 0,167 | 0,276 |
   | 2 | 0,090 | 0,184 | 0,255 | 0,321 | 0,430 |
   | 3 | 0,140 | 0,245 | 0,320 | 0,410 | 0,570 |

   Por isso não se deve esperar acurácia muito alta: o erro irredutível
   (Bayes) é alto. Nenhum dos modelos testados passou de ~0,67 na validação
   cruzada, independentemente da família.

## Sexo

| sex | type 1 | type 2 | type 3 |
|---|---|---|---|
| F | 14,5% | 36,1% | 49,4% |
| I (infantil) | 68,8% | 20,5% | 10,7% |
| M | 19,6% | 39,1% | 41,2% |

`sex = I` é um indicador forte da classe 1. F e M têm perfis parecidos entre si.
Como é categórico sem ordem, usamos **one-hot encoding**.

## Correlação e escala

- Todos os atributos numéricos têm correlação de Pearson entre 0,83 e 0,99
  entre si (`length` × `diameter` = 0,987). Há muita redundância: o tamanho do
  animal domina tudo.
- A correlação de Spearman com `type` fica entre 0,52 (`shucked_weight`) e 0,65
  (`shell_weight`).
- As escalas diferem (`height` ~0,14, `whole_weight` ~0,82), então modelos
  baseados em distância ou margem (kNN, SVM, redes neurais) precisam de
  **padronização**, ajustada só nos dados de treino de cada fold.

### Ideia para engenharia de atributos

Como tamanho e peso são quase colineares, a informação extra sobre a idade
deve estar nas **proporções**. Por exemplo, a fração de carne
(`shucked_weight / whole_weight`) tende a cair com a idade, enquanto a concha
fica proporcionalmente mais pesada. Testado em `02_comparacao_modelos.md` e
`03_experimentos.md`.

## Valores suspeitos

| Verificação | treino | app |
|---|---|---|
| `height == 0` | 2 | 0 |
| `height > 0,4` (outlier extremo) | 1 | 1 |
| `diameter > length` | 1 | – |
| soma das partes > `whole_weight` | 117 | – |

Decisão: **não remover nada.** São poucos casos, o app também tem outliers de
`height` (máximo de 1,13), e os modelos escolhidos (SVM com padronização) são
pouco sensíveis a alguns pontos isolados. A soma das partes passar do peso
total é esperada (erro de medição e água perdida no processo), não é erro de
digitação.

## Treino × aplicação (mudança de distribuição)

- O app tem abalones ligeiramente maiores: as médias ficam 0,07 a 0,13 desvios
  padrão acima das do treino.
- O app tem menos infantis (28,7% contra 33,3% no treino). Aplicando
  P(type | sex) do treino à distribuição de sexo do app, a distribuição de
  classes esperada no app é de aproximadamente **32% / 33% / 35%**.
- **Validação adversarial**: um RandomForest treinado para distinguir linhas do
  treino das linhas do app obtém **AUC = 0,51 ± 0,02**. Treino e app são
  praticamente indistinguíveis, então a validação local deve ser um bom
  estimador do resultado no servidor.
