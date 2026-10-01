# 5. Avaliação final no holdout e previsões

Script: `final.py` (saída em `results/final.log` e `results/holdout.json`;
previsões em `predictions_app.csv`).

## Resultado no holdout (627 exemplos, usados uma única vez)

Modelo treinado nos 2505 exemplos de desenvolvimento, com a configuração
fixada em `results/final_config.json` antes de olhar o holdout.

| métrica | valor |
|---|---|
| **Acurácia no holdout** | **0,6778** |
| IC 95% (bootstrap, 2000 reamostragens) | [0,641; 0,715] |
| Acurácia esperada pela CV 10×3 | 0,6679 ± 0,022 |
| Acurácia aninhada do procedimento SVC | 0,6677 ± 0,017 |
| Acurácia no treino (dev) | 0,6898 |
| Baseline (classe majoritária) | 0,3445 |

O holdout concorda com a CV: a diferença (+1 p.p.) está bem dentro da
variação entre folds, e o IC contém a estimativa da CV. **Não há sinal de que
as várias comparações feitas no conjunto de desenvolvimento tenham gerado uma
estimativa otimista.** A estimativa mais honesta para o servidor é algo em
torno de **0,66–0,68** (ver incertezas abaixo).

### Matriz de confusão (holdout)

| | prev 1 | prev 2 | prev 3 |
|---|---|---|---|
| **real 1** | 165 | 44 | 7 |
| **real 2** | 31 | 119 | 51 |
| **real 3** | 13 | 56 | 141 |

| classe | precisão | recall | F1 |
|---|---|---|---|
| 1 | 0,790 | 0,764 | 0,777 |
| 2 | 0,543 | 0,592 | 0,567 |
| 3 | 0,709 | 0,671 | 0,690 |

- A classe 2 (intermediária) é a mais difícil: fica "espremida" entre as
  outras duas e recebe erros dos dois lados.
- Confusões entre classes não adjacentes (1↔3) são raras (20 de 627), então o
  modelo respeita a ordem das classes sem precisar de uma formulação ordinal.
- Por sexo: infantis têm 0,784 de acurácia (a maioria é classe 1), enquanto F
  e M ficam em ~0,61, onde está a confusão entre 2 e 3.

## Curva de aprendizado (CV 10 folds no desenvolvimento)

| n treino | acc treino | acc validação | gap |
|---|---|---|---|
| 225 | 0,717 | 0,622 | 0,095 |
| 515 | 0,707 | 0,645 | 0,062 |
| 805 | 0,707 | 0,655 | 0,052 |
| 1094 | 0,715 | 0,668 | 0,047 |
| 1384 | 0,709 | 0,662 | 0,047 |
| 1674 | 0,706 | 0,664 | 0,042 |
| 1964 | 0,697 | 0,661 | 0,036 |
| 2254 | 0,693 | 0,668 | 0,025 |

A validação estabiliza em ~0,66–0,67 a partir de ~1000 exemplos, e o gap
continua caindo. O modelo está perto do teto imposto pelos dados (viés
dominante, pouca variância): mais dados ajudariam pouco, e modelos mais
complexos não ajudaram (ver etapas 2 e 4). Treinar com os 3132 exemplos no
final é seguro e deve ajudar um pouco.

## Modelo final e previsões do app

- O pipeline final (mesma configuração) foi **re-treinado com todos os 3132
  exemplos** (desenvolvimento + holdout). O holdout já cumpriu seu papel de
  estimar o desempenho; descartar 20% dos dados no modelo final só pioraria o
  resultado.
- Previsões dos 1045 exemplos do app estão em `predictions_app.csv`.

Distribuição das classes previstas:

| type | treino real | esperado no app* | previsto no app |
|---|---|---|---|
| 1 | 34,4% | 32,0% | 29,2% |
| 2 | 32,0% | 32,8% | 33,7% |
| 3 | 33,6% | 35,2% | 37,1% |

\* aplicando P(type | sex) do treino às proporções de sexo do app.

A tendência está correta (o app tem menos infantis e abalones um pouco
maiores, portanto menos classe 1). O modelo exagera um pouco essa tendência,
o que é esperado de um classificador que usa tamanho: na validação
adversarial (AUC = 0,51) treino e app são praticamente indistinguíveis, então
não há razão para corrigir as previsões.

## Incertezas e riscos

- Com 1045 exemplos no app, o erro padrão de uma acurácia de ~0,67 é
  √(0,67·0,33/1045) ≈ **1,5 p.p.** Resultados no servidor entre ~0,64 e ~0,70
  são compatíveis com o modelo ter o desempenho estimado.
- Se o servidor avalia apenas parte do app (placar público), o intervalo é
  ainda mais largo.

## Envio

**Resultado no servidor (30/09/2026): acurácia = 0,6565.**

| estimativa | acurácia |
|---|---|
| CV 10×3 (desenvolvimento) | 0,668 ± 0,022 |
| CV aninhada | 0,668 ± 0,017 |
| Holdout | 0,678 (IC 95%: 0,641–0,715) |
| **Servidor (app)** | **0,656** |

O resultado ficou dentro do intervalo previsto (~0,64–0,70) e a 1,2 p.p. da
CV, menos que 1 desvio padrão entre folds e perto do erro padrão de ~1,5 p.p.
de uma amostra de 1045 exemplos. O holdout ficou do lado otimista da
variação; a CV repetida foi o melhor estimador. A validação cumpriu seu papel:
não houve surpresa entre a estimativa local e o servidor.

As duas primeiras tentativas (28 e 29/09) foram recusadas pelo servidor com o
erro 101 ("Erro ao verificar o desenvolvedor"), um problema do lado do
servidor que foi resolvido depois.

Para reenviar: preencher `DEV_KEY` em `.env` (modelo em `.env-example`) e
rodar `pipenv run python final.py --enviar`. Sem `--enviar`, o script só faz
a avaliação local e gera o CSV.
