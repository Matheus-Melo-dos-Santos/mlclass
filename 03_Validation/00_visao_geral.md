# Atividade 03: validação de classificadores (Abalone)

## Resumo

| | |
|---|---|
| Modelo final | SVC com kernel RBF (C = 3, gamma = 0,07) |
| Atributos | 7 medidas + one-hot de `sex` + 3 razões de peso (`shell/whole`, `shucked/whole`, `viscera/whole`), padronizados |
| Acurácia CV 10×3 (desenvolvimento) | 0,668 ± 0,022 |
| Acurácia CV aninhada (procedimento SVC) | 0,668 ± 0,017 |
| **Acurácia no holdout** (627 exemplos, usado uma vez) | **0,678** (IC 95%: 0,641–0,715) |
| Baseline (classe majoritária) | 0,344 |
| **Acurácia no servidor** (app, 1045 exemplos) | **0,656** |

## Metodologia

1. **Holdout trancado**: 20% estratificado, separado no início e usado só na
   avaliação final.
2. **CV estratificada repetida** (10×3) no restante, com os mesmos folds para
   todos os modelos.
3. **Pipelines** (padronização, one-hot e criação de atributos) ajustados
   dentro de cada fold, sem vazamento.
4. **Teste t pareado corrigido** (Nadeau & Bengio) para decidir se as
   diferenças entre modelos são reais.
5. **CV aninhada** para estimar sem viés o desempenho após o ajuste de
   hiperparâmetros.
6. Escolha dos hiperparâmetros pelo **centro do platô** da superfície de
   acurácia, não pelo máximo isolado (ruído).
7. **Validação adversarial** para verificar se treino e app têm a mesma
   distribuição.
8. Modelo final re-treinado com **todos** os dados.

## Documentos

| arquivo | conteúdo |
|---|---|
| [01_analise_exploratoria.md](01_analise_exploratoria.md) | dados, classes, correlações, outliers, treino × app |
| [02_comparacao_modelos.md](02_comparacao_modelos.md) | protocolo de validação e comparação de 12 modelos |
| [03_experimentos.md](03_experimentos.md) | ablação dos atributos criados e abordagem ordinal |
| [04_ajuste_hiperparametros.md](04_ajuste_hiperparametros.md) | CV aninhada, ensemble, superfície do SVC e explicação do algoritmo |
| [05_avaliacao_final.md](05_avaliacao_final.md) | holdout, matriz de confusão, curva de aprendizado, previsões |

## Principais aprendizados

- **O teto está nos dados, não no algoritmo.** Os modelos razoáveis ficam
  entre 0,63 e 0,67. A classe 2 (intermediária) se confunde com a 3 porque,
  depois de adulto, o abalone quase não muda de tamanho com a idade.
- **Engenharia de atributos guiada pelo domínio** (proporções entre pesos)
  rendeu mais que trocar de algoritmo: +1,3 p.p. no SVC e ganhos em quase
  todos os modelos.
- **O argmax de um GridSearch é ruidoso.** O "melhor" do grid interno ficou
  pior que os valores padrão numa CV mais robusta. As diferenças no topo são
  menores que o erro padrão da CV.
- **Ensemble e abordagem ordinal não ajudaram**: os modelos erram nos mesmos
  exemplos.

## Como reproduzir

A partir da raiz do repositório (ambiente `pipenv`), dentro de `03_Validation`:

```
pipenv run python eda.py             # análise exploratória
pipenv run python compare_models.py  # ~1,5 min
pipenv run python experiments.py     # ~0,5 min
pipenv run python tune.py            # ~8 min (CV aninhada)
pipenv run python svc_surface.py     # ~3 min; gera results/final_config.json
pipenv run python final.py           # holdout + predictions_app.csv (NÃO envia)
pipenv run python final.py --enviar  # idem + envio (1 vez a cada 12 h)
```

Código compartilhado (divisão dos dados, pipelines, atributos e teste t
corrigido) em `common.py`. Tabelas intermediárias em `results/`.
