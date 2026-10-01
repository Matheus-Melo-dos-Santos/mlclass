"""
Análise exploratória do dataset abalone.

Resultados documentados em 01_analise_exploratoria.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

NUM_COLS = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    train = pd.read_csv(BASE_DIR / "abalone_dataset.csv")
    app = pd.read_csv(BASE_DIR / "abalone_app.csv")

    section("Formato e valores faltantes")
    print(f"treino: {train.shape} | app: {app.shape}")
    print(f"NaN treino: {train.isna().sum().sum()} | NaN app: {app.isna().sum().sum()}")
    print(f"Duplicatas exatas (treino): {train.duplicated().sum()}")
    print(f"Duplicatas só nos atributos (treino): {train.drop(columns='type').duplicated().sum()}")

    section("Distribuição das classes")
    counts = train["type"].value_counts().sort_index()
    print(pd.DataFrame({"n": counts, "%": (100 * counts / len(train)).round(1)}))

    section("Sex x type (contagem e % por linha)")
    print(pd.crosstab(train["sex"], train["type"], margins=True))
    print((100 * pd.crosstab(train["sex"], train["type"], normalize="index")).round(1))
    print("\nSex no app:")
    print(app["sex"].value_counts())

    section("Estatísticas descritivas (treino)")
    print(train[NUM_COLS].describe().T.round(4))
    section("Estatísticas descritivas (app)")
    print(app[NUM_COLS].describe().T.round(4))

    section("Diferença de distribuição treino x app (média e desvio padronizados)")
    diff = (app[NUM_COLS].mean() - train[NUM_COLS].mean()) / train[NUM_COLS].std()
    print(diff.round(3))

    section("Média dos atributos por classe")
    print(train.groupby("type")[NUM_COLS].mean().round(4))

    section("Valores suspeitos")
    print(f"height == 0 -> treino: {(train.height == 0).sum()}, app: {(app.height == 0).sum()}")
    print(f"height > 0.4 -> treino: {(train.height > 0.4).sum()}, app: {(app.height > 0.4).sum()}")
    parts = train["shucked_weight"] + train["viscera_weight"] + train["shell_weight"]
    print(f"soma das partes > whole_weight (treino): {(parts > train.whole_weight).sum()}")
    print(f"diameter > length (treino): {(train.diameter > train.length).sum()}")

    section("Outliers pelo critério IQR (1,5x) por atributo (treino)")
    q1, q3 = train[NUM_COLS].quantile(0.25), train[NUM_COLS].quantile(0.75)
    iqr = q3 - q1
    out = ((train[NUM_COLS] < q1 - 1.5 * iqr) | (train[NUM_COLS] > q3 + 1.5 * iqr)).sum()
    print(out)

    section("Correlação de Pearson entre atributos numéricos")
    print(train[NUM_COLS].corr().round(3))

    section("Correlação de Spearman atributo x type")
    print(train[NUM_COLS + ["type"]].corr(method="spearman")["type"].drop("type").round(3))

    section("Sobreposição entre classes: faixa de shell_weight por classe (quantis)")
    print(train.groupby("type")["shell_weight"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack().round(4))

    section("Distribuição de classes esperada no app a partir de P(type | sex) do treino")
    p_type_given_sex = pd.crosstab(train["sex"], train["type"], normalize="index")
    expected = (app["sex"].value_counts(normalize=True).reindex(p_type_given_sex.index) @ p_type_given_sex) * 100
    print(expected.round(1))

    section("Validação adversarial: treino x app são distinguíveis?")
    adversarial_validation(train.drop(columns="type"), app)


def adversarial_validation(train_X, app_X):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = pd.get_dummies(pd.concat([train_X, app_X], ignore_index=True), columns=["sex"], dtype=float)
    origin = np.r_[np.zeros(len(train_X)), np.ones(len(app_X))]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(clf, X, origin, cv=cv, scoring="roc_auc")
    print(f"AUC (RandomForest, 5 folds): {auc.mean():.4f} ± {auc.std():.4f}  (0,5 = indistinguíveis)")


if __name__ == "__main__":
    np.random.seed(0)
    main()
