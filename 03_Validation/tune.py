"""
Etapa 3: ajuste de hiperparâmetros com validação cruzada aninhada.

- Laço externo (5 folds x 2 repetições): estima, sem viés de seleção, o
  desempenho de "rodar o GridSearch e usar o melhor modelo".
- Laço interno (5 folds): escolhe os hiperparâmetros (GridSearchCV).

O conjunto de atributos (originais / razões / todas) é tratado como mais um
hiperparâmetro. Depois, cada família é reajustada em todo o conjunto de
desenvolvimento para obter os hiperparâmetros finais, salvos em
results/best_params.json. Também avalia um ensemble (soft voting).

Resultados documentados em 04_ajuste_hiperparametros.md.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from common import (
    RESULTS_DIR,
    SEED,
    corrected_paired_ttest,
    describe_prep,
    dev_holdout_split,
    feature_set_grid,
    make_pipeline,
)


def search_spaces():
    fs = feature_set_grid()
    return {
        "SVC (RBF)": (
            SVC(random_state=SEED),
            {
                "prep": fs,
                "model__C": [0.3, 1, 3, 10, 30],
                "model__gamma": [0.01, 0.03, 0.1, 0.3],
            },
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=5000),
            {"prep": fs, "model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"prep": fs, "model__n_neighbors": [15, 25, 35, 50, 75], "model__weights": ["uniform", "distance"]},
        ),
        "MLP": (
            MLPClassifier(max_iter=3000, early_stopping=True, random_state=SEED),
            {
                "prep": fs,
                "model__hidden_layer_sizes": [(16,), (64,), (64, 32)],
                "model__alpha": [1e-3, 1e-2, 1e-1, 1.0],
            },
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
            {
                "prep": fs,
                "model__min_samples_leaf": [5, 10, 20],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(learning_rate=0.03, random_state=SEED),
            {
                "prep": fs,
                "model__max_depth": [2, 3],
                "model__max_iter": [100, 300],
                "model__l2_regularization": [0.0, 1.0],
                "model__min_samples_leaf": [20, 50],
            },
        ),
    }


def readable(params):
    out = {k.replace("model__", ""): v for k, v in params.items() if k != "prep"}
    out["features"] = describe_prep(params["prep"])
    return out


def main():
    X_dev, _, y_dev, _ = dev_holdout_split()
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    final_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    nested_rows, nested_folds, best_params, final_folds = [], {}, {}, {}
    for name, (model, grid) in search_spaces().items():
        start = time.perf_counter()
        search = GridSearchCV(make_pipeline(model), grid, cv=inner, scoring="accuracy", n_jobs=-1)
        nested = cross_val_score(search, X_dev, y_dev, cv=outer, scoring="accuracy")
        nested_folds[name] = nested

        search.fit(X_dev, y_dev)
        best_params[name] = readable(search.best_params_)
        tuned = search.best_estimator_
        final_folds[name] = cross_val_score(clone(tuned), X_dev, y_dev, cv=final_cv, scoring="accuracy", n_jobs=-1)
        elapsed = time.perf_counter() - start

        nested_rows.append(
            {
                "modelo": name,
                "acc_nested": nested.mean(),
                "std_nested": nested.std(),
                "acc_inner_best": search.best_score_,
                "acc_cv_10x3_tunado": final_folds[name].mean(),
                "params": best_params[name],
                "tempo_s": elapsed,
            }
        )
        print(
            f"{name:22s} nested={nested.mean():.4f} ± {nested.std():.4f}  "
            f"inner_best={search.best_score_:.4f}  10x3={final_folds[name].mean():.4f}  "
            f"{best_params[name]}  ({elapsed:.0f}s)"
        )

    table = pd.DataFrame(nested_rows).sort_values("acc_nested", ascending=False)
    table.to_csv(RESULTS_DIR / "nested_cv.csv", index=False)
    print("\n", table.drop(columns="params").round(4).to_string(index=False))

    n_test = len(X_dev) // 5
    best = table.iloc[0]["modelo"]
    print(f"\nTeste t pareado corrigido (CV externa) contra {best}:")
    for name in table["modelo"].iloc[1:]:
        _, p = corrected_paired_ttest(nested_folds[best], nested_folds[name], len(X_dev) - n_test, n_test)
        print(f"  {name:22s} diff={nested_folds[best].mean() - nested_folds[name].mean():+.4f}  p={p:.3f}")

    print("\nEnsemble soft voting com os hiperparâmetros ajustados (CV 10x3):")
    spaces = search_spaces()
    members = ["SVC (RBF)", "LogisticRegression", "MLP", "ExtraTrees"]
    estimators = []
    for name in members:
        model, _ = spaces[name]
        params = {k: v for k, v in best_params[name].items() if k != "features"}
        if name == "SVC (RBF)":
            params["probability"] = True
        estimators.append((name, make_pipeline(clone(model).set_params(**params), engineered=best_params[name]["features"])))
    for voting in ("soft", "hard"):
        ens = VotingClassifier(estimators, voting=voting)
        scores = cross_val_score(ens, X_dev, y_dev, cv=final_cv, scoring="accuracy", n_jobs=-1)
        _, p = corrected_paired_ttest(scores, final_folds["SVC (RBF)"], len(X_dev) * 0.9, len(X_dev) * 0.1)
        print(f"  voting={voting}: {scores.mean():.4f} ± {scores.std():.4f}  (vs SVC tunado: {scores.mean() - final_folds['SVC (RBF)'].mean():+.4f}, p={p:.3f})")
        final_folds[f"Voting ({voting})"] = scores

    pd.DataFrame(final_folds).to_csv(RESULTS_DIR / "tuned_cv_folds.csv", index=False)
    with open(RESULTS_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else str(o))


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
