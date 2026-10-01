"""
Etapa 1: comparação de famílias de modelos no conjunto de desenvolvimento
(80% do dataset) com validação cruzada estratificada repetida (10 folds x 3).

Também testa o efeito da engenharia de atributos (common.add_features).
Resultados documentados em 02_comparacao_modelos.md.
"""
import time

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import RESULTS_DIR, SEED, corrected_paired_ttest, dev_holdout_split, make_pipeline

N_SPLITS, N_REPEATS = 10, 3


def models():
    return {
        "Dummy (classe majoritária)": DummyClassifier(strategy="most_frequent"),
        "GaussianNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000),
        "kNN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "DecisionTree (depth=5)": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=SEED),
        "SVC (RBF)": SVC(C=1.0, gamma="scale", random_state=SEED),
        "MLP (64)": MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=2000, early_stopping=True, random_state=SEED),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X_dev, _, y_dev, _ = dev_holdout_split()
    print(f"Conjunto de desenvolvimento: {len(X_dev)} exemplos")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows, fold_scores = [], {}
    for engineered in (False, True):
        for name, model in models().items():
            pipe = make_pipeline(model, engineered=engineered)
            start = time.perf_counter()
            res = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring="accuracy", return_train_score=True, n_jobs=-1)
            elapsed = time.perf_counter() - start
            key = f"{name}{' + FE' if engineered else ''}"
            fold_scores[key] = res["test_score"]
            rows.append(
                {
                    "modelo": name,
                    "features": "originais + FE" if engineered else "originais",
                    "acc_treino": res["train_score"].mean(),
                    "acc_cv": res["test_score"].mean(),
                    "std_cv": res["test_score"].std(),
                    "gap": res["train_score"].mean() - res["test_score"].mean(),
                    "tempo_s": elapsed,
                }
            )
            print(f"{key:40s} cv={res['test_score'].mean():.4f} ± {res['test_score'].std():.4f}  ({elapsed:.1f}s)")

    table = pd.DataFrame(rows).sort_values("acc_cv", ascending=False)
    table.to_csv(RESULTS_DIR / "compare_models.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(RESULTS_DIR / "compare_models_folds.csv", index=False)
    print("\n", table.round(4).to_string(index=False))

    n_test = len(X_dev) // N_SPLITS
    n_train = len(X_dev) - n_test
    best = max(fold_scores, key=lambda k: fold_scores[k].mean())
    print(f"\nTeste t pareado corrigido (Nadeau-Bengio) contra o melhor: {best}")
    for key, scores in sorted(fold_scores.items(), key=lambda kv: -kv[1].mean()):
        if key == best:
            continue
        t, p = corrected_paired_ttest(fold_scores[best], scores, n_train, n_test)
        print(f"  {key:40s} diff={fold_scores[best].mean() - scores.mean():+.4f}  p={p:.3f}")


if __name__ == "__main__":
    main()
