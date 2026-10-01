"""
Etapa 2: experimentos pontuais no conjunto de desenvolvimento.

1. Ablação dos atributos criados (remove um de cada vez) usando SVC RBF.
2. Abordagem ordinal (Frank & Hall, 2001): as classes são faixas de idade
   ordenadas, então treinamos dois classificadores binários P(y>1) e P(y>2)
   e combinamos em P(y=k).

Resultados documentados em 03_experimentos.md.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from common import CAT_COLS, ENG_COLS, NUM_COLS, RESULTS_DIR, SEED, add_features, dev_holdout_split, make_pipeline


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.models_ = [clone(self.estimator).fit(X, (y > c).astype(int)) for c in self.classes_[:-1]]
        return self

    def predict_proba(self, X):
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        cum = np.hstack([np.ones((len(X), 1)), gt, np.zeros((len(X), 1))])
        return np.clip(cum[:, :-1] - cum[:, 1:], 0, None)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def pipeline_with(cols, model):
    columns = ColumnTransformer(
        [
            ("num", StandardScaler(), cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    return Pipeline([("features", FunctionTransformer(add_features)), ("columns", columns), ("model", model)])


def main():
    X_dev, _, y_dev, _ = dev_holdout_split()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)

    def score(pipe):
        s = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring="accuracy", n_jobs=-1)
        return s.mean(), s.std()

    svc = SVC(C=1.0, gamma="scale", random_state=SEED)

    print("Ablação de atributos (SVC RBF)")
    rows = []
    full = score(pipeline_with(NUM_COLS + ENG_COLS, svc))
    rows.append({"config": "todos", "acc_cv": full[0], "std": full[1]})
    rows.append({"config": "sem FE", "acc_cv": score(pipeline_with(NUM_COLS, svc))[0], "std": np.nan})
    for col in ENG_COLS:
        m, s = score(pipeline_with(NUM_COLS + [c for c in ENG_COLS if c != col], svc))
        rows.append({"config": f"sem {col}", "acc_cv": m, "std": s})
    for col in ENG_COLS:
        m, s = score(pipeline_with(NUM_COLS + [col], svc))
        rows.append({"config": f"só {col}", "acc_cv": m, "std": s})
    no_sex = Pipeline(
        [
            ("features", FunctionTransformer(add_features)),
            ("columns", ColumnTransformer([("num", StandardScaler(), NUM_COLS + ENG_COLS)])),
            ("model", svc),
        ]
    )
    m, s = score(no_sex)
    rows.append({"config": "todos sem sex", "acc_cv": m, "std": s})
    ablation = pd.DataFrame(rows)
    ablation["delta"] = ablation["acc_cv"] - full[0]
    print(ablation.round(4).to_string(index=False))
    ablation.to_csv(RESULTS_DIR / "ablation.csv", index=False)

    print("\nOrdinal (Frank & Hall) x multiclasse direto, com FE")
    rows = []
    for name, base in {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000),
        "SVC (RBF)": SVC(C=1.0, gamma="scale", probability=True, random_state=SEED),
    }.items():
        m1, s1 = score(make_pipeline(base, engineered=True))
        m2, s2 = score(make_pipeline(OrdinalClassifier(base), engineered=True))
        rows.append({"modelo": name, "multiclasse": m1, "ordinal": m2, "delta": m2 - m1})
    ordinal = pd.DataFrame(rows)
    print(ordinal.round(4).to_string(index=False))
    ordinal.to_csv(RESULTS_DIR / "ordinal.csv", index=False)


if __name__ == "__main__":
    main()
