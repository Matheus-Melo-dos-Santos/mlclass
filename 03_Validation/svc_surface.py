"""
Etapa 4: superfície de acurácia do SVC RBF (C x gamma x conjunto de
atributos) com validação cruzada 10 folds x 3 repetições.

Como dezenas de configurações ficam dentro de 1 erro padrão da melhor, a
escolhida é a de maior média na vizinhança 3x3 do grid (C x gamma): o centro
do platô, menos sensível ao ruído da CV do que o máximo isolado.

Resultados documentados em 04_ajuste_hiperparametros.md.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

from common import FEATURE_SETS, RESULTS_DIR, SEED, describe_prep, dev_holdout_split, feature_set_grid, make_pipeline

N_SPLITS, N_REPEATS = 10, 3


def main():
    X_dev, _, y_dev, _ = dev_holdout_split()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    grid = {
        "prep": feature_set_grid(),
        "model__C": [0.1, 0.3, 1, 2, 3, 5, 10, 30],
        "model__gamma": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2],
    }
    search = GridSearchCV(make_pipeline(SVC(random_state=SEED)), grid, cv=cv, scoring="accuracy", n_jobs=-1)
    search.fit(X_dev, y_dev)

    res = pd.DataFrame(search.cv_results_)
    res["features"] = res["param_prep"].map(describe_prep)
    res["C"] = res["param_model__C"].astype(float)
    res["gamma"] = res["param_model__gamma"].astype(float)
    res["se"] = res["std_test_score"] / np.sqrt(N_SPLITS * N_REPEATS)
    res = res[["features", "C", "gamma", "mean_test_score", "std_test_score", "se"]]

    for fs in FEATURE_SETS:
        pivot = res[res.features == fs].pivot(index="C", columns="gamma", values="mean_test_score")
        print(f"\nAcurácia CV 10x3 - features={fs}")
        print(pivot.round(4).to_string())

    res["vizinhanca"] = np.nan
    for fs in FEATURE_SETS:
        pivot = res[res.features == fs].pivot(index="C", columns="gamma", values="mean_test_score")
        smooth = pivot.rolling(3, center=True, min_periods=1).mean().T.rolling(3, center=True, min_periods=1).mean().T
        for idx in res.index[res.features == fs]:
            res.loc[idx, "vizinhanca"] = smooth.loc[res.loc[idx, "C"], res.loc[idx, "gamma"]]

    res.to_csv(RESULTS_DIR / "svc_surface.csv", index=False)

    best = res.loc[res.mean_test_score.idxmax()]
    threshold = best.mean_test_score - best.se
    n_within = (res.mean_test_score >= threshold).sum()
    ranked = res.sort_values("vizinhanca", ascending=False)
    chosen = ranked.iloc[0]
    print("\nTop 8 pela média da vizinhança 3x3:")
    print(ranked.head(8).round(4).to_string(index=False))
    print(f"\nMelhor ponto isolado: {best.to_dict()}")
    print(f"Limiar 1-SE: {threshold:.4f} ({n_within} configurações dentro do limiar)")
    print(f"Escolhida (maior média na vizinhança): {chosen.to_dict()}")

    with open(RESULTS_DIR / "final_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "SVC (RBF)",
                "C": float(chosen.C),
                "gamma": float(chosen.gamma),
                "features": chosen.features,
                "cv_10x3_mean": float(chosen.mean_test_score),
                "cv_10x3_std": float(chosen.std_test_score),
                "best_raw": {k: (v if isinstance(v, str) else float(v)) for k, v in best.to_dict().items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
