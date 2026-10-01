"""
Etapa 5: avaliação final e geração das previsões.

1. Treina a configuração escolhida (results/final_config.json) no conjunto de
   desenvolvimento e avalia UMA vez no holdout (20% separado desde o início).
2. Curva de aprendizado no conjunto de desenvolvimento.
3. Treina com todo o dataset e prevê abalone_app.csv -> predictions_app.csv.
4. Só envia para o servidor com a flag --enviar (limite de 1 envio a cada 12h).

Uso:
    python final.py            # avaliação local + previsões (não envia)
    python final.py --enviar   # idem + envio para o servidor

Resultados documentados em 05_avaliacao_final.md.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.svm import SVC

from common import BASE_DIR, RESULTS_DIR, SEED, dev_holdout_split, load_app, load_train, make_pipeline

URL = "https://aydanomachado.com/mlclass/03_Validation.php"


def build_model(config):
    svc = SVC(C=config["C"], gamma=config["gamma"], random_state=SEED)
    return make_pipeline(svc, engineered=config["features"])


def bootstrap_ci(y_true, y_pred, n=2000, alpha=0.05):
    rng = np.random.default_rng(SEED)
    hits = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    samples = rng.choice(hits, size=(n, len(hits)), replace=True).mean(axis=1)
    return np.quantile(samples, [alpha / 2, 1 - alpha / 2])


def holdout_evaluation(config):
    X_dev, X_hold, y_dev, y_hold = dev_holdout_split()
    model = build_model(config).fit(X_dev, y_dev)
    y_pred = model.predict(X_hold)
    acc = accuracy_score(y_hold, y_pred)
    lo, hi = bootstrap_ci(y_hold, y_pred)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_dev, y_dev)

    print(f"Holdout: {len(X_hold)} exemplos (nunca usados na seleção/ajuste)")
    print(f"Acurácia no holdout: {acc:.4f}  (IC 95% bootstrap: [{lo:.4f}, {hi:.4f}])")
    print(f"Acurácia do baseline (classe majoritária): {dummy.score(X_hold, y_hold):.4f}")
    print(f"Acurácia esperada pela CV 10x3: {config['cv_10x3_mean']:.4f} ± {config['cv_10x3_std']:.4f}")
    print(f"Acurácia no treino (dev): {model.score(X_dev, y_dev):.4f}")

    labels = sorted(y_hold.unique())
    cm = pd.DataFrame(
        confusion_matrix(y_hold, y_pred, labels=labels),
        index=[f"real {c}" for c in labels],
        columns=[f"prev {c}" for c in labels],
    )
    print("\nMatriz de confusão (holdout):")
    print(cm.to_string())
    print("\nRelatório por classe (holdout):")
    print(classification_report(y_hold, y_pred, digits=4))

    print("Acurácia no holdout por sexo:")
    for sex, idx in X_hold.groupby("sex").groups.items():
        print(f"  {sex}: {accuracy_score(y_hold.loc[idx], pd.Series(y_pred, index=X_hold.index).loc[idx]):.4f}  (n={len(idx)})")

    return {"holdout_acc": acc, "holdout_ci95": [lo, hi], "confusion_matrix": cm.values.tolist()}


def learning_curve_report(config):
    X_dev, _, y_dev, _ = dev_holdout_split()
    sizes, train_s, val_s = learning_curve(
        build_model(config),
        X_dev,
        y_dev,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
        scoring="accuracy",
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "n_treino": sizes,
            "acc_treino": train_s.mean(axis=1),
            "acc_validacao": val_s.mean(axis=1),
            "std_validacao": val_s.std(axis=1),
        }
    )
    table["gap"] = table.acc_treino - table.acc_validacao
    print("\nCurva de aprendizado (CV 10 folds no conjunto de desenvolvimento):")
    print(table.round(4).to_string(index=False))
    table.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)


def predict_app(config):
    X, y = load_train()
    app = load_app()
    model = build_model(config).fit(X, y)
    y_pred = model.predict(app)

    out = pd.DataFrame({"type": y_pred})
    out.to_csv(BASE_DIR / "predictions_app.csv", index=False)
    print(f"\nModelo final treinado com {len(X)} exemplos; previsões salvas em predictions_app.csv")
    dist = pd.DataFrame(
        {
            "treino_%": (100 * y.value_counts(normalize=True)).sort_index(),
            "previsto_app_%": (100 * out.type.value_counts(normalize=True)).sort_index(),
        }
    )
    print("Distribuição das classes (treino real x previsto no app):")
    print(dist.round(1).to_string())
    return y_pred


def submit(y_pred):
    import requests
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    dev_key = os.getenv("DEV_KEY")
    if not dev_key:
        raise SystemExit("DEV_KEY não encontrada em 03_Validation/.env")
    data = {"dev_key": dev_key, "predictions": pd.Series(y_pred).to_json(orient="values")}
    r = requests.post(url=URL, data=data)
    print(" - Resposta do servidor:\n", r.text, "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enviar", action="store_true", help="envia as previsões para o servidor")
    args = parser.parse_args()

    with open(RESULTS_DIR / "final_config.json", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Configuração final: {config}\n")

    summary = holdout_evaluation(config)
    learning_curve_report(config)
    y_pred = predict_app(config)

    with open(RESULTS_DIR / "holdout.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.enviar:
        submit(y_pred)
    else:
        print("\nEnvio para o servidor NÃO realizado (use --enviar).")


if __name__ == "__main__":
    main()
