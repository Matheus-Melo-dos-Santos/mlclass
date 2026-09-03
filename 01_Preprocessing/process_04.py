"""
Pré-processamento do dataset de diabetes para o k-NN.

O k-NN não aceita NaN. Insulina falta em ~65% das linhas: excluir essas
linhas deixa ~198 exemplos e a acurácia no servidor cai para ~0,56.

Melhor resultado no servidor: 0,622.

Estratégia:
1. Remover linhas com 3 ou mais atributos faltantes.
2. Imputar os demais faltantes com KNNImputer (n=5) — não sortear [min, max]
   e não descartar quem não tem insulina.
3. Padronizar treino e teste com o mesmo StandardScaler. Sem isso, atributos
   de escala grande (Insulin, Glucose) dominam a distância euclidiana.

Saídas:
  - diabetes_dataset_knn.csv
  - diabetes_app_knn.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

INPUT_TRAIN = BASE_DIR / "diabetes_dataset.csv"
INPUT_APP = BASE_DIR / "diabetes_app.csv"
OUTPUT_TRAIN = BASE_DIR / "diabetes_dataset_knn.csv"
OUTPUT_APP = BASE_DIR / "diabetes_app_knn.csv"

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def main():
    train = pd.read_csv(INPUT_TRAIN)
    app = pd.read_csv(INPUT_APP)
    print(f"Linhas no dataset original: {len(train)}")

    missing_count = train[FEATURE_COLS].isna().sum(axis=1)
    train = train[missing_count < 3].copy()
    print(f"Linhas após remover linhas com >= 3 valores faltantes: {len(train)}")
    print("Valores faltantes a imputar por coluna:")
    print(train[FEATURE_COLS].isna().sum())

    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train[FEATURE_COLS])
    X_train = scaler.fit_transform(X_train)
    X_app = scaler.transform(app[FEATURE_COLS].to_numpy())

    train_out = pd.DataFrame(X_train, columns=FEATURE_COLS, index=train.index)
    train_out["Outcome"] = train["Outcome"].values
    app_out = pd.DataFrame(X_app, columns=FEATURE_COLS)

    train_out.to_csv(OUTPUT_TRAIN, index=False)
    app_out.to_csv(OUTPUT_APP, index=False)

    print(f"\nTreino salvo em: {OUTPUT_TRAIN}")
    print(f"Teste padronizado salvo em: {OUTPUT_APP}")
    print(f"Total de linhas de treino: {len(train_out)}")
    print("Valores faltantes restantes (treino):")
    print(train_out.isna().sum())


if __name__ == "__main__":
    main()
