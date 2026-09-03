"""
Script de processamento do dataset de diabetes - Teste 3.

Critérios aplicados:

1. Seleção:
   - Remover todas as linhas com 3 ou mais itens (colunas) faltantes.

2. Processamento dos dados:
   - Para as linhas restantes (com até 2 itens faltantes), preencher
     cada valor faltante com um valor aleatório entre [min_coluna, max_coluna].

Saída: diabetes_dataset_knn_03.csv
"""

import pandas as pd
import numpy as np

# Reprodutibilidade do sorteio aleatório
np.random.seed(42)

INPUT_FILE = "diabetes_dataset.csv"
OUTPUT_FILE = "diabetes_dataset_knn_03.csv"

# Colunas que podem conter dados faltantes (todas exceto Outcome, que é o rótulo)
FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def main():
    # Leitura do dataset
    df = pd.read_csv(INPUT_FILE)
    print(f"Linhas no dataset original: {len(df)}")

    # ---------------------------------------------------------------
    # 1. Seleção: remover linhas com 3 ou mais itens faltantes
    # ---------------------------------------------------------------
    missing_count = df[FEATURE_COLS].isna().sum(axis=1)
    df = df[missing_count < 3].copy()
    print(f"Linhas após remover linhas com >= 3 valores faltantes: {len(df)}")

    # Min/max de cada coluna, calculados sobre os valores válidos existentes
    # ANTES de qualquer preenchimento (evita que valores sintéticos
    # influenciem os limites usados para gerar outros valores aleatórios)
    col_min = df[FEATURE_COLS].min()
    col_max = df[FEATURE_COLS].max()

    # ---------------------------------------------------------------
    # 2. Processamento dos dados: preencher todos os valores faltantes
    #    restantes com um valor aleatório entre [min_coluna, max_coluna]
    # ---------------------------------------------------------------
    for col in FEATURE_COLS:
        mask = df[col].isna()
        n_faltantes = mask.sum()
        if n_faltantes > 0:
            valores_aleatorios = np.random.uniform(col_min[col], col_max[col], size=n_faltantes)
            df.loc[mask, col] = valores_aleatorios

    # Confirma que não restam valores faltantes nas colunas de features
    print("\nValores faltantes restantes por coluna:")
    print(df[FEATURE_COLS].isna().sum())

    # ---------------------------------------------------------------
    # Salvando o arquivo de saída
    # ---------------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset final salvo em: {OUTPUT_FILE}")
    print(f"Total de linhas final: {len(df)}")


if __name__ == "__main__":
    main()