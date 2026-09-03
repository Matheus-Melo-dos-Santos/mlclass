"""
Script de processamento do dataset de diabetes.

Critérios aplicados:
1. Seleção:
   - Remover todas as linhas com 3 ou mais itens (colunas) faltantes.
2. Processamento dos dados:
   - Excluir linhas que não contêm valor de Insulina (mesmo após o filtro acima).
   - Para as linhas restantes, se sobrar exatamente 1 coluna faltante,
     preencher com um valor aleatório entre [min_coluna, max_coluna]
     (min/max calculados a partir dos valores válidos daquela coluna).

Saída: diabetes_dataset_knn.csv
"""

import pandas as pd
import numpy as np

# Reprodutibilidade do sorteio aleatório
np.random.seed(42)

INPUT_FILE = "diabetes_dataset.csv"
OUTPUT_FILE = "diabetes_dataset_knn_01.csv"

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

    # ---------------------------------------------------------------
    # 2. Processamento dos dados
    # ---------------------------------------------------------------

    # 2.1 Excluir linhas sem valor de Insulina
    df = df[df["Insulin"].notna()].copy()
    print(f"Linhas após excluir linhas sem Insulina: {len(df)}")

    # 2.2 Para as linhas restantes, se sobrar exatamente 1 coluna faltante,
    #     preencher com valor aleatório entre [min, max] da própria coluna
    remaining_missing = df[FEATURE_COLS].isna().sum(axis=1)
    rows_to_fill = df.index[remaining_missing == 1]

    print(f"Linhas com exatamente 1 valor faltante a preencher: {len(rows_to_fill)}")

    # Min/max de cada coluna (calculados sobre os valores válidos já existentes)
    col_min = df[FEATURE_COLS].min()
    col_max = df[FEATURE_COLS].max()

    for idx in rows_to_fill:
        # Identifica qual coluna está faltando nessa linha
        col_faltante = df.loc[idx, FEATURE_COLS][df.loc[idx, FEATURE_COLS].isna()].index[0]
        valor_aleatorio = np.random.uniform(col_min[col_faltante], col_max[col_faltante])
        df.loc[idx, col_faltante] = valor_aleatorio

    # Confirma que não restam valores faltantes nas colunas de features
    print("Valores faltantes restantes por coluna:")
    print(df[FEATURE_COLS].isna().sum())

    # ---------------------------------------------------------------
    # Salvando o arquivo de saída
    # ---------------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset final salvo em: {OUTPUT_FILE}")
    print(f"Total de linhas final: {len(df)}")


if __name__ == "__main__":
    main()
