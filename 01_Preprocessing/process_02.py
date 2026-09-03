"""
Script de processamento do dataset de diabetes - Teste 2.

Critérios aplicados:

1. Seleção:
   - Remover todas as linhas com 3 ou mais itens (colunas) faltantes.

2. Processamento dos dados (para as linhas restantes, com 0, 1 ou 2 itens faltantes):
   - Se faltar exatamente 1 coluna: preencher com um valor aleatório
     entre [min_coluna, max_coluna].
   - Se faltarem exatamente 2 colunas:
       - Se uma delas for Insulina: excluir a linha.
       - Caso contrário (nenhuma das 2 for Insulina): preencher ambas
         com um valor aleatório entre [min_coluna, max_coluna] de cada
         coluna respectivamente.

Saída: diabetes_dataset_knn_02.csv
"""

import pandas as pd
import numpy as np

# Reprodutibilidade do sorteio aleatório
np.random.seed(42)

INPUT_FILE = "diabetes_dataset.csv"
OUTPUT_FILE = "diabetes_dataset_knn_02.csv"

# Colunas que podem conter dados faltantes (todas exceto Outcome, que é o rótulo)
FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def preencher_aleatorio(df, col_min, col_max, idx, colunas_faltantes):
    """Preenche as colunas faltantes de uma linha com valor aleatório uniforme
    entre o min e o max de cada coluna."""
    for col in colunas_faltantes:
        valor_aleatorio = np.random.uniform(col_min[col], col_max[col])
        df.loc[idx, col] = valor_aleatorio


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
    # 2. Processamento dos dados
    # ---------------------------------------------------------------
    remaining_missing = df[FEATURE_COLS].isna().sum(axis=1)

    idx_um_faltante = df.index[remaining_missing == 1]
    idx_dois_faltantes = df.index[remaining_missing == 2]

    print(f"Linhas com exatamente 1 valor faltante: {len(idx_um_faltante)}")
    print(f"Linhas com exatamente 2 valores faltantes: {len(idx_dois_faltantes)}")

    # 2.1 Exatamente 1 coluna faltante -> preencher com valor aleatório
    for idx in idx_um_faltante:
        linha = df.loc[idx, FEATURE_COLS]
        colunas_faltantes = linha[linha.isna()].index.tolist()
        preencher_aleatorio(df, col_min, col_max, idx, colunas_faltantes)

    # 2.2 Exatamente 2 colunas faltantes
    linhas_para_excluir = []
    for idx in idx_dois_faltantes:
        linha = df.loc[idx, FEATURE_COLS]
        colunas_faltantes = linha[linha.isna()].index.tolist()

        if "Insulin" in colunas_faltantes:
            # Uma das faltantes é Insulina -> excluir a linha
            linhas_para_excluir.append(idx)
        else:
            # Nenhuma das faltantes é Insulina -> preencher ambas
            preencher_aleatorio(df, col_min, col_max, idx, colunas_faltantes)

    df = df.drop(index=linhas_para_excluir)
    print(f"Linhas excluídas (2 faltantes incluindo Insulina): {len(linhas_para_excluir)}")

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
