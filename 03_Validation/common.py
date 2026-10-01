"""
Código compartilhado: leitura dos dados, divisão holdout, pré-processamento
e engenharia de atributos.

Todo pré-processamento fica dentro de um Pipeline do scikit-learn para que
o fit aconteça só nos dados de treino de cada fold (sem vazamento).
"""
import os
import warnings
from pathlib import Path

# SVC(probability=True) foi depreciado no sklearn 1.9; o env var propaga o
# filtro para os processos paralelos do joblib.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

SEED = 42
HOLDOUT_SIZE = 0.2
TARGET = "type"
CAT_COLS = ["sex"]
NUM_COLS = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
RATIO_COLS = ["shell_ratio", "shucked_ratio", "viscera_ratio"]
ENG_COLS = RATIO_COLS + ["density", "log_whole_weight", "log_shell_weight"]
FEATURE_SETS = {"originais": [], "razoes": RATIO_COLS, "todas": ENG_COLS}


def load_train():
    df = pd.read_csv(BASE_DIR / "abalone_dataset.csv")
    return df.drop(columns=TARGET), df[TARGET]


def load_app():
    return pd.read_csv(BASE_DIR / "abalone_app.csv")


def dev_holdout_split():
    """Holdout estratificado fixo. Só é usado na avaliação final (final.py)."""
    X, y = load_train()
    return train_test_split(X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=SEED)


def add_features(X):
    X = X.copy()
    eps = 1e-3
    X["shell_ratio"] = X["shell_weight"] / (X["whole_weight"] + eps)
    X["shucked_ratio"] = X["shucked_weight"] / (X["whole_weight"] + eps)
    X["viscera_ratio"] = X["viscera_weight"] / (X["whole_weight"] + eps)
    volume = X["length"] * X["diameter"] * X["height"]
    X["density"] = X["whole_weight"] / (volume + eps)
    X["log_whole_weight"] = np.log1p(X["whole_weight"])
    X["log_shell_weight"] = np.log1p(X["shell_weight"])
    return X


def make_preprocessor(engineered=False, scale=True):
    """engineered: False, True (todas as ENG_COLS) ou nome em FEATURE_SETS."""
    if isinstance(engineered, str):
        eng_cols = FEATURE_SETS[engineered]
    else:
        eng_cols = ENG_COLS if engineered else []
    num_step = StandardScaler() if scale else "passthrough"
    columns = ColumnTransformer(
        [
            ("num", num_step, NUM_COLS + eng_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
    steps = []
    if eng_cols:
        steps.append(("features", FunctionTransformer(add_features)))
    steps.append(("columns", columns))
    return Pipeline(steps)


def make_pipeline(model, engineered=False, scale=True):
    return Pipeline(
        [
            ("prep", make_preprocessor(engineered=engineered, scale=scale)),
            ("model", model),
        ]
    )


def feature_set_grid():
    return [make_preprocessor(name) for name in FEATURE_SETS]


def describe_prep(prep):
    n_num = len(prep.named_steps["columns"].transformers[0][2])
    return {len(NUM_COLS) + len(cols): name for name, cols in FEATURE_SETS.items()}[n_num]


def corrected_paired_ttest(scores_a, scores_b, n_train, n_test):
    """
    Teste t pareado corrigido de Nadeau & Bengio (2003) para comparar dois
    modelos avaliados nos mesmos folds de uma validação cruzada repetida.
    A correção compensa a dependência entre folds (os conjuntos de treino se
    sobrepõem), que faz o teste t comum ser otimista demais.
    """
    from scipy import stats

    diff = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diff.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return t, p
