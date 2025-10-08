# ==============================================================
# preprocessing.py — UCI Heart (id=45) 
# ==============================================================

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_heart_data(test_size=0.3, random_state=42):
    # 1) Cargar dataset
    heart = fetch_ucirepo(id=45)
    X_raw = heart.data.features.copy()
    y_raw = heart.data.targets.iloc[:, 0].copy()

    # 2) Recodificar Y: 0 -> 0, resto -> 1
    y_raw = y_raw.apply(lambda v: 0 if v == 0 else 1).astype(int)

    # 3) Eliminar filas con NaN (no se pide imputar en el enunciado)
    df = pd.concat([X_raw, y_raw.rename("target")], axis=1).dropna()
    X = df.drop(columns="target")
    y = df["target"].astype(int)

    # 4) Definir columnas categóricas y numéricas
    #    Estas son categóricas de facto en este dataset
    cat_candidates = ["cp", "restecg", "slope", "thal", "sex", "fbs", "exang"]
    cat_cols = [c for c in cat_candidates if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 5) Split 70/30 (evitar data leakage: ajustar transformadores solo con train)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6) One-Hot en categóricas (fit en train, transform en test)
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        X_train_cat = ohe.fit_transform(X_train_df[cat_cols])
        X_test_cat  = ohe.transform(X_test_df[cat_cols])
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        X_train_cat = np.empty((len(X_train_df), 0))
        X_test_cat  = np.empty((len(X_test_df), 0))
        cat_feature_names = []

    # 7) Estandarizar numéricas (fit en train, transform en test)
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_df[num_cols])
        X_test_num  = scaler.transform(X_test_df[num_cols])
        num_feature_names = [f"std_{c}" for c in num_cols]
    else:
        X_train_num = np.empty((len(X_train_df), 0))
        X_test_num  = np.empty((len(X_test_df), 0))
        num_feature_names = []

    # 8) Concatenar
    X_train = np.hstack([X_train_num, X_train_cat])
    X_test  = np.hstack([X_test_num,  X_test_cat])
    feature_names = num_feature_names + cat_feature_names

    # 9) Sanity check
    assert not np.isnan(X_train).any(), "Hay NaN en X_train"
    assert not np.isnan(X_test).any(),  "Hay NaN en X_test"

    print("✅ UCI id=45 preprocesado (sin imputar, filas con NaN eliminadas).")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), feature_names

# ==============================================================
# load_wine_data — dataset Wine (3 clases, 13 features)
#  - Estandarización de TODAS las características
#  - Split 70/30 con estratificación
# ==============================================================

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_wine_data(test_size=0.3, random_state=42):
    data = load_wine(as_frame=True)
    X = data.data.copy()      # (178, 13)
    y = data.target.copy()    # valores {0,1,2}

    # Estandarizar TODAS las features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 70/30 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    feature_names = X.columns.tolist()
    class_names = data.target_names.tolist()

    print("✅ Wine cargado y estandarizado")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, class_names
