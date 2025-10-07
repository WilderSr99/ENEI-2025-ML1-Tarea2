# src/utils/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_heart_data(test_size=0.3, random_state=42):
    """
    Carga y prepara el dataset de enfermedades cardíacas del repositorio UCI.
    Estandariza las características y divide los datos en entrenamiento y prueba.
    """

    try:
        # Intentar cargar desde OpenML (data_id=53)
        heart = fetch_openml(data_id=53, as_frame=True)
        df = heart.frame.copy()
        print("Dataset cargado desde OpenML correctamente.")
    except Exception as e:
        # Si falla, usar respaldo desde GitHub
        print("No se pudo cargar desde OpenML, usando fuente alternativa...")
        url = "https://raw.githubusercontent.com/selva86/datasets/master/Heart.csv"
        df = pd.read_csv(url)
        print("Dataset cargado desde GitHub (backup).")

    # Limpieza de nombres
    df.columns = [col.replace('#', '').strip() for col in df.columns]

    # Separar variables
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), X.columns.tolist()
