# ==============================================================
# logistic_core.py — Regresión logística binaria desde cero
# ==============================================================

import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)          # evita overflow en exp
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    eps = 1e-15                         # evita log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def logistic_regression_train(X, y, lr=0.01, n_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []

    for i in range(n_iter):
        z = X @ w + b
        p = sigmoid(z)

        # gradientes
        dw = (X.T @ (p - y)) / n_samples
        db = np.sum(p - y) / n_samples

        # actualización
        w -= lr * dw
        b -= lr * db

        # pérdida
        loss = compute_loss(y, p)
        losses.append(loss)

        if i % 100 == 0:
            print(f"Iteración {i:4d} | Pérdida: {loss:.4f}")

        # parada temprana si casi no mejora
        if i > 10 and abs(losses[-1] - losses[-2]) < 1e-8:
            break

    return w, b, losses

def predict(X, w, b, threshold=0.5):
    p = sigmoid(X @ w + b)
    return (p >= threshold).astype(int)
