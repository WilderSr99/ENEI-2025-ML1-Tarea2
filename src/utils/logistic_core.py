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

# ==============================================================
# Extensión: Softmax (Regresión logística multinomial)
# ==============================================================

def softmax_stable(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

def nll_softmax(W, Xb, y):
    n = Xb.shape[0]
    logits = Xb @ W
    P = softmax_stable(logits)
    Y = one_hot(y, W.shape[1])
    eps = 1e-12
    ll = np.sum(Y * np.log(P + eps))
    return -ll / n

def grad_softmax(W, Xb, y):
    n = Xb.shape[0]
    logits = Xb @ W
    P = softmax_stable(logits)
    Y = one_hot(y, W.shape[1])
    grad = (Xb.T @ (P - Y)) / n
    return grad

def fit_softmax_gd(X, y, lr=0.1, n_iter=3000, tol=1e-6, verbose=False):
    n, d = X.shape
    K = int(np.max(y)) + 1
    Xb = np.c_[np.ones((n, 1)), X]
    W = np.random.randn(d + 1, K) * 0.01
    losses = []
    prev_loss = None

    for t in range(n_iter):
        loss = nll_softmax(W, Xb, y)
        grad = grad_softmax(W, Xb, y)
        W -= lr * grad
        losses.append(loss)

        if verbose and t % 100 == 0:
            print(f"Iter {t:4d} | NLL={loss:.6f}")
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    return W, np.array(losses)

def predict_proba_softmax(X, W):
    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    return softmax_stable(Xb @ W)

def predict_softmax(X, W):
    return np.argmax(predict_proba_softmax(X, W), axis=1)