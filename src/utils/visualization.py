# ==============================================================
# visualization.py — Gráficos de convergencia
# ==============================================================

import matplotlib.pyplot as plt
from pathlib import Path

def plot_convergence(losses, lr, save_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(losses, label=f"lr={lr}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Pérdida (Log-Loss)")
    plt.title(f"Convergencia del gradiente (lr={lr})")
    plt.grid(True)
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
