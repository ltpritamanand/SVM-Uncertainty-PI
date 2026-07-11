"""Re-generate comparison plots from existing CSV without re-running models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

CSV = Path(__file__).parent / "results" / "sweep_comparison.csv"
OUT = CSV.parent

df = pd.read_csv(CSV)

markers = {"ANN": "o", "QRF": "s", "SVM": "D", "XGB": "^", "GPR": "P", "NGB": "X"}
colors  = {"ANN": "C0", "QRF": "C1", "SVM": "C2", "XGB": "C3", "GPR": "C4", "NGB": "C5"}

# --- sigma_opt + sigma_in comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
titles   = ["sigma opt (init noise)", "sigma input (data noise)"]
columns  = ["sigma_opt", "sigma_in"]
ylabels  = [r"$\sigma^2$ opt", r"$\sigma^2$ in"]
draw_orders = {
    "sigma_opt": ["ANN", "QRF", "XGB", "GPR", "NGB", "SVM"],
    "sigma_in":  ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"],
}

for ax, col, title, ylabel in zip(axes, columns, titles, ylabels):
    for m in draw_orders[col]:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        ax.plot(sub["n_train"], sub[col],
                marker=markers[m], color=colors[m], label=m, lw=2)
    ax.set_xlabel("Total Training Points", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)

fig.tight_layout()
fig.savefig(OUT / "sweep_comparison.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("Saved sweep_comparison.png")

# --- sum_rmse comparison ---
draw_order = ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"]
fig, ax = plt.subplots(figsize=(8, 5))
for m in draw_order:
    sub = df[df["model"] == m]
    if sub.empty:
        continue
    ax.plot(sub["n_train"], sub["sum_rmse"],
            marker=markers[m], color=colors[m], label=m, lw=2)
ax.set_xlabel("Total Training Points", fontsize=14)
ax.set_ylabel("Mean of RMSE", fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "sweep_sumrmse.png", dpi=160)
plt.close(fig)
print("Saved sweep_sumrmse.png")

# --- sigma2_model = sigma2_opt + sigma2_in comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
for m in draw_order:
    sub = df[df["model"] == m]
    if sub.empty:
        continue
    ax.plot(sub["n_train"], sub["sigma2_model"],
            marker=markers[m], color=colors[m], label=m, lw=2)
ax.set_xlabel("Total Training Points", fontsize=14)
ax.set_ylabel(r"$\sigma^2$ model", fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "sweep_sigma2_model.png", dpi=160)
plt.close(fig)
print("Saved sweep_sigma2_model.png")
