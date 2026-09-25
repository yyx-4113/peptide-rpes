"""generate_figures.py — regenerate the manuscript figure set from cached artefacts.

Requires the large artefacts (peptide CSV + pca_latent.npz) obtained from the
Zenodo deposit (see README). Outputs PNGs to output/figures/.

    Figure 1  (composite): A search convergence / B three-layer / C ablation
    Figure 2             : representation benchmark (descriptors vs PCA vs ESM-2)
    Figure S1            : PCA scree (fit on the full deduplicated library)
    Figure S2            : latent projection coloured by length and by RPES
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

from config import PEPTIDE_CSV, MODEL_DIR, OUT_DIR

REPO_OUT = OUT_DIR
CSV = PEPTIDE_CSV
LATENT_NPZ = os.path.join(MODEL_DIR, "pca_latent.npz")
RPES_NPZ = os.path.join(OUT_DIR, "rpes_scores.npz")
OUTDIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUTDIR, exist_ok=True)

C_NAT, C_RND, C_SRCH = "#1f77b4", "#d62728", "#2ca02c"
C_DESC, C_PCA, C_ESM = "#1f77b4", "#ff7f0e", "#9467bd"
GRID = "#dddddd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300,
})

with open(os.path.join(REPO_OUT, "rpes_full_results.json")) as f:
    full = json.load(f)
with open(os.path.join(REPO_OUT, "esm2_comparison.json")) as f:
    esm = json.load(f)

tl, ab = full["three_layer_comparison"], full["ablation_study"]
NAT = tl["layer1_natural_best"]; RND = tl["layer2_random_best"]
L3_MEAN = tl["layer3_al_mean"]; L3_STD = tl["layer3_al_std"]
L3_DETAILS = tl["layer3_details"]
FULL = ab["full_method"]; FULL_STD = ab["full_method_std"]
A_MEAN = ab["abl_A_no_al_mean"]; A_STD = ab["abl_A_no_al_std"]
B_MEAN = ab["abl_B_no_cond_mean"]; B_STD = ab["abl_B_no_cond_std"]
C_PRED = ab["abl_C_predicted_best"]
DESC_CV, DESC_STD = esm["phys_only"]["cv_r2"], esm["phys_only"]["cv_std"]
PCA_CV, PCA_STD = esm["pca_phys"]["cv_r2"], esm["pca_phys"]["cv_std"]
ESM_CV, ESM_STD = esm["esm2_phys"]["cv_r2"], esm["esm2_phys"]["cv_std"]

# ----------------------------- Figure 1 -------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
fig.subplots_adjust(wspace=0.45, left=0.06, right=0.98, top=0.88, bottom=0.16)

ax = axes[0]
colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
for i, det_s in enumerate(L3_DETAILS):
    hist = det_s["best_history"]
    ax.plot(range(len(hist)), hist, marker="o", ms=4, lw=1.4, color=colors[i],
            label=f"seed {det_s['seed']}")
ax.axhline(NAT, color=C_NAT, ls="--", lw=1.3, label=f"Layer 1 natural = {NAT:.4f}")
ax.axhline(RND, color=C_RND, ls=":", lw=1.3, label=f"Layer 2 random = {RND:.4f}")
ax.set_xlabel("Search round"); ax.set_ylabel("Best RPES seen")
ax.set_title("A. Search convergence", fontweight="bold")
ax.set_ylim(0.78, 0.85); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, color=GRID, lw=0.6)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

ax = axes[1]
labels = ["Layer 1\nNatural", "Layer 2\nRandom latent", "Layer 3\nSearch"]
vals = [NAT, RND, L3_MEAN]; errs = [0.0, 0.0, L3_STD]
bars = ax.bar(labels, vals, yerr=errs, capsize=4, width=0.62,
              color=[C_NAT, C_RND, C_SRCH], edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Best RPES"); ax.set_title("B. Three-layer comparison", fontweight="bold")
ax.set_ylim(0, 0.95); ax.grid(True, axis="y", color=GRID, lw=0.6)

ax = axes[2]
labels = ["Full\nmethod", "A: no loop", "B: no\nguidance", "C: descriptors\nonly"]
vals = [FULL, A_MEAN, B_MEAN, C_PRED]; errs = [FULL_STD, A_STD, B_STD, 0.0]
bars = ax.bar(labels, vals, yerr=errs, capsize=4, width=0.62,
              color=["#2ca02c", "#8c564b", "#d62728", "#1f77b4"], edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("Best RPES"); ax.set_title("C. Ablation study", fontweight="bold")
ax.set_ylim(0, 0.95); ax.grid(True, axis="y", color=GRID, lw=0.6)

fig.suptitle("Figure 1. Latent-space search evaluation under RPES", fontsize=12,
             fontweight="bold", y=0.98)
fig.savefig(os.path.join(OUTDIR, "Fig1.png"), bbox_inches="tight"); plt.close(fig)
print("wrote Fig1.png")

# ----------------------------- Figure 2 -------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 3.4))
labels = ["Descriptors\n(10)", "PCA latent\n+ descriptors\n(64+10)", "ESM-2\n+ descriptors\n(320+10)"]
vals = [DESC_CV, PCA_CV, ESM_CV]; errs = [DESC_STD, PCA_STD, ESM_STD]
bars = ax.bar(labels, vals, yerr=errs, capsize=5, width=0.6,
              color=[C_DESC, C_PCA, C_ESM], edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("3-fold CV R$^2$ predicting RPES"); ax.set_ylim(0.85, 0.97)
ax.set_title("Figure 2. Representation benchmark\n(5,000-peptide CV)", fontweight="bold")
ax.grid(True, axis="y", color=GRID, lw=0.6)
fig.savefig(os.path.join(OUTDIR, "Fig2.png"), bbox_inches="tight"); plt.close(fig)
print("wrote Fig2.png")

# ----------------------- Figure S1 — PCA scree ------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
MAX_LEN = 20

def build_onehot(seqs):
    n = len(seqs)
    padded = np.array([list(s.ljust(MAX_LEN, "\x00")[:MAX_LEN]) for s in seqs], dtype="U1")
    arr = np.full((n, MAX_LEN), -1, dtype=np.int8)
    for ci, ch in enumerate(AA):
        arr[padded == ch] = ci
    oh = np.zeros((n, MAX_LEN, 20), dtype=np.float32)
    valid = arr >= 0
    idx = np.where(valid, arr, 0)
    flat = oh.reshape(n * MAX_LEN, 20)
    vf = valid.reshape(-1); iff = idx.reshape(-1)
    flat[vf, iff[vf]] = 1.0
    return oh.reshape(n, MAX_LEN * 20)

print("loading deduplicated peptide library ...")
df = pd.read_csv(CSV, usecols=["peptide"]).drop_duplicates(subset=["peptide"]).reset_index(drop=True)
print("  unique peptides:", len(df))
X = build_onehot(df["peptide"].astype(str).values)
print("fitting PCA(64) on full library", X.shape, "...")
pca = PCA(n_components=64, random_state=42)
pca.fit(X)
evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
print(f"  cumulative explained variance at 64 PCs = {cum[-1]:.4f} (manuscript: 0.489)")

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.bar(range(1, 65), evr, color=C_NAT, edgecolor="none", width=0.9, label="Per-PC variance")
ax.plot(range(1, 65), cum, color=C_RND, lw=1.6, label="Cumulative")
ax.axhline(0.489, color="black", ls="--", lw=1.0)
ax.text(64, 0.462, "full-library fit:\n0.489 at 64 PCs", fontsize=7.5, ha="right", va="top")
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
ax.set_title("Figure S1. PCA scree (64-dim latent)", fontweight="bold")
ax.set_xlim(0.5, 64.5); ax.grid(True, axis="y", color=GRID, lw=0.6)
ax.legend(fontsize=8, loc="center right")
axins = ax.inset_axes([0.30, 0.42, 0.38, 0.42])
axins.bar(range(1, 11), evr[:10], color=C_NAT, width=0.8)
axins.plot(range(1, 11), cum[:10], color=C_RND, lw=1.4)
axins.set_title("PCs 1-10", fontsize=7.5); axins.tick_params(labelsize=6.5)
axins.grid(True, axis="y", color=GRID, lw=0.4)
ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)
fig.savefig(os.path.join(OUTDIR, "FigS1.png"), bbox_inches="tight"); plt.close(fig)
print("wrote FigS1.png")

# ------------------- Figure S2 — latent projection --------------------------
print("loading latents + rpes ...")
lat = np.load(LATENT_NPZ)["latents"].astype(np.float32)
rpes = np.load(RPES_NPZ)["rpes"].astype(np.float32)
plens = df["peptide"].astype(str).str.len().values
proj_rng = np.random.default_rng(7)
pidx = proj_rng.choice(len(lat), size=5000, replace=False)
lx, ly = lat[pidx, 0], lat[pidx, 1]
p_len, pr = plens[pidx], rpes[pidx]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
fig.subplots_adjust(wspace=0.42, left=0.07, right=0.98, top=0.84, bottom=0.14)
ax = axes[0]
sc = ax.scatter(lx, ly, c=p_len, cmap="plasma", s=6, alpha=0.6, edgecolors="none")
fig.colorbar(sc, ax=ax).set_label("Peptide length (aa)")
ax.set_xlabel("Latent PC1"); ax.set_ylabel("Latent PC2")
ax.set_title("A. Coloured by length", fontweight="bold", fontsize=10)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.locator_params(axis="both", nbins=5)
ax = axes[1]
sc = ax.scatter(lx, ly, c=pr, cmap="viridis", s=6, alpha=0.6, edgecolors="none")
fig.colorbar(sc, ax=ax).set_label("RPES")
ax.set_xlabel("Latent PC1"); ax.set_ylabel("Latent PC2")
ax.set_title("B. Coloured by RPES", fontweight="bold", fontsize=10)
ax.grid(True, color=GRID, lw=0.5, alpha=0.5); ax.locator_params(axis="both", nbins=5)
fig.suptitle("Figure S2. PCA latent projection (5,000-peptide sample)", fontsize=11,
             fontweight="bold", y=0.97)
fig.savefig(os.path.join(OUTDIR, "FigS2.png"), bbox_inches="tight"); plt.close(fig)
print("wrote FigS2.png")
print("ALL FIGURES DONE ->", OUTDIR)
