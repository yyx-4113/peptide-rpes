"""compute_c1_components.py — T1-1: empirically demonstrate the ceiling effect.

Recomputes the six RPES component percentile-rank arrays over the full library
and contrasts RPES (geometric mean) against the ceiling-prone weighted-average
aggregation it replaces. Saves component_ranks.npz and c1_ceiling.json.
"""
import os, sys, json, time, random
import numpy as np
sys.path.insert(0, r'D:\projects\peptide-rpes')
from phase1_pipeline import (OUTPUT_DIR, MODEL_DIR, SEEDS, AA_LIST, AA_TO_IDX,
    MAX_PEPTIDE_LEN, N_AMINO_ACIDS, VAE_LATENT_DIM, AL_ROUNDS, AL_CANDIDATES,
    PCALatentSpace, load_peptide_library, flatten_onehot, compute_physicochemical,
    KD_SCALE, AA_CHARGE, log)
import phase1_pipeline
from config import DATA_DIR, MODEL_DIR as CFG_MODEL_DIR, OUT_DIR as CFG_OUT_DIR
phase1_pipeline.DATA_ROOT = DATA_DIR
phase1_pipeline.OUTPUT_DIR = CFG_OUT_DIR
phase1_pipeline.MODEL_DIR = CFG_MODEL_DIR
OUTPUT_DIR = CFG_OUT_DIR
MODEL_DIR = CFG_MODEL_DIR
from phase1_pipeline import train_rf_classifier, SklearnBiLSTMProxy
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from new_methods import compute_rpes

t0 = time.time()
log("C1: loading library")
peptides_df = load_peptide_library()
unique_peptides = peptides_df['peptide'].tolist()
n = len(unique_peptides)

random.seed(42); np.random.seed(42)
rf_model = train_rf_classifier(peptides_df)
bilstm_model = SklearnBiLSTMProxy(); bilstm_model.fit(unique_peptides)
X_phys_full = np.stack([compute_physicochemical(s) for s in unique_peptides])
n_subset = min(5000, n); np.random.seed(42)
subset_idx = np.random.choice(n, n_subset, replace=False)
X_sub = X_phys_full[subset_idx]
y_sub = rf_model.predict_proba(X_sub)[:, 1]; y_bin = (y_sub > 0.5).astype(int)
xgb = XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False,
                    eval_metric='logloss', random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
for clf in [xgb, svm, knn]:
    clf.fit(X_sub, y_bin)
ml_classifiers = [xgb, svm, knn]

log("C1: computing RPES + component ranks over full library (this is the slow step)")
rpes, component_ranks = compute_rpes(unique_peptides, rf_model, bilstm_model,
                                     ml_classifiers, None, None)
ranks = [np.asarray(c, dtype=np.float64) for c in component_ranks]
names = ['rf', 'bilstm', 'composite', 'ensemble', 'hydro', 'length']
np.savez(os.path.join(OUTPUT_DIR, 'component_ranks.npz'),
         **{nm: rk for nm, rk in zip(names, ranks)})

# ---- Ceiling demonstration ----
# (a) RPES geometric mean (already have rpes)
# (b) equal-weight arithmetic mean of the six percentile ranks (the naive composite RPES replaces)
arith_equal = np.mean(np.vstack(ranks), axis=0)
# (c) original paper weights [0.30,0.20,0.15,0.15,0.10,0.10] applied to the six ranks
w = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
weighted_rank = (w * np.vstack(ranks)).sum(axis=0)  # ensemble rank ~0.5 -> constant 0.075

def ceiling_metrics(x, label):
    x = np.asarray(x, dtype=np.float64)
    return {
        'label': label,
        'max': float(x.max()),
        'mean': float(x.mean()),
        'p99': float(np.percentile(x, 99)),
        'p95': float(np.percentile(x, 95)),
        'frac_above_0.95': float((x > 0.95).mean()),
        'frac_within_0.05_of_max': float((x >= x.max() - 0.05).mean()),
        'frac_in_top_5pct_band': float(((x >= 0.95) & (x <= 1.0)).mean()),
    }

results = {
    'n_peptides': n,
    'rpes_geometric_mean': ceiling_metrics(rpes, 'RPES (geometric mean of ranks)'),
    'arith_mean_equal': ceiling_metrics(arith_equal, 'Arithmetic mean of ranks (equal weight)'),
    'weighted_mean_ranks': ceiling_metrics(weighted_rank, 'Weighted mean of ranks (paper weights)'),
    'component_rank_means': {nm: float(rk.mean()) for nm, rk in zip(names, ranks)},
    'component_rank_correlations_with_rpes': {
        nm: float(np.corrcoef(rk, rpes)[0, 1]) for nm, rk in zip(names, ranks)},
    'elapsed_sec': time.time() - t0,
}
with open(os.path.join(OUTPUT_DIR, 'c1_ceiling.json'), 'w') as f:
    json.dump(results, f, indent=2)
log(f"C1 done in {results['elapsed_sec']:.1f}s. RPES max={results['rpes_geometric_mean']['max']:.4f}, "
    f"arith-mean max={results['arith_mean_equal']['max']:.4f} "
    f"(frac>0.95={results['arith_mean_equal']['frac_above_0.95']:.4f}), "
    f"weighted max={results['weighted_mean_ranks']['max']:.4f} "
    f"(frac>0.95={results['weighted_mean_ranks']['frac_above_0.95']:.4f})")
