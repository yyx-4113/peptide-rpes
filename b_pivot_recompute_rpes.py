"""b_pivot_recompute_rpes.py — Deterministic RPES recomputation (B-pivot).

WHY THIS EXISTS
---------------
The RPES distribution (and hence its reported max) was non-deterministic:
  - train_rf_classifier() sampled its training set with an UNSEEDED random.sample
  - the composite_ml training subset used an UNSEEDED np.random.choice
This produced cache max=0.8148 vs manuscript headline max=0.8386 depending on
the random draw. Both seeds are now fixed (42) in the source files AND explicit
here, so the result is byte-stable and auditable.

WHAT IT DOES
------------
1. Seeds every RNG BEFORE any model is built.
2. Recomputes RPES for all unique peptides with the exact same component stack
   as compute_rpes() (rf / bilstm / composite_ml / ensemble / hydro / length).
3. Backs up the pre-fix cache (rpes_scores_pre_seedfix.npz) for audit trail.
4. Overwrites output/rpes_scores.npz deterministically.
5. Writes a single authoritative stats file (b_pivot_rpes_stats.json).
6. Syncs new_methods_results.json and rpes_full_results.json so cache == headline.

Run with the managed venv that has torch + xgboost (fair-esm NOT required here).
"""
import os, sys, json, time, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- Determinism: seed EVERYTHING before any model is built ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from phase1_pipeline import (
    OUTPUT_DIR, AA_LIST, KD_SCALE, AA_CHARGE,
    load_peptide_library, compute_physicochemical,
    train_rf_classifier, SklearnBiLSTMProxy,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from new_methods import compute_rpes

# ---- Portable path override (config.py) ----
import phase1_pipeline
from config import DATA_DIR, MODEL_DIR as CFG_MODEL_DIR, OUT_DIR as CFG_OUT_DIR
phase1_pipeline.DATA_ROOT = DATA_DIR          # where merged_peptide_library.csv lives
phase1_pipeline.OUTPUT_DIR = CFG_OUT_DIR      # where JSON / rpes_scores.npz are written
phase1_pipeline.MODEL_DIR = CFG_MODEL_DIR     # where pca_latent.npz is read/written
OUTPUT_DIR = CFG_OUT_DIR                       # rebind local name used by this script's writes

t0 = time.time()
log = print

peptides_df = load_peptide_library()
unique_peptides = peptides_df['peptide'].tolist()
n = len(unique_peptides)
log(f"Loaded {n:,} unique peptides")

# RF bioactivity classifier — now deterministic (random.sample seeded above)
rf_model = train_rf_classifier(peptides_df)

# BiLSTM neuro proxy — deterministic (percentile pseudo-labels, no RNG sampling)
bilstm_model = SklearnBiLSTMProxy()
bilstm_model.fit(unique_peptides)

# composite_ml: XGB + SVM + KNN on a FIXED 5000-subset (was the main bug)
np.random.seed(SEED)  # explicit, independent of any other draw
X_phys_full = np.stack([compute_physicochemical(s) for s in unique_peptides])
n_subset = min(5000, n)
subset_idx = np.random.choice(n, n_subset, replace=False)
X_sub = X_phys_full[subset_idx]
y_sub = rf_model.predict_proba(X_sub)[:, 1]
y_bin = (y_sub > 0.5).astype(int)

xgb = XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False,
                    eval_metric='logloss', random_state=SEED)
svm = SVC(kernel='rbf', probability=True, random_state=SEED)
knn = KNeighborsClassifier(n_neighbors=5)
for clf in [xgb, svm, knn]:
    clf.fit(X_sub, y_bin)
ml_classifiers = [xgb, svm, knn]
log(f"composite_ml trained on fixed subset of {n_subset} (seed={SEED})")

# ---- Compute RPES deterministically for the full library ----
rpes, component_ranks = compute_rpes(
    unique_peptides, rf_model, bilstm_model, ml_classifiers, None, None
)

# ---- Back up old cache for audit trail ----
cache = os.path.join(OUTPUT_DIR, 'rpes_scores.npz')
if os.path.exists(cache):
    backup = os.path.join(OUTPUT_DIR, 'rpes_scores_pre_seedfix.npz')
    if not os.path.exists(backup):
        os.replace(cache, backup)
        log(f"Backed up pre-fix cache -> {backup}")

np.savez(cache, rpes=rpes)
log(f"Saved deterministic RPES -> {cache}")

# ---- Authoritative stats ----
stats = {
    'mean': float(rpes.mean()),
    'std': float(rpes.std()),
    'max': float(rpes.max()),
    'p99': float(np.percentile(rpes, 99)),
    'no_ceiling': bool(rpes.max() < 0.9999),
    'seed': SEED,
    'n_peptides': n,
    'computed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}
top20_idx = np.argsort(rpes)[-20:][::-1]
top20 = [{'rank': i + 1, 'peptide': unique_peptides[j], 'rpes': float(rpes[j])}
         for i, j in enumerate(top20_idx)]

result = {
    'method': 'RPES (Rank-Percentile Ensemble Score) — deterministic recompute (B-pivot)',
    'rpes_stats': stats,
    'top20_peptides': top20,
    'component_rank_means': {
        name: float(np.mean(component_ranks[i]))
        for i, name in enumerate(
            ['rf', 'bilstm', 'composite_ml', 'ensemble', 'hydro', 'length'])
    },
}
out = os.path.join(OUTPUT_DIR, 'b_pivot_rpes_stats.json')
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
log(f"Saved authoritative stats -> {out}")

# ---- Sync existing headline JSONs so cache == headline ----
nm_path = os.path.join(OUTPUT_DIR, 'new_methods_results.json')
if os.path.exists(nm_path):
    nm = json.load(open(nm_path))
    nm['rpes_stats'] = stats
    nm['rpes_deterministic'] = True
    nm['rpes_seed'] = SEED
    json.dump(nm, open(nm_path, 'w'), indent=2)
    log(f"Synced new_methods_results.json rpes_stats (max={stats['max']:.4f})")

rf_path = os.path.join(OUTPUT_DIR, 'rpes_full_results.json')
if os.path.exists(rf_path):
    rfj = json.load(open(rf_path))
    rfj['rpes_distribution'] = stats
    json.dump(rfj, open(rf_path, 'w'), indent=2)
    log(f"Synced rpes_full_results.json rpes_distribution")

log(f"TOTAL TIME: {time.time() - t0:.1f}s")
log(f"FINAL RPES max (deterministic, seed={SEED}) = {stats['max']:.4f}")
