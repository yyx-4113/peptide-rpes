"""compute_c1_old_score.py — T1-1: authentic ceiling demonstration.

The ceiling effect lives in the OLD weighted-average composite (each component a
bounded [0,1] raw score, then averaged) that RPES replaces. We recompute that
old score over the full library (raw weighted sum, no final norm01, plus the
codebase's own norm01 version for authenticity) and contrast its saturation with
RPES's geometric mean. Saves c1_ceiling.json.
"""
import os, sys, json, time, random
import numpy as np
sys.path.insert(0, r'D:\projects\peptide-rpes')
from phase1_pipeline import (OUTPUT_DIR, MODEL_DIR, AA_LIST, KD_SCALE, AA_CHARGE, log)
import phase1_pipeline
from config import DATA_DIR, MODEL_DIR as CFG_MODEL_DIR, OUT_DIR as CFG_OUT_DIR
phase1_pipeline.DATA_ROOT = DATA_DIR
phase1_pipeline.OUTPUT_DIR = CFG_OUT_DIR
phase1_pipeline.MODEL_DIR = CFG_MODEL_DIR
OUTPUT_DIR = CFG_OUT_DIR
MODEL_DIR = CFG_MODEL_DIR
from phase1_pipeline import (train_rf_classifier, SklearnBiLSTMProxy,
    compute_rescue_scores, compute_physicochemical, load_peptide_library)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from new_methods import compute_rpes

t0 = time.time()
log("C1-old: training models")
peptides_df = load_peptide_library()
unique_peptides = peptides_df['peptide'].tolist()
n = len(unique_peptides)
random.seed(42); np.random.seed(42)
rf_model = train_rf_classifier(peptides_df)
bilstm_model = SklearnBiLSTMProxy(); bilstm_model.fit(unique_peptides)
X_phys_full = np.stack([compute_physicochemical(s) for s in unique_peptides])
n_subset = min(5000, n); np.random.seed(42)
subset_idx = np.random.choice(n, n_subset, replace=False)
X_sub = X_phys_full[subset_idx]; y_sub = rf_model.predict_proba(X_sub)[:, 1]; y_bin = (y_sub > 0.5).astype(int)
xgb = XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False, eval_metric='logloss', random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
for clf in [xgb, svm, knn]: clf.fit(X_sub, y_bin)
ml_classifiers = [xgb, svm, knn]

log("C1-old: computing RAW component scores over full library")
rf_raw = rf_model.predict_proba(X_phys_full)[:, 1]
bilstm_raw = bilstm_model.predict_proba_batched(unique_peptides, batch_size=5000)
ml_preds = np.mean([clf.predict_proba(X_phys_full)[:, 1] for clf in ml_classifiers], axis=0)
hydros = np.array([np.mean([KD_SCALE.get(aa, 0) for aa in s]) for s in unique_peptides])
hq1, hq3 = np.percentile(hydros, [25, 75])
hydro_raw = np.exp(-((hydros - (hq1 + hq3) / 2) ** 2) / (2 * max((hq3 - hq1) / 2, 0.1) ** 2))
lens = np.array([len(s) for s in unique_peptides])
length_raw = np.exp(-((lens - 9.9) ** 2) / (2 * 4.7 ** 2))
# OLD weighted-average composite (raw [0,1] scores), paper weights [0.30,0.20,0.15,0.15,0.10,0.10]
# ensemble term held at 0 (neutral) -> weight 0.15 contributes 0
w = np.array([0.30, 0.20, 0.15, 0.0, 0.10, 0.10])  # drop ensemble slot's weight effect
old_weighted_raw = (w[0]*rf_raw + w[1]*bilstm_raw + w[2]*ml_preds + w[4]*hydro_raw + w[5]*length_raw) / w.sum()
# Authentic codebase rescue score (with norm01) for reference
old_rescue = compute_rescue_scores(unique_peptides, rf_model, bilstm_model, ml_classifiers, None, None)

# RPES geometric mean (already computed)
rpes = np.load(os.path.join(OUTPUT_DIR, 'rpes_scores.npz'))['rpes']

def ceiling_metrics(x, label):
    x = np.asarray(x, dtype=np.float64)
    return {'label': label, 'max': float(x.max()), 'mean': float(x.mean()),
            'p99': float(np.percentile(x, 99)), 'p95': float(np.percentile(x, 95)),
            'frac_above_0.85': float((x > 0.85).mean()),
            'frac_within_0.05_of_max': float((x >= x.max() - 0.05).mean())}

results = {
    'n_peptides': n,
    'rpes_geometric_mean': ceiling_metrics(rpes, 'RPES (geometric mean of ranks)'),
    'old_weighted_raw': ceiling_metrics(old_weighted_raw, 'OLD weighted-average of raw [0,1] components (paper weights)'),
    'old_rescue_norm01': ceiling_metrics(old_rescue, 'OLD rescue score (codebase, norm01)'),
    'elapsed_sec': time.time() - t0,
}
with open(os.path.join(OUTPUT_DIR, 'c1_ceiling.json'), 'w') as f:
    json.dump(results, f, indent=2)
log(f"C1 done in {results['elapsed_sec']:.1f}s")
log(f"  RPES max={results['rpes_geometric_mean']['max']:.4f} frac>0.85={results['rpes_geometric_mean']['frac_above_0.85']:.4f}")
log(f"  OLD weighted-raw max={results['old_weighted_raw']['max']:.4f} frac>0.85={results['old_weighted_raw']['frac_above_0.85']:.4f}")
log(f"  OLD rescue(norm01) max={results['old_rescue_norm01']['max']:.4f} frac>0.95={results['old_rescue_norm01']['frac_within_0.05_of_max']:.4f}")
print(json.dumps(results, indent=2))
