"""b_pivot_matched_init_ablation.py
Matched-initialisation re-run of Ablation A, plus a scale-comparability correction.

WHY THIS SCRIPT EXISTS
----------------------
Two defects in the original experiment (rpes_full_experiment.py) make the
"iterative loop contributes zero" claim unsafe:

  D1. Ablation A ("no loop") does NOT hold the starting pool constant.
      Layer 3 initialises from the top half of the library by RPES;
      Ablation A draws latents from N(0, 0.3) in latent space. The observed
      gap (0.8180 vs 0.8000) therefore confounds loop, initialisation and
      candidate budget.

  D2. Candidate scores are computed on a DIFFERENT scale from pool scores.
      compute_rpes() converts each component to a percentile rank WITHIN THE
      PASSED BATCH (rankdata(x)/len(x)). Layer 3 calls it on ~240 decoded
      candidates, so those scores are batch-relative, while the initial pool
      carries library-relative RPES (n = 403,461). The two are not comparable,
      so "improvement = 0.0000" is partly an artefact of the mismatch.

WHAT THIS SCRIPT DOES
---------------------
  1. Recomputes the six RPES component RAW values over the full library and
     verifies the recomputed RPES reproduces the cached array (max 0.8342).
  2. Builds library-relative percentile maps for every component, so decoded
     candidates can be scored on the SAME scale as library peptides.
  3. Re-runs, for seeds 42 / 123 / 456:
       L3_orig  Layer 3 exactly as originally implemented (batch-relative)
                -> expected to reproduce 0.8180 +/- 0.0070, improvement 0.0000
       L3_corr  Layer 3 with library-relative candidate scoring
       M1       matched initialisation, single pass, 240 candidates
                (= Layer 3 round 1 only; isolates the loop, init held constant)
       M2       matched initialisation, single pass, 2400 candidates
                (= same total candidate budget as the 10-round loop, no loop)
       A_corr   original Ablation A procedure (N(0,0.3) latents) but scored
                library-relatively, so it is comparable to the others
  4. Writes output/ablation_matched_init.json.

All RNG calls replicate rpes_full_experiment.py exactly, so L3_orig must
reproduce the published numbers; that is the script's self-check.
"""

import os, sys, json, time, random
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_pipeline import (
    OUTPUT_DIR, MODEL_DIR, AA_LIST, AA_TO_IDX, MAX_PEPTIDE_LEN,
    N_AMINO_ACIDS, VAE_LATENT_DIM,
    PCALatentSpace, load_peptide_library, flatten_onehot,
    compute_physicochemical, KD_SCALE, AA_CHARGE, log,
)

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

SEEDS = [42, 123, 456]
N_ROUNDS = 10
N_CAND_PER_ROUND = 250
TOP_PARENTS = 30

t_start = time.time()
log("=" * 70)
log("MATCHED-INITIALISATION ABLATION + SCALE-CORRECTION RE-RUN")
log("=" * 70)

# ------------------------------------------------------------------
# 1. Load library, models, PCA  (identical to rpes_full_experiment.py)
# ------------------------------------------------------------------
log("Loading library and models...")
peptides_df = load_peptide_library()
unique_peptides = peptides_df['peptide'].tolist()
n_lib = len(unique_peptides)
log(f"  library size = {n_lib:,}")

pca_data = np.load(os.path.join(MODEL_DIR, 'pca_latent.npz'), allow_pickle=True)
vae_latents = pca_data['latents']

random.seed(42)
np.random.seed(42)
rf_model = train_rf_classifier(peptides_df)
bilstm_model = SklearnBiLSTMProxy()
bilstm_model.fit(unique_peptides)

X_phys_full = np.stack([compute_physicochemical(s) for s in unique_peptides])
np.random.seed(42)
subset_idx = np.random.choice(n_lib, 5000, replace=False)
X_sub = X_phys_full[subset_idx]
y_bin = (rf_model.predict_proba(X_sub)[:, 1] > 0.5).astype(int)

xgb = XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False,
                    eval_metric='logloss', random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
for clf in [xgb, svm, knn]:
    clf.fit(X_sub, y_bin)
ml_classifiers = [xgb, svm, knn]

rpes_cache = os.path.join(OUTPUT_DIR, 'rpes_scores.npz')
rpes_cached = np.load(rpes_cache)['rpes']
log(f"  cached RPES loaded: max={rpes_cached.max():.4f}, mean={rpes_cached.mean():.4f}")

vae_peptides = random.sample(unique_peptides, 52517)
pca = PCALatentSpace()
pca.fit(flatten_onehot(vae_peptides))
log(f"  PCA decoder ready ({time.time()-t_start:.1f}s)")

# ------------------------------------------------------------------
# 2. Library raw components + verification
# ------------------------------------------------------------------
log("Computing library raw component values...")
t0 = time.time()

rf_raw_lib = rf_model.predict_proba(X_phys_full)[:, 1]
log(f"  rf_raw done ({time.time()-t0:.1f}s)")

bilstm_raw_lib = bilstm_model.predict_proba_batched(unique_peptides, batch_size=5000)
log(f"  bilstm_raw done ({time.time()-t0:.1f}s)")

ml_preds = []
for clf in ml_classifiers:
    try:
        ml_preds.append(clf.predict_proba(X_phys_full)[:, 1])
    except Exception:
        ml_preds.append(np.zeros(n_lib))
composite_raw_lib = np.mean(ml_preds, axis=0)
log(f"  composite_raw done ({time.time()-t0:.1f}s)")

hydros_lib = np.array([np.mean([KD_SCALE.get(aa, 0) for aa in s]) for s in unique_peptides])
hydro_q1, hydro_q3 = np.percentile(hydros_lib, [25, 75])
hydro_center = (hydro_q1 + hydro_q3) / 2
hydro_sigma = max((hydro_q3 - hydro_q1) / 2, 0.1)
hydro_raw_lib = np.exp(-((hydros_lib - hydro_center) ** 2) / (2 * hydro_sigma ** 2))

lens_lib = np.array([len(s) for s in unique_peptides])
length_raw_lib = np.exp(-((lens_lib - 9.9) ** 2) / (2 * 4.7 ** 2))

ens_raw_lib = np.zeros(n_lib)   # surrogate term held neutral (no external surrogate)

def to_rank(x):
    return (rankdata(x, method='average') / len(x)).astype(np.float64)

lib_ranks = [to_rank(rf_raw_lib), to_rank(bilstm_raw_lib), to_rank(composite_raw_lib),
             to_rank(ens_raw_lib), to_rank(hydro_raw_lib), to_rank(length_raw_lib)]
rpes_lib_recomputed = np.exp(np.mean(np.log(np.vstack(lib_ranks)), axis=0))

# --- verification against the cached array ---
verify = {
    'n_library': int(n_lib),
    'cached_max': float(rpes_cached.max()),
    'recomputed_max': float(rpes_lib_recomputed.max()),
    'cached_mean': float(rpes_cached.mean()),
    'recomputed_mean': float(rpes_lib_recomputed.mean()),
    'pearson_r_vs_cached': float(np.corrcoef(rpes_cached, rpes_lib_recomputed)[0, 1]),
    'mean_abs_diff': float(np.mean(np.abs(rpes_cached - rpes_lib_recomputed))),
    'argmax_peptide': unique_peptides[int(np.argmax(rpes_lib_recomputed))],
}
log("VERIFICATION (recomputed vs cached RPES):")
for k, v in verify.items():
    log(f"  {k}: {v}")

# ------------------------------------------------------------------
# 3. Library-relative scoring for arbitrary candidate sequences
# ------------------------------------------------------------------
SORTED = [np.sort(a) for a in
          (rf_raw_lib, bilstm_raw_lib, composite_raw_lib,
           ens_raw_lib, hydro_raw_lib, length_raw_lib)]

def _pct(sorted_arr, v):
    left = np.searchsorted(sorted_arr, v, side='left')
    right = np.searchsorted(sorted_arr, v, side='right')
    return (left + right + 1) / (2.0 * len(sorted_arr))

def score_library_relative(seqs):
    """RPES of candidate sequences on the LIBRARY percentile scale."""
    if len(seqs) == 0:
        return np.array([])
    Xp = np.stack([compute_physicochemical(s) for s in seqs])
    rf_r = rf_model.predict_proba(Xp)[:, 1]
    bi_r = bilstm_model.predict_proba_batched(seqs, batch_size=5000)
    mlp = []
    for clf in ml_classifiers:
        try:
            mlp.append(clf.predict_proba(Xp)[:, 1])
        except Exception:
            mlp.append(np.zeros(len(seqs)))
    comp_r = np.mean(mlp, axis=0)
    ens_r = np.zeros(len(seqs))
    hy = np.array([np.mean([KD_SCALE.get(aa, 0) for aa in s]) for s in seqs])
    hy_r = np.exp(-((hy - hydro_center) ** 2) / (2 * hydro_sigma ** 2))
    ln = np.array([len(s) for s in seqs])
    ln_r = np.exp(-((ln - 9.9) ** 2) / (2 * 4.7 ** 2))
    raws = [rf_r, bi_r, comp_r, ens_r, hy_r, ln_r]
    ranks = np.vstack([_pct(S, r) for S, r in zip(SORTED, raws)])
    return np.exp(np.mean(np.log(ranks), axis=0))

def score_batch_relative(seqs):
    """RPES of candidate sequences WITHIN THE BATCH (the original behaviour)."""
    if len(seqs) == 0:
        return np.array([])
    Xp = np.stack([compute_physicochemical(s) for s in seqs])
    rf_r = rf_model.predict_proba(Xp)[:, 1]
    bi_r = bilstm_model.predict_proba_batched(seqs, batch_size=5000)
    mlp = []
    for clf in ml_classifiers:
        try:
            mlp.append(clf.predict_proba(Xp)[:, 1])
        except Exception:
            mlp.append(np.zeros(len(seqs)))
    comp_r = np.mean(mlp, axis=0)
    ens_r = np.zeros(len(seqs))
    hy = np.array([np.mean([KD_SCALE.get(aa, 0) for aa in s]) for s in seqs])
    hq1, hq3 = np.percentile(hy, [25, 75])
    hy_r = np.exp(-((hy - (hq1 + hq3) / 2) ** 2) / (2 * max((hq3 - hq1) / 2, 0.1) ** 2))
    ln = np.array([len(s) for s in seqs])
    ln_r = np.exp(-((ln - 9.9) ** 2) / (2 * 4.7 ** 2))
    raws = [rf_r, bi_r, comp_r, ens_r, hy_r, ln_r]
    ranks = np.vstack([(rankdata(r, method='average') / len(r)).astype(np.float64)
                       for r in raws])
    return np.exp(np.mean(np.log(ranks), axis=0))

# ------------------------------------------------------------------
# 4. Runs
# ------------------------------------------------------------------
results = {'verification': verify, 'seeds': SEEDS, 'runs': {}}

def make_init(seed):
    """Identical initial pool to Layer 3 in rpes_full_experiment.py."""
    np.random.seed(seed)
    random.seed(seed)
    top_half = np.argsort(rpes_lib_recomputed)[len(rpes_lib_recomputed) // 2:]
    init_idx = np.random.choice(top_half, 5000, replace=False)
    return vae_latents[init_idx].copy(), rpes_lib_recomputed[init_idx].copy()

def gen_candidates(parents, n_per, noise_scale):
    out = []
    for parent in parents:
        out.append(parent + np.random.randn(n_per, parent.shape[0]) * noise_scale)
    return np.vstack(out)[:n_per * len(parents)]

def decode(cands):
    seqs, lats = [], []
    for c in cands:
        s = pca.decode_latent(c[:VAE_LATENT_DIM])
        if len(s) >= 2:
            seqs.append(s)
            lats.append(c)
    return seqs, np.array(lats)

# ---- L3_orig: original implementation (batch-relative candidate scoring) ----
log("\n[1/5] L3_orig  (original: batch-relative candidate scoring)")
l3o = []
for seed in SEEDS:
    pool_lat, pool_rpes = make_init(seed)
    best_hist, noise = [], 0.30
    for rnd in range(N_ROUNDS):
        best_hist.append(float(np.max(pool_rpes)))
        parents = pool_lat[np.argsort(pool_rpes)[-TOP_PARENTS:]]
        n_per = max(1, N_CAND_PER_ROUND // len(parents))
        cands = gen_candidates(parents, n_per, noise)
        seqs, lats = decode(cands)
        if len(seqs) >= 15:
            sc = score_batch_relative(seqs)
            tk = min(15, len(seqs))
            ti = np.argsort(sc)[-tk:]
            pool_lat = np.vstack([pool_lat, lats[ti][:, :VAE_LATENT_DIM]])
            pool_rpes = np.concatenate([pool_rpes, sc[ti]])
        noise *= (0.05 / 0.30) ** (1.0 / N_ROUNDS)
    l3o.append({'seed': seed, 'initial_best': best_hist[0], 'final_best': best_hist[-1],
                'improvement': best_hist[-1] - best_hist[0], 'history': best_hist})
    log(f"  seed {seed}: {best_hist[0]:.4f} -> {best_hist[-1]:.4f} "
        f"(+{best_hist[-1]-best_hist[0]:.4f})")

# ---- L3_corr: same loop, library-relative candidate scoring ----
log("\n[2/5] L3_corr  (same loop, library-relative candidate scoring)")
l3c = []
for seed in SEEDS:
    pool_lat, pool_rpes = make_init(seed)
    best_hist, noise = [], 0.30
    for rnd in range(N_ROUNDS):
        best_hist.append(float(np.max(pool_rpes)))
        parents = pool_lat[np.argsort(pool_rpes)[-TOP_PARENTS:]]
        n_per = max(1, N_CAND_PER_ROUND // len(parents))
        cands = gen_candidates(parents, n_per, noise)
        seqs, lats = decode(cands)
        if len(seqs) >= 15:
            sc = score_library_relative(seqs)
            tk = min(15, len(seqs))
            ti = np.argsort(sc)[-tk:]
            pool_lat = np.vstack([pool_lat, lats[ti][:, :VAE_LATENT_DIM]])
            pool_rpes = np.concatenate([pool_rpes, sc[ti]])
        noise *= (0.05 / 0.30) ** (1.0 / N_ROUNDS)
    l3c.append({'seed': seed, 'initial_best': best_hist[0], 'final_best': best_hist[-1],
                'improvement': best_hist[-1] - best_hist[0], 'history': best_hist})
    log(f"  seed {seed}: {best_hist[0]:.4f} -> {best_hist[-1]:.4f} "
        f"(+{best_hist[-1]-best_hist[0]:.4f})")

# ---- M1 / M2: matched initialisation, single pass, library-relative ----
def matched_single_pass(n_per):
    out = []
    for seed in SEEDS:
        pool_lat, pool_rpes = make_init(seed)
        init_best = float(np.max(pool_rpes))
        parents = pool_lat[np.argsort(pool_rpes)[-TOP_PARENTS:]]
        cands = gen_candidates(parents, n_per, 0.30)
        seqs, _ = decode(cands)
        cand_best = float(np.max(score_library_relative(seqs))) if seqs else float('nan')
        out.append({'seed': seed, 'initial_best': init_best,
                    'candidate_best': cand_best,
                    'best_overall': max(init_best, cand_best),
                    'n_candidates': len(seqs)})
    return out

log("\n[3/5] M1  (matched init, single pass, 240 candidates)")
m1 = matched_single_pass(N_CAND_PER_ROUND // TOP_PARENTS)   # n_per = 8 -> 240
for r in m1:
    log(f"  seed {r['seed']}: init={r['initial_best']:.4f} cand={r['candidate_best']:.4f} "
        f"best={r['best_overall']:.4f}")

log("\n[4/5] M2  (matched init, single pass, 2400 candidates = loop's total budget)")
m2 = matched_single_pass((N_CAND_PER_ROUND // TOP_PARENTS) * N_ROUNDS)  # n_per = 80 -> 2400
for r in m2:
    log(f"  seed {r['seed']}: init={r['initial_best']:.4f} cand={r['candidate_best']:.4f} "
        f"best={r['best_overall']:.4f}")

log("\n[5/5] A_corr  (original Ablation A procedure, library-relative scoring)")
a_corr = []
for seed in SEEDS:
    np.random.seed(seed)
    random.seed(seed)
    rand_latent = np.random.randn(250, VAE_LATENT_DIM) * 0.3
    seqs = [pca.decode_latent(v) for v in rand_latent]
    valid = [s for s in seqs if len(s) >= 2]
    sc = score_library_relative(valid)
    a_corr.append({'seed': seed, 'best': float(np.max(sc)), 'n_valid': len(valid)})
    log(f"  seed {seed}: best={np.max(sc):.4f} (n_valid={len(valid)})")

def summ(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return {'per_seed': [float(x) for x in v],
            'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': int(len(v))}

results['runs'] = {
    'L3_orig': {'final': summ(l3o, 'final_best'),
                'improvement': summ(l3o, 'improvement'), 'details': l3o},
    'L3_corr': {'final': summ(l3c, 'final_best'),
                'improvement': summ(l3c, 'improvement'), 'details': l3c},
    'M1_matched_init_240': {'best': summ(m1, 'best_overall'), 'details': m1},
    'M2_matched_init_2400': {'best': summ(m2, 'best_overall'), 'details': m2},
    'A_corr_random_latent': {'best': summ(a_corr, 'best'), 'details': a_corr},
}
results['library_optimum'] = float(rpes_lib_recomputed.max())
results['elapsed_sec'] = float(time.time() - t_start)

out_path = os.path.join(OUTPUT_DIR, 'ablation_matched_init.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
log(f"\nWrote {out_path}")
log(f"Total elapsed: {results['elapsed_sec']:.1f}s")
