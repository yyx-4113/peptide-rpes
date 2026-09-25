"""rpes_full_experiment.py — Re-run all experiments under RPES framework.

Replaces original weighted-average rescue score experiments:
  Exp 1: Three-layer comparison (Natural Screening / Random Sampling / PepDesign-Active)
  Exp 2: Ablation study (A: No AL, B: No Conditioning, C: Physicochemical Only)
  Exp 3: Peptide property analysis (RPES top-20 sequence characteristics)

All experiments use RPES as the objective function.
"""

import os, sys, json, time, random, gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_pipeline import (
    OUTPUT_DIR, MODEL_DIR, SEEDS, AA_LIST, AA_TO_IDX, MAX_PEPTIDE_LEN,
    N_AMINO_ACIDS, VAE_LATENT_DIM, AL_ROUNDS, AL_CANDIDATES,
    PCALatentSpace, load_peptide_library, flatten_onehot,
    compute_physicochemical, KD_SCALE, AA_CHARGE, log
)

# ---- Portable path override (config.py) ----
import phase1_pipeline
from config import DATA_DIR, MODEL_DIR as CFG_MODEL_DIR, OUT_DIR as CFG_OUT_DIR
phase1_pipeline.DATA_ROOT = DATA_DIR       # where merged_peptide_library.csv lives
phase1_pipeline.OUTPUT_DIR = CFG_OUT_DIR   # where JSON / rpes_scores.npz are written
phase1_pipeline.MODEL_DIR = CFG_MODEL_DIR  # where pca_latent.npz is read/written
OUTPUT_DIR = CFG_OUT_DIR                    # rebind local names imported above
MODEL_DIR = CFG_MODEL_DIR

# ============================================================
# Load cached data
# ============================================================
log("Loading cached data...")
peptides_df = load_peptide_library()
unique_peptides = peptides_df['peptide'].tolist()

pca_data = np.load(os.path.join(MODEL_DIR, 'pca_latent.npz'), allow_pickle=True)
vae_latents = pca_data['latents']

# Load RPES from new_methods output
new_methods_results = json.load(open(os.path.join(OUTPUT_DIR, 'new_methods_results.json')))
rpes_stats = new_methods_results['rpes_stats']

# Recompute RPES if not in results (re-use from new_methods.py compute_rpes)
# For this we need the models. Let's train them quick.
from phase1_pipeline import train_rf_classifier, SklearnBiLSTMProxy, train_surrogate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# FIX (2026-09-25): seed ALL RNGs before any model is built. Previously
# train_rf_classifier used an UNSEEDED random.sample for its training set and the
# composite_ml subset used an UNSEEDED np.random.choice — both made RPES
# (hence its reported max) non-deterministic (cache 0.8148 vs headline 0.8386).
random.seed(42)
np.random.seed(42)
rf_model = train_rf_classifier(peptides_df)
bilstm_model = SklearnBiLSTMProxy()
bilstm_model.fit(unique_peptides)

# ML classifiers for composite_ml
X_phys_full = np.stack([compute_physicochemical(s) for s in unique_peptides])
n_subset = min(5000, len(unique_peptides))
np.random.seed(42)  # FIX: deterministic composite_ml training subset
subset_idx = np.random.choice(len(unique_peptides), n_subset, replace=False)
X_sub = X_phys_full[subset_idx]
y_sub = rf_model.predict_proba(X_sub)[:, 1]
y_bin = (y_sub > 0.5).astype(int)

xgb = XGBClassifier(n_estimators=50, max_depth=4, use_label_encoder=False,
                    eval_metric='logloss', random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
for clf in [xgb, svm, knn]:
    clf.fit(X_sub, y_bin)
ml_classifiers = [xgb, svm, knn]

# Load RPES from pre-computed file or recompute
rpes_cache = os.path.join(OUTPUT_DIR, 'rpes_scores.npz')
if os.path.exists(rpes_cache):
    log("Loading cached RPES scores...")
    rpes = np.load(rpes_cache)['rpes']
else:
    log("Computing RPES scores (this will take ~4 minutes)...")
    from new_methods import compute_rpes
    rpes, component_ranks = compute_rpes(
        unique_peptides, rf_model, bilstm_model,
        ml_classifiers, None, None
    )
    np.savez(rpes_cache, rpes=rpes)

phys_features = X_phys_full  # already computed

# Build PCA model for decoding
vae_peptides = random.sample(unique_peptides, 52517)
pca = PCALatentSpace()
pca.fit(flatten_onehot(vae_peptides))

log(f"Loaded: RPES scores ({len(rpes)}), latents ({vae_latents.shape}), "
    f"phys features ({phys_features.shape})")

# ============================================================
# EXPERIMENT 1: Three-Layer Comparison (RPES)
# ============================================================
log("\n" + "=" * 60)
log("EXPERIMENT 1: Three-Layer Comparison (RPES)")
log("=" * 60)

# Layer 1: Natural screening — best RPES from natural library
layer1_best_idx = np.argmax(rpes)
layer1_best_score = rpes[layer1_best_idx]
layer1_best_peptide = unique_peptides[layer1_best_idx]
log(f"Layer 1 (Natural Screening): best RPES = {layer1_best_score:.4f}")
log(f"  Peptide: {layer1_best_peptide}")

# Layer 2: Random VAE latent sampling — score random latents
np.random.seed(42)
random_latent_idx = np.random.choice(len(vae_latents), 3000, replace=False)
layer2_scores = rpes[random_latent_idx]
layer2_best = layer2_scores.max()
layer2_mean = layer2_scores.mean()
layer2_top5 = np.mean(np.sort(layer2_scores)[-5:])
log(f"Layer 2 (Random Sampling): best = {layer2_best:.4f}, "
    f"mean = {layer2_mean:.4f}, top5_mean = {layer2_top5:.4f}")

# Layer 3: RPES-guided active learning (3 replicates)
log("\nLayer 3 (PepDesign-Active with RPES):")
al_layer3_results = []
for seed in [42, 123, 456]:
    np.random.seed(seed)
    random.seed(seed)

    # Initial pool: random 5000 from top 50% of RPES
    top_half = np.argsort(rpes)[len(rpes)//2:]
    init_idx = np.random.choice(top_half, 5000, replace=False)
    pool_latent = vae_latents[init_idx].copy()
    pool_rpes = rpes[init_idx].copy()

    best_per_round = []
    noise_scale = 0.30

    for rnd in range(10):
        curr_best = np.max(pool_rpes)
        best_per_round.append(float(curr_best))

        top_indices = np.argsort(pool_rpes)[-30:]
        parents = pool_latent[top_indices]

        # Generate candidates
        candidates = []
        n_per = max(1, 250 // len(parents))
        for parent in parents:
            noise = np.random.randn(n_per, parent.shape[0]) * noise_scale
            candidates.append(parent + noise)
        candidates = np.vstack(candidates)[:250]

        # Decode and score
        new_seqs = []
        new_latents = []
        for cand in candidates:
            seq = pca.decode_latent(cand[:VAE_LATENT_DIM])
            if len(seq) >= 2:
                new_seqs.append(seq)
                new_latents.append(cand)

        if len(new_seqs) >= 15 and len(new_latents) >= 15:
            # Score new peptides using RPES
            new_phys = np.stack([compute_physicochemical(s) for s in new_seqs])
            new_combined = np.hstack([np.array(new_latents)[:, :VAE_LATENT_DIM], new_phys])
            from new_methods import compute_rpes
            new_rpes, _ = compute_rpes(
                new_seqs, rf_model, bilstm_model,
                ml_classifiers, None, None
            )

            top_k = min(15, len(new_seqs))
            top_new_idx = np.argsort(new_rpes)[-top_k:]
            pool_latent = np.vstack([pool_latent,
                                     np.array(new_latents)[top_new_idx][:, :VAE_LATENT_DIM]])
            pool_rpes = np.concatenate([pool_rpes, new_rpes[top_new_idx]])

        noise_scale *= (0.05 / 0.30) ** (1.0 / 10)

    al_layer3_results.append({
        'seed': seed,
        'best_history': best_per_round,
        'initial_best': best_per_round[0],
        'final_best': best_per_round[-1],
        'improvement': best_per_round[-1] - best_per_round[0],
    })
    log(f"  Seed {seed}: initial={best_per_round[0]:.4f} -> "
        f"final={best_per_round[-1]:.4f} (+{best_per_round[-1]-best_per_round[0]:.4f})")

layer3_final = [r['final_best'] for r in al_layer3_results]
log(f"\nLayer 3 summary: mean={np.mean(layer3_final):.4f} +/- {np.std(layer3_final):.4f}")

three_layer_results = {
    'layer1_natural_best': float(layer1_best_score),
    'layer1_peptide': layer1_best_peptide,
    'layer2_random_best': float(layer2_best),
    'layer2_random_mean': float(layer2_mean),
    'layer2_random_top5': float(layer2_top5),
    'layer3_al_mean': float(np.mean(layer3_final)),
    'layer3_al_std': float(np.std(layer3_final)),
    'layer3_al_min': float(min(layer3_final)),
    'layer3_al_max': float(max(layer3_final)),
    'layer3_details': al_layer3_results,
    'layer3_improvement_vs_layer2_pct': float(
        (np.mean(layer3_final) - layer2_best) / max(layer2_best, 1e-8) * 100
    ),
    'layer3_pct_of_natural': float(
        np.mean(layer3_final) / max(layer1_best_score, 1e-8) * 100
    ),
}
log(f"\nLayer 3 improvement over random: {three_layer_results['layer3_improvement_vs_layer2_pct']:.1f}%")
log(f"Layer 3 as % of natural best: {three_layer_results['layer3_pct_of_natural']:.1f}%")

# ============================================================
# EXPERIMENT 2: Ablation Study (RPES)
# ============================================================
log("\n" + "=" * 60)
log("EXPERIMENT 2: Ablation Study (RPES)")
log("=" * 60)

# Ablation A: No Active Learning (single-pass generation)
log("\nAblation A: No Active Learning (single-pass generation)...")
# Single-pass: generate 250 random latent vectors, decode, score best
np.random.seed(42)
abl_A_scores = []
for _ in range(5):  # 5 replicates
    rand_latent = np.random.randn(250, VAE_LATENT_DIM) * 0.3
    seqs_A = [pca.decode_latent(v) for v in rand_latent]
    valid_A = [(i, s) for i, s in enumerate(seqs_A) if len(s) >= 2]
    if valid_A:
        valid_seqs = [s for _, s in valid_A]
        from new_methods import compute_rpes
        rpes_A, _ = compute_rpes(valid_seqs, rf_model, bilstm_model,
                                 ml_classifiers, None, None)
        abl_A_scores.append(np.max(rpes_A))
log(f"  Ablation A: best RPES = {np.mean(abl_A_scores):.4f} +/- {np.std(abl_A_scores):.4f}")

# Ablation B: No Conditioning (random latent search, no optimization)
log("\nAblation B: No Conditioning (random latent search)...")
abl_B_scores = []
for _ in range(5):
    rand_idx = np.random.choice(len(vae_latents), 500, replace=False)
    abl_B_scores.append(np.max(rpes[rand_idx]))
log(f"  Ablation B: best RPES = {np.mean(abl_B_scores):.4f} +/- {np.std(abl_B_scores):.4f}")

# Ablation C: Physicochemical Features Only (no VAE latent)
log("\nAblation C: Physicochemical Features Only (no VAE latent)...")
# Train surrogate on physchem only, score natural library
from sklearn.ensemble import HistGradientBoostingRegressor
phys_only = phys_features  # (N, 10)
kf_C = np.random.choice(len(phys_only), min(15000, len(phys_only)), replace=False)
hgbr_C = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
hgbr_C.fit(phys_only[kf_C], rpes[kf_C])
# Score natural library
abl_C_scores = hgbr_C.predict(phys_only)
# Check overfitting via simple split
split_C = int(len(phys_only) * 0.8)
train_C = hgbr_C.score(phys_only[:split_C], rpes[:split_C])
test_C_subset = np.random.choice(
    np.arange(split_C, len(phys_only)), min(5000, len(phys_only) - split_C), replace=False
)
test_C = hgbr_C.score(phys_only[test_C_subset], rpes[test_C_subset])
log(f"  Ablation C: train R^2 = {train_C:.4f}, test R^2 = {test_C:.4f}")
log(f"  Ablation C: predicted best RPES = {abl_C_scores.max():.4f}")

ablation_results = {
    'full_method': float(np.mean(layer3_final)),
    'full_method_std': float(np.std(layer3_final)),
    'abl_A_no_al_mean': float(np.mean(abl_A_scores)),
    'abl_A_no_al_std': float(np.std(abl_A_scores)),
    'abl_A_delta': float(np.mean(layer3_final) - np.mean(abl_A_scores)),
    'abl_B_no_cond_mean': float(np.mean(abl_B_scores)),
    'abl_B_no_cond_std': float(np.std(abl_B_scores)),
    'abl_B_delta': float(np.mean(layer3_final) - np.mean(abl_B_scores)),
    'abl_C_physchem_train_r2': float(train_C),
    'abl_C_physchem_test_r2': float(test_C),
    'abl_C_predicted_best': float(abl_C_scores.max()),
}

log(f"\nAblation Summary:")
log(f"  Full method:          {ablation_results['full_method']:.4f}")
log(f"  A (No AL):            {ablation_results['abl_A_no_al_mean']:.4f} "
    f"(d = {ablation_results['abl_A_delta']:+.4f})")
log(f"  B (No Conditioning):  {ablation_results['abl_B_no_cond_mean']:.4f} "
    f"(d = {ablation_results['abl_B_delta']:+.4f})")
log(f"  C (Physchem only):    train R^2={train_C:.3f}, test R^2={test_C:.3f}")

# ============================================================
# EXPERIMENT 3: Peptide Property Analysis (RPES top-20)
# ============================================================
log("\n" + "=" * 60)
log("EXPERIMENT 3: RPES Top-20 Peptide Property Analysis")
log("=" * 60)

top20_idx = np.argsort(rpes)[-20:][::-1]
top20_peptides = [unique_peptides[i] for i in top20_idx]
top20_scores = rpes[top20_idx]

# Amino acid composition analysis
aa_freq_top20 = {aa: 0 for aa in AA_LIST}
total_aa_top20 = 0
for seq in top20_peptides:
    for aa in seq:
        if aa in aa_freq_top20:
            aa_freq_top20[aa] += 1
            total_aa_top20 += 1

# Natural background frequencies
aa_freq_natural = {aa: 0 for aa in AA_LIST}
total_aa_natural = 0
for seq in unique_peptides:
    for aa in seq:
        if aa in aa_freq_natural:
            aa_freq_natural[aa] += 1
            total_aa_natural += 1

# Compute enrichments
enrichments = {}
for aa in AA_LIST:
    f_top = aa_freq_top20[aa] / max(total_aa_top20, 1)
    f_nat = aa_freq_natural[aa] / max(total_aa_natural, 1)
    enrichments[aa] = {
        'top20_pct': round(f_top * 100, 1),
        'natural_pct': round(f_nat * 100, 1),
        'enrichment': round(f_top / max(f_nat, 1e-8), 2),
    }

# Sort by enrichment
sorted_enrich = sorted(enrichments.items(), key=lambda x: x[1]['enrichment'], reverse=True)

log("\nTop-20 RPES peptide amino acid enrichment:")
log(f"{'AA':<6} {'Top20%':<10} {'Natural%':<10} {'Enrichment':<12}")
log("-" * 40)
for aa, stats in sorted_enrich[:10]:
    if stats['enrichment'] > 1.0:
        log(f"{aa:<6} {stats['top20_pct']:<10.1f} {stats['natural_pct']:<10.1f} "
            f"{stats['enrichment']:<12.2f}")

# Length distribution
top20_lengths = [len(s) for s in top20_peptides]
all_lengths = [len(s) for s in unique_peptides]
log(f"\nLength: top20 mean={np.mean(top20_lengths):.1f} (SD={np.std(top20_lengths):.1f}), "
    f"natural mean={np.mean(all_lengths):.1f}")

# Sequence motif analysis
log("\nTop-20 RPES peptides:")
for i, (seq, score) in enumerate(zip(top20_peptides, top20_scores)):
    # Compute properties
    hydro = np.mean([KD_SCALE.get(aa, 0) for aa in seq])
    charge = sum(AA_CHARGE.get(aa, 0) for aa in seq)
    log(f"  {i+1:2d}. {seq:20s}  RPES={score:.4f}  "
        f"len={len(seq):2d}  hydro={hydro:+.2f}  charge={charge:+.0f}")

peptide_analysis = {
    'top20_peptides': top20_peptides,
    'top20_scores': top20_scores.tolist(),
    'top20_mean_length': float(np.mean(top20_lengths)),
    'top20_aa_composition': {aa: stats for aa, stats in enrichments.items()},
    'natural_mean_length': float(np.mean(all_lengths)),
    'enriched_residues': [
        {'aa': aa, 'enrichment': stats['enrichment'],
         'top20_pct': stats['top20_pct'], 'natural_pct': stats['natural_pct']}
        for aa, stats in sorted_enrich if stats['enrichment'] > 1.5
    ],
}

# ============================================================
# SAVE ALL RESULTS
# ============================================================
all_results = {
    'experiment': 'RPES Full Re-run',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'rpes_distribution': rpes_stats,
    'three_layer_comparison': three_layer_results,
    'ablation_study': ablation_results,
    'peptide_analysis': peptide_analysis,
}

output_path = os.path.join(OUTPUT_DIR, 'rpes_full_results.json')
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

log(f"\n{'=' * 60}")
log(f"All experiments complete. Results saved to: {output_path}")
log(f"{'=' * 60}")

# Print key numbers for manuscript
log("\n=== KEY MANUSCRIPT NUMBERS (RPES) ===")
log(f"Layer 1 (Natural):     {three_layer_results['layer1_natural_best']:.4f}")
log(f"Layer 2 (Random):      {three_layer_results['layer2_random_best']:.4f}")
log(f"Layer 3 (PepDesign):   {three_layer_results['layer3_al_mean']:.4f} +/- {three_layer_results['layer3_al_std']:.4f}")
log(f"Layer 3 vs Layer 2:    {three_layer_results['layer3_improvement_vs_layer2_pct']:.1f}%")
log(f"Ablation A (no AL):    {ablation_results['abl_A_no_al_mean']:.4f} (d={ablation_results['abl_A_delta']:+.4f})")
log(f"Ablation B (no Cond):  {ablation_results['abl_B_no_cond_mean']:.4f} (d={ablation_results['abl_B_delta']:+.4f})")
log(f"Top peptide:           {top20_peptides[0]} (RPES={top20_scores[0]:.4f})")
log(f"Enriched AAs:          {[e['aa'] for e in peptide_analysis['enriched_residues']]}")
