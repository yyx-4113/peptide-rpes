"""b_pivot_pca_recon.py — recompute PCA reconstruction accuracy + decoder
validity on the FULL 403,461-peptide library, using the same PCA procedure as
the experiment (fit on a 52,517 random sample, seed=42; inverse-transform +
argmax decode). This replaces the original 52,517-based 76.7% figure with a
full-library-verified number so every manuscript figure is traceable.
"""
import os, sys, time, random, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_pipeline import (
    load_peptide_library, flatten_onehot, PCALatentSpace,
    AA_LIST, N_AMINO_ACIDS, MAX_PEPTIDE_LEN,
)
import phase1_pipeline
from config import DATA_DIR, OUT_DIR
phase1_pipeline.DATA_ROOT = DATA_DIR  # where merged_peptide_library.csv lives

t0 = time.time()
df = load_peptide_library()
peptides = df['peptide'].tolist()
print(f"loaded {len(peptides):,} peptides ({time.time()-t0:.1f}s)")

# One-hot all peptides
oh = flatten_onehot(peptides)
print(f"onehot shape {oh.shape} ({time.time()-t0:.1f}s)")

# Fit PCA on a 52,517 random sample (seed=42) — same protocol as experiment
random.seed(42)
sample = random.sample(peptides, 52517)
oh_sample = flatten_onehot(sample)
pca = PCALatentSpace(n_components=64)
pca.fit(oh_sample)
print(f"PCA explained variance = {pca.pca.explained_variance_ratio_.sum():.3f}")

# Encode + decode full library
latent = pca.encode(oh)
recon = pca.decode(latent)  # (N, 400)
recon2d = recon.reshape(-1, MAX_PEPTIDE_LEN, N_AMINO_ACIDS)
idx = recon2d.argmax(axis=2)              # (N, L)
maxp = recon2d.max(axis=2)                # (N, L)

# Reconstruction accuracy: per-position identity over full length
accs = []
valid = 0
for i, seq in enumerate(peptides):
    ri = idx[i]
    rseq = ''.join(AA_LIST[j] for j in ri if j < len(AA_LIST))
    if len(rseq) >= 2:
        valid += 1
    min_len = min(len(seq), len(rseq))
    if min_len > 0:
        matches = sum(1 for k in range(min_len) if seq[k] == rseq[k])
        accs.append(matches / max(len(seq), 1))
acc = float(np.mean(accs))
print(f"reconstruction accuracy = {acc:.4f} (SD unknown)")
print(f"decoder validity (>=2 aa) = {valid/len(peptides)*100:.1f}%")

# Also report mean decoded length
lens = [len(''.join(AA_LIST[j] for j in idx[i] if j < len(AA_LIST))) for i in range(0, len(peptides), 1000)]
print(f"mean decoded length (sampled) = {np.mean(lens):.2f}")

out = {
    'n_peptides': len(peptides),
    'pca_explained_variance': float(pca.pca.explained_variance_ratio_.sum()),
    'reconstruction_accuracy': acc,
    'decoder_validity_pct': valid / len(peptides) * 100,
    'mean_decoded_length': float(np.mean(lens)),
}
with open(os.path.join(OUT_DIR, 'b_pivot_pca_recon.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("saved output/b_pivot_pca_recon.json")
