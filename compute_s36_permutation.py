"""compute_s36_permutation.py — T2-12 (multiplicity control for S3.6 enrichment).

Adds a permutation test + BH-FDR over the six focal residues (H, G, Y, P, W, A)
of the top-20 RPES peptides, and a background regression of RPES on the six
residue fractions (matched length/hydrophobicity background) to show the
association is not purely a selection artifact. Output: s36_permutation.json.
"""
import os, json, time
import numpy as np
from collections import Counter

OUT = r"D:\projects\peptide-rpes\output"
LIB = r"D:\projects\peptide-rpes\data\merged_peptide_library.csv"
FOCAL = ['H', 'G', 'Y', 'P', 'W', 'A']
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

t0 = time.time()
rpes = np.load(os.path.join(OUT, 'rpes_scores.npz'))['rpes']

import pandas as pd
df = pd.read_csv(LIB)
df = df.drop_duplicates(subset=['peptide']).reset_index(drop=True)  # align to dedup library (403,461)
peptides = df['peptide'].astype(str).tolist()
lib_n = len(peptides)
assert lib_n == len(rpes), (lib_n, len(rpes))

kd_arr = np.array([KD.get(chr(c), 0.0) for c in range(256)])
def hydro_of(s):
    arr = kd_arr[np.frombuffer(s.encode('ascii'), dtype=np.uint8)]
    return float(arr.mean()) if len(arr) else 0.0

top20 = json.load(open(os.path.join(OUT, 'rpes_full_results.json')))['peptide_analysis']['top20_peptides']
top20_hydro = np.array([hydro_of(s) for s in top20])
top20_hydro_mean = float(top20_hydro.mean())
top20_hydro_sd = float(top20_hydro.std(ddof=0))

print("computing background hydrophobicity ...", flush=True)
bg_hydro = np.array([hydro_of(s) for s in peptides])
mask = np.array([(len(s) in (9, 10)) and (abs(bg_hydro[i] - top20_hydro_mean) <= top20_hydro_sd)
                 for i, s in enumerate(peptides)])
matched = [peptides[i] for i in np.where(mask)[0]]
n_matched = len(matched)
print("matched background n =", n_matched, flush=True)

# per-peptide focal counts / length
def focal_matrix(seqs):
    F = np.zeros((len(seqs), len(FOCAL)))
    L = np.zeros(len(seqs))
    for i, s in enumerate(seqs):
        c = Counter(s)
        L[i] = len(s)
        for j, a in enumerate(FOCAL):
            F[i, j] = c.get(a, 0)
    return F, L

Fm, Lm = focal_matrix(matched)

# observed top20 fractions
top_counts = Counter(''.join(top20))
total_top = sum(top_counts[a] for a in AA_LIST)
obs_top_frac = np.array([top_counts.get(a, 0) / total_top for a in FOCAL])
matched_bg_frac = Fm.sum(0) / Lm.sum()
obs_enrich = obs_top_frac / matched_bg_frac

# ---- permutation test for enrichment (chunked) ----
rng = np.random.default_rng(42)
N_PERM = 200000
CHUNK = 20000
perm_ge = np.zeros(len(FOCAL))
for start in range(0, N_PERM, CHUNK):
    n = min(CHUNK, N_PERM - start)
    idx = rng.integers(0, n_matched, size=(n, 20))
    sc = Fm[idx].sum(1)          # (n, 6)
    sl = Lm[idx].sum(1)          # (n,)
    sf = sc / sl[:, None]        # (n, 6)
    perm_ge += (sf >= obs_enrich[None, :]).sum(0)
perm_p = perm_ge / N_PERM
# BH-FDR
order = np.argsort(perm_p)
m = len(FOCAL)
q = np.empty(m)
prev = 1.0
for rank, i in enumerate(reversed(order)):
    r = m - rank
    val = perm_p[i] * m / r
    prev = min(prev, val)
    q[i] = prev
q = np.clip(q, 0, 1)

# ---- background regression: RPES ~ 6 focal fractions (matched background) ----
y = rpes[mask]
X = (Fm / Lm[:, None])  # per-peptide fraction (n_matched, 6)
Xc = X - X.mean(0)
Xs = Xc / Xc.std(0)
Xs1 = np.column_stack([np.ones(n_matched), Xs])
beta, *_ = np.linalg.lstsq(Xs1, y, rcond=None)
yhat = Xs1 @ beta
ss_res = float(((y - yhat) ** 2).sum())
ss_tot = float(((y - y.mean()) ** 2).sum())
R2 = 1 - ss_res / ss_tot
obs_coef = beta[1:]
# vectorized permutation: precompute projection pinv (6 x n_matched)
N_PERM2 = 20000
rng2 = np.random.default_rng(7)
P = np.linalg.pinv(Xs1)               # (7, n_matched)
yarr = y.copy()
# generate permutation matrix (N_PERM2, n_matched)
Yp = np.empty((N_PERM2, n_matched))
for k in range(N_PERM2):
    Yp[k] = rng2.permutation(yarr)
Bp = Yp @ P.T                        # (N_PERM2, 7) coefficients
Yhatp = Bp @ Xs1.T                    # (N_PERM2, n_matched)
ss_tot_perm = ((Yp - Yp.mean(1, keepdims=True)) ** 2).sum(1)
ss_res_perm = ((Yp - Yhatp) ** 2).sum(1)
r2_perm = 1 - ss_res_perm / ss_tot_perm
p_R2 = float((r2_perm >= R2).mean())
coef_perm = Bp[:, 1:]                 # (N_PERM2, 6)
coef_p = np.array([float((np.abs(coef_perm[:, j]) >= abs(obs_coef[j])).mean())
                   for j in range(len(FOCAL))])

out = {
    'focal_residues': FOCAL,
    'top20_n': len(top20),
    'matched_background_n': int(n_matched),
    'library_n': lib_n,
    'top20_fraction_pct': (obs_top_frac * 100).round(3).tolist(),
    'matched_background_fraction_pct': (matched_bg_frac * 100).round(3).tolist(),
    'observed_matched_enrichment': obs_enrich.round(4).tolist(),
    'permutation_p': perm_p.round(6).tolist(),
    'bh_fdr_q': q.round(6).tolist(),
    'background_regression': {
        'n': int(n_matched),
        'R2': round(R2, 6),
        'permutation_p_R2': round(p_R2, 6),
        'standardized_coef': obs_coef.round(5).tolist(),
        'coef_permutation_p': coef_p.round(6).tolist(),
        'note': 'RPES regressed on the six residue fractions over the length/hydrophobicity-matched background; R2 and per-coefficient p by label permutation.',
    },
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(out, open(os.path.join(OUT, 's36_permutation.json'), 'w'), indent=2)
print(json.dumps(out, indent=2))
print("Done in %.1fs" % out['elapsed_sec'])
