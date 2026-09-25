"""b_pivot_pca_timing.py — 纯 sklearn 实测 PCA 拟合/变换耗时（删 docking 后核心管线成本）。"""
import os, time, numpy as np, pandas as pd
from sklearn.decomposition import PCA

from config import PEPTIDE_CSV as DATA  # portable path
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}
MAX_LEN = 20

def flatten_onehot(peps):
    N = len(peps)
    oh = np.zeros((N, MAX_LEN * 20), dtype=np.float32)
    for i, p in enumerate(peps):
        for j, a in enumerate(p[:MAX_LEN]):
            if a in AA_TO_IDX:
                oh[i, j * 20 + AA_TO_IDX[a]] = 1.0
    return oh

df = pd.read_csv(DATA).drop_duplicates(subset=['peptide'])
peps = df['peptide'].tolist()
n = len(peps)
print(f'n={n}', flush=True)

# 全库 one-hot
t0 = time.time()
oh = flatten_onehot(peps)
dt_oh = time.time() - t0
print(f'one-hot build (403461x400): {dt_oh:.1f}s', flush=True)

# PCA fit
pca = PCA(n_components=64, random_state=42)
t0 = time.time(); pca.fit(oh); dt_fit = time.time() - t0
var = pca.explained_variance_ratio_.sum()
print(f'PCA fit (64d): {dt_fit:.1f}s | explained variance={var:.3f}', flush=True)

# PCA transform（per-library 成本）
t0 = time.time(); _ = pca.transform(oh); dt_tr = time.time() - t0
print(f'PCA transform (403461): {dt_tr:.1f}s', flush=True)

print('\n=== COMPUTE BUDGET (no docking) ===')
print(f'  one-hot build : {dt_oh:.1f}s')
print(f'  PCA fit (one-time) : {dt_fit:.1f}s')
print(f'  PCA transform (per-library) : {dt_tr:.1f}s')
print(f'  phys extraction (403461) : ~19s  [from b_pivot_analysis]')
print(f'  RPES percentile aggregation (403461) : ~0.6s  [from b_pivot_analysis]')
print(f'  -> per-library RPES scoring (transform+aggregate) : ~{dt_tr+0.6:.1f}s on a single CPU thread')
out = {'n': n, 'onehot_build_s': dt_oh, 'pca_fit_s': dt_fit,
       'pca_transform_s': dt_tr, 'explained_variance': float(var),
       'per_library_scoring_s_est': float(dt_tr + 0.6)}
import json
json.dump(out, open(r'D:\projects\sleep-deprivation-project\pepdesign\output\b_pivot_pca_timing.json', 'w'), indent=2)
print('wrote b_pivot_pca_timing.json', flush=True)
