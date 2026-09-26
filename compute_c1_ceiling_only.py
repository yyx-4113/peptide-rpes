"""compute_c1_ceiling_only.py — finish T1-1 ceiling metrics from saved arrays."""
import os, json, time
import numpy as np
OUT = r'D:\projects\peptide-rpes\output'
names = ['rf', 'bilstm', 'composite', 'ensemble', 'hydro', 'length']
ranks = [np.asarray(np.load(os.path.join(OUT, 'component_ranks.npz'))[nm], dtype=np.float64)
         for nm in names]
rpes = np.asarray(np.load(os.path.join(OUT, 'rpes_scores.npz'))['rpes'], dtype=np.float64)
n = len(rpes)
t0 = time.time()

arith_equal = np.mean(np.vstack(ranks), axis=0)
w = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10]).reshape(6, 1)
weighted_rank = (w * np.vstack(ranks)).sum(axis=0)

def ceiling_metrics(x, label):
    x = np.asarray(x, dtype=np.float64)
    return {'label': label, 'max': float(x.max()), 'mean': float(x.mean()),
            'p99': float(np.percentile(x, 99)), 'p95': float(np.percentile(x, 95)),
            'frac_above_0.95': float((x > 0.95).mean()),
            'frac_within_0.05_of_max': float((x >= x.max() - 0.05).mean())}

results = {
    'n_peptides': n,
    'rpes_geometric_mean': ceiling_metrics(rpes, 'RPES (geometric mean of ranks)'),
    'arith_mean_equal': ceiling_metrics(arith_equal, 'Arithmetic mean of ranks (equal weight)'),
    'weighted_mean_ranks': ceiling_metrics(weighted_rank, 'Weighted mean of ranks (paper weights)'),
    'component_rank_correlations_with_rpes': {
        nm: float(np.corrcoef(rk, rpes)[0, 1]) for nm, rk in zip(names, ranks)},
    'elapsed_sec': time.time() - t0,
}
with open(os.path.join(OUT, 'c1_ceiling.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
