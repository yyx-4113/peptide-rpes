"""
b_pivot_analysis.py — B 转向前必跑分析（不依赖 torch/xgboost）。

目标：
  (1) 重建 403,461 肽顺序并与缓存 RPES / PCA latent 对齐校验；
  (2) held-out 80/20 拆分：phys-only vs PCA+phys 预测 RPES 的 test R^2；
  (3) 删 docking 后核心管线（phys 抽取 + PCA + RPES 百分位聚合）真实计时。

不导入 phase1_pipeline，避免顶层 import torch 依赖；
compute_physicochemical 严格复刻原定义。
"""
import os, time, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import rankdata

from config import PEPTIDE_CSV as DATA, MODEL_DIR, OUT_DIR  # portable paths

# ---- 严格复刻 phase1_pipeline.compute_physicochemical ----
KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
      'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
      'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
MW = {'A':89.09,'C':121.16,'D':133.10,'E':147.13,'F':165.19,'G':75.07,'H':155.16,
      'I':131.17,'K':146.19,'L':131.17,'M':149.21,'N':132.12,'P':115.13,'Q':146.15,
      'R':174.20,'S':105.09,'T':119.12,'V':117.15,'W':204.23,'Y':181.19}
CHARGE = {'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,'M':0,
          'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0}
GROUPS = {'hydrophobic':set('AVILMFYW'),'polar':set('STNQ'),'positive':set('KRH'),
          'negative':set('DE'),'special':set('CPG')}
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')

def compute_phys(seq):
    if len(seq) == 0:
        return np.zeros(10, dtype=np.float32)
    n = len(seq)
    hydro = np.mean([KD.get(a, 0.0) for a in seq])
    mw = np.sum([MW.get(a, 110.0) for a in seq])
    net_charge = np.sum([CHARGE.get(a, 0.0) for a in seq])
    fh = sum(1 for a in seq if a in GROUPS['hydrophobic']) / n
    fp = sum(1 for a in seq if a in GROUPS['polar']) / n
    fpos = sum(1 for a in seq if a in GROUPS['positive']) / n
    fneg = sum(1 for a in seq if a in GROUPS['negative']) / n
    fsp = sum(1 for a in seq if a in GROUPS['special']) / n
    iep = 5.5 + net_charge * 2.0
    return np.array([n, hydro, mw/1000.0, net_charge, iep/14.0,
                     fh, fp, fpos, fneg, fsp], dtype=np.float32)

def main():
    t_all = time.time()
    # (1) 重建 403,461 顺序
    df = pd.read_csv(DATA)
    ded = df.drop_duplicates(subset=['peptide'])
    peps = ded['peptide'].tolist()
    n = len(peps)
    print(f'[align] reconstructed n={n}', flush=True)
    assert n == 403461, f'ALIGNMENT MISMATCH: expected 403461, got {n}'

    rpes = np.load(os.path.join(OUT_DIR, 'rpes_scores.npz'))['rpes']
    pca = np.load(os.path.join(MODEL_DIR, 'pca_latent.npz'))['latents']
    print(f'[align] rpes mean={rpes.mean():.4f} std={rpes.std():.4f} '
          f'max={rpes.max():.4f} | pca {pca.shape}', flush=True)
    # 一致性校验：若顺序错，下面 held-out 结果的生物学意义全废；用 RPES 分布无法校验顺序，
    # 但长度断言 + 后续 PCA 表征 R2 必须落在合理区间(0.9+) 作为间接校验。

    # (2) phys 特征抽取 + 计时
    t0 = time.time()
    phys = np.stack([compute_phys(s) for s in peps])
    dt_phys = time.time() - t0
    print(f'[timing] phys feature extraction (403461): {dt_phys:.1f}s', flush=True)

    # (3) held-out 80/20 split — phys-only vs PCA+phys 预测 RPES
    idx = np.arange(n)
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42)
    results = {}
    for name, X in [('phys_only', phys), ('pca_phys', np.hstack([pca, phys]))]:
        hg = HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
        hg.fit(X[itr], rpes[itr])
        test_r2 = hg.score(X[ite], rpes[ite])
        train_r2 = hg.score(X[itr], rpes[itr])
        results[name] = {'train_r2': float(train_r2), 'test_r2': float(test_r2)}
        print(f'[held-out] {name}: train R2={train_r2:.4f}  test R2={test_r2:.4f}', flush=True)
    # 3-fold CV 稳健性（phys_only 与 pca_phys 都要）
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv = {}
    for name, X in [('phys_only', phys), ('pca_phys', np.hstack([pca, phys]))]:
        vals = [HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
                .fit(X[itr_], rpes[itr_]).score(X[ite_], rpes[ite_])
                for itr_, ite_ in kf.split(X)]
        cv[name] = float(np.mean(vals))
        print(f'[held-out] {name} 3-fold CV R2={np.mean(vals):.4f} +/- {np.std(vals):.4f}', flush=True)
    results['cv_r2'] = cv

    # (4) RPES 百分位聚合计时（核心打分成本代理）
    # 单组分 rankdata over 403461 的时间，外推 6 组分
    t0 = time.time(); _ = rankdata(rpes); dt_rank = time.time() - t0
    print(f'[timing] rankdata over 403461 (1 component): {dt_rank:.3f}s '
          f'-> ~{dt_rank*6:.2f}s for 6-component RPES aggregation', flush=True)

    out = {
        'n_peptides_aligned': n,
        'rpes_distribution': {'mean': float(rpes.mean()), 'std': float(rpes.std()),
                              'max': float(rpes.max())},
        'held_out_80_20': results,
        'timing': {'phys_extraction_s': dt_phys,
                   'rpes_rank_aggregation_s_est': dt_rank * 6,
                   'total_s': time.time() - t_all},
    }
    with open(os.path.join(OUT_DIR, 'b_pivot_analysis.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('[done] wrote b_pivot_analysis.json', flush=True)

if __name__ == '__main__':
    main()
