"""
b_pivot_esm2.py — ESM-2 头对头（C2 核心）。

在 5000 肽基准子集上：
  phys-only(10) vs PCA+phys(74) vs ESM2+phys(330) 预测 RPES 的 3-fold CV R^2。
ESM-2 用 esm2_t6_8M_UR50D（fair_esm 2.0.0 顶层包名 esm），CPU 推理。
目标 RPES 取已对齐缓存 rpes_scores.npz；PCA latent 取已对齐缓存 pca_latent.npz。
"""
import os, time, json, random
import numpy as np, pandas as pd, torch
from esm import pretrained
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from config import PEPTIDE_CSV as DATA, MODEL_DIR, OUT_DIR  # portable paths

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
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cpu')

    df = pd.read_csv(DATA).drop_duplicates(subset=['peptide'])
    peps = df['peptide'].tolist(); N = len(peps)
    print(f'[load] N={N}', flush=True)
    rpes = np.load(os.path.join(OUT_DIR, 'rpes_scores.npz'))['rpes']
    pca = np.load(os.path.join(MODEL_DIR, 'pca_latent.npz'))['latents']

    n_esm = 5000
    esm_idx = random.sample(range(N), n_esm)
    esm_peps = [peps[i] for i in esm_idx]

    # ---- ESM-2 embedding ----
    print('[esm] loading model esm2_t6_8M_UR50D...', flush=True)
    t0 = time.time()
    model, alphabet = pretrained.load_model_and_alphabet_hub('esm2_t6_8M_UR50D')
    model.eval(); model.to(device)
    batch_converter = alphabet.get_batch_converter()
    load_t = time.time() - t0
    print(f'[esm] model load: {load_t:.1f}s', flush=True)

    t0 = time.time()
    batch_size = 64
    emb_list = []
    for s in range(0, n_esm, batch_size):
        batch = [(str(i), esm_peps[i]) for i in range(s, min(s + batch_size, n_esm))]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[6], return_contacts=False)
        rep = out['representations'][6]            # (b, L+2, 320)
        emb = rep.mean(dim=1).cpu().numpy().astype(np.float32)  # mean-pool incl bos/eos
        emb_list.append(emb)
    esm_emb = np.vstack(emb_list)
    emb_t = time.time() - t0
    np.save(os.path.join(MODEL_DIR, 'esm2_embeddings.npy'), esm_emb)
    print(f'[esm] embedded {n_esm} peptides in {emb_t:.1f}s '
          f'({emb_t/n_esm*1000:.1f} ms/pep) -> {esm_emb.shape}', flush=True)

    esm_phys = np.stack([compute_phys(s) for s in esm_peps])
    esm_pca = pca[esm_idx]
    esm_rpes = rpes[esm_idx]

    def cv_r2(X, y):
        kf = KFold(3, shuffle=True, random_state=42)
        vals = [HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=42)
                .fit(X[tr], y[tr]).score(X[te], y[te]) for tr, te in kf.split(X)]
        return float(np.mean(vals)), float(np.std(vals))

    results = {'n_benchmark': n_esm, 'model': 'esm2_t6_8M_UR50D',
               'embedding_time_s': emb_t, 'model_load_s': load_t,
               'esm_embedding_dim': int(esm_emb.shape[1])}
    for name, X in [('phys_only', esm_phys),
                    ('pca_phys', np.hstack([esm_pca, esm_phys])),
                    ('esm2_phys', np.hstack([esm_emb, esm_phys]))]:
        r2, sd = cv_r2(X, esm_rpes)
        results[name] = {'cv_r2': r2, 'cv_std': sd}
        print(f'[cv] {name:12s} R2={r2:.4f} +/- {sd:.4f}', flush=True)

    results['total_s'] = time.time() - t_all
    with open(os.path.join(OUT_DIR, 'esm2_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('[done] wrote esm2_comparison.json', flush=True)

if __name__ == '__main__':
    main()
