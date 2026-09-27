"""T2-9 sensitivity sweep for S3.6 matched enrichment.

Re-runs the matched-background enrichment + permutation test at alternative
bandwidths (length window 9-10 vs 8-10; hydrophobicity +/- 0.5/1/2 SD of the
top-20 mean) to confirm that the headline conclusion (His/Gly/Tyr/Pro survive
20-residue BH; Trp/Ala do not) is robust to the bandwidth choice.
"""
import os, json, time, numpy as np
from collections import Counter

OUT = r"D:\projects\peptide-rpes\output"
LIB = r"D:\projects\peptide-rpes\data\merged_peptide_library.csv"
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
FOCAL = ['H', 'G', 'Y', 'P', 'W', 'A']
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
      'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
      'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

t0 = time.time()
rpes = np.load(os.path.join(OUT, 'rpes_scores.npz'))['rpes']
import pandas as pd
df = pd.read_csv(LIB).drop_duplicates(subset=['peptide']).reset_index(drop=True)
peptides = df['peptide'].astype(str).tolist()
lib_n = len(peptides)
assert lib_n == len(rpes)
kd_arr = np.array([KD.get(chr(c), 0.0) for c in range(256)])
def hydro_of(s):
    return float(kd_arr[np.frombuffer(s.encode('ascii'), dtype=np.uint8)].mean())

top20 = json.load(open(os.path.join(OUT, 'rpes_full_results.json')))['peptide_analysis']['top20_peptides']
top20_hydro = np.array([hydro_of(s) for s in top20])
center = float(top20_hydro.mean()); sd = float(top20_hydro.std(ddof=0))
print("top20 hydro mean=%.4f sd=%.4f" % (center, sd), flush=True)

bg_hydro = np.array([hydro_of(s) for s in peptides])
top_counts = Counter(''.join(top20)); total_top = sum(top_counts[a] for a in AA_LIST)
obs_top_frac20 = np.array([top_counts.get(a,0)/total_top for a in AA_LIST])

def focal_matrix(seqs, aas):
    F = np.zeros((len(seqs), len(aas))); L = np.zeros(len(seqs))
    for i,s in enumerate(seqs):
        c = Counter(s); L[i]=len(s)
        for j,a in enumerate(aas): F[i,j]=c.get(a,0)
    return F, L

results = {}
N_PERM = 100000
for lw in [(9,10),(8,10)]:
    for sm in [0.5,1.0,2.0]:
        mask = np.array([(len(s) in lw) and (abs(bg_hydro[i]-center)<=sm*sd) for i,s in enumerate(peptides)])
        matched = [peptides[i] for i in np.where(mask)[0]]
        Fm20, Lm20 = focal_matrix(matched, AA_LIST)
        mb_frac20 = Fm20.sum(0)/Lm20.sum()
        obs_enrich20 = obs_top_frac20/mb_frac20
        n_matched = len(matched)
        rng = np.random.default_rng(42)
        perm_ge = np.zeros(len(AA_LIST))
        CHUNK=20000
        for st in range(0,N_PERM,CHUNK):
            n=min(CHUNK,N_PERM-st)
            idx=rng.integers(0,n_matched,size=(n,20))
            sc=Fm20[idx].sum(1); sl=Lm20[idx].sum(1)
            sf=sc/sl[:,None]/mb_frac20[None,:]
            perm_ge += (sf>=obs_enrich20[None,:]).sum(0)
        perm_p20 = perm_ge/N_PERM
        # BH over 20
        order=np.argsort(perm_p20); m=20; q=np.empty(m); prev=1.0
        for rank,i in enumerate(reversed(order)):
            r=m-rank; val=perm_p20[i]*m/r; prev=min(prev,val); q[i]=prev
        q20=np.clip(q,0,1)
        fid=[AA_LIST.index(a) for a in FOCAL]
        fp={a:(round(perm_p20[AA_LIST.index(a)],5), round(q20[AA_LIST.index(a)],5)) for a in FOCAL}
        key="len_%d-%d_sd_%.1f" % (lw[0],lw[1],sm)
        results[key]={'matched_n':int(n_matched),'obs_enrich':{a:round(float(obs_enrich20[AA_LIST.index(a)]),3) for a in FOCAL},
                      'focal_p_q':fp}
        print(key, 'n=',n_matched, fp, flush=True)

print(json.dumps(results, indent=2))
json.dump(results, open(os.path.join(OUT,'s36_sensitivity.json'),'w'), indent=2)
print("Done in %.1fs" % (time.time()-t0))
