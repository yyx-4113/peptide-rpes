# peptide-rpes

**RPES — a rank-percentile ensemble score for ceiling-effect-free peptide bioactivity ranking, with a benchmark of PCA and protein-language-model representations for short food-derived peptides.**

This repository is the reproduction package for the B-pivot rewrite of the rejected manuscript *"Mechanism-Conditioned Latent-Space Optimization of Sleep-Promoting Peptides via Active Learning with Virtual Cell Validation."* The sleep-promoting / virtual-cell / docking claims were removed; what remains is a transparent scoring module (RPES) and an honest representation benchmark.

*Author:* Yongxin Yang, B.M. — The Second Affiliated Hospital of Fujian University of Traditional Chinese Medicine, Fuzhou, Fujian 350003, China. ORCID: 0009-0004-9698-6552.

---

## What this repository contains

| File | Purpose |
|------|---------|
| `b_pivot_analysis.py` | Held-out 80/20 split (phys-only vs PCA+phys predicting RPES) + real timing of the docking-free pipeline. Does **not** require torch/xgboost. |
| `b_pivot_pca_timing.py` | Pure-sklearn timing of one-hot build, PCA fit/transform. |
| `b_pivot_esm2.py` | ESM-2 head-to-head: descriptors vs PCA+descriptors vs ESM-2+descriptors predicting RPES (3-fold CV on a 5,000-peptide benchmark). Requires `fair-esm` (`pip install fair-esm`; the importable top-level package is `esm`). |
| `b_pivot_recompute_rpes.py` | **Deterministic** RPES recomputation (seed = 42) for all 403,461 peptides; backs up the pre-fix cache and syncs the headline JSONs. Requires torch + xgboost. |
| `b_pivot_pca_recon.py` | PCA reconstruction accuracy + decoder validity on the full library. Requires torch (via `phase1_pipeline`). |
| `b_pivot_matched_init_ablation.py` | **Matched-initialisation ablation (v0.5).** Re-runs the Layer 3 search and three no-loop controls (A′: matched init, 240 candidates; A″: matched init, 2,400 candidates; A: original random-latent procedure) with decoded candidates scored on the **library** percentile scale. Writes `output/ablation_matched_init.json`. Requires torch + xgboost; needs `data/training_peptides.csv` and `models/pca_latent.npz`. Runtime ≈ 3 min. |
| `rpes_full_experiment.py` | C3 three-layer comparison + ablation (A/B/C) + RPES top-20 peptide analysis. Requires torch + xgboost. |
| `generate_figures.py` | Regenerates the manuscript figure set (Fig 1 A/B/C composite, Fig 2, Fig S1 scree, Fig S2 latent projection) into `output/figures/`. Requires the large artefacts (peptide CSV + `pca_latent.npz`) and matplotlib. |
| `phase1_pipeline.py` | The original pipeline module (PCA latent space, physicochemical descriptors, RF/BiLSTM models). Included as a dependency; **imports torch at the top level.** |
| `new_methods.py` | `compute_rpes()` and the DTR code (DTR is unused in this revision; kept for audit). |
| `config.py` | **Portable path configuration.** All paths default to directories inside this repo; override with `PEPRPES_DATA_DIR`, `PEPRPES_MODEL_DIR`, `PEPRPES_OUT_DIR`. |
| `output/*.json` | All result JSONs (deterministic, seed = 42). |
| `output/rpes_scores.npz` | The authoritative deterministic RPES for 403,461 peptides (max = 0.8342). |
| `output/rpes_scores_pre_seedfix.npz` | Audit-trail backup of the non-deterministic pre-fix cache. |

### Large artefacts (provided via Zenodo, not in git)

- `data/merged_peptide_library.csv` — 451,785 raw / 403,461 unique food-derived peptides (walnut, mulberry, black sesame).
- `models/pca_latent.npz` — PCA latent for 403,461 peptides (~103 MB).
- `models/esm2_embeddings.npy` — ESM-2 embeddings for 5,000 benchmark peptides.
- `data/training_peptides.csv` — **373 labelled peptides (73 bioactive positives, 300 negatives)** used by `train_rf_classifier` to build the `rf_score` component of RPES. Small (3.2 kB) but **required**; it was inadvertently omitted from the v1.0.0 GitHub archive and is supplied with the manuscript's supplementary material.

Download these from the Zenodo deposit (see *Data availability* below) and place them in `data/` and `models/`, or set `PEPRPES_DATA_DIR` / `PEPRPES_MODEL_DIR` to point at your own copies. The scripts re-generate `pca_latent.npz` and `esm2_embeddings.npy` if they are absent; `training_peptides.csv` cannot be regenerated and must be supplied.

---

## Installation

```bash
# Python 3.13+ recommended.
pip install numpy pandas scipy scikit-learn torch fair-esm xgboost
```

The scripts were developed and verified with: Python 3.13.12, numpy 2.x, pandas 2.x, scikit-learn 1.x, scipy 1.x, torch 2.14.0 (CPU), fair-esm 2.0.0 (import name `esm`), xgboost 3.x.

---

## Reproduction

All numbers in the manuscript trace to deterministic recomputations (seed = 42).

```bash
# 1) Determinism fix + authoritative RPES (≈3–5 min on CPU; requires torch + xgboost)
python b_pivot_recompute_rpes.py

# 2) Held-out 80/20 + timing (torch/xgboost-free; needs the peptide CSV + rpes_scores.npz)
python b_pivot_analysis.py

# 3) PCA timing (torch-free)
python b_pivot_pca_timing.py

# 4) ESM-2 head-to-head (≈1 min on CPU; requires fair-esm)
python b_pivot_esm2.py

# 5) C3 three-layer + ablation + top-20 (requires torch + xgboost)
python rpes_full_experiment.py

# 6) PCA reconstruction accuracy (requires torch)
python b_pivot_pca_recon.py

# 7) Regenerate the manuscript figures into output/figures/ (requires matplotlib;
#    needs the peptide CSV + pca_latent.npz from the Zenodo deposit)
python generate_figures.py
```

If your data / models live elsewhere, set the environment variables:

```bash
export PEPRPES_DATA_DIR=/path/to/data
export PEPRPES_MODEL_DIR=/path/to/models
export PEPRPES_OUT_DIR=/path/to/output
```

---

## Key results (all deterministic, seed = 42)

- **RPES eliminates the weighted-average ceiling:** composite max = **0.8342** (mean 0.416, SD 0.142, p99 0.738) over 403,461 peptides; `no_ceiling = True`.
- **Representation benchmark (C2):** predicting RPES, 3-fold CV R² on 5,000 peptides — descriptors **0.936** > PCA+descriptors **0.916** > ESM-2+descriptors **0.907**; full-library held-out test R² descriptors **0.9581** vs PCA+descriptors **0.9581** (Δ < 0.0001). Hand-crafted descriptors suffice; PCA and a general PLM are redundant.
- **Latent-space search (C3):** reaches **98.1%** of the natural optimum (0.8180 vs 0.8342) and **+3.4%** over random latent sampling, but the iterative loop contributes **zero** improvement and never exceeds the best natural peptide → positioned as in-library rediscovery, not de novo invention.
- **Compute budget:** scoring a 403,461-peptide library ≈ **1 s** once the one-hot matrix and PCA transform are cached (docking removed).

---

## Honesty gates (read before citing)

1. RPES's six components are, directly or indirectly, functions of the ten physicochemical descriptors; RPES is therefore a near-deterministic function of those descriptors (a surrogate predicts it with R² ≈ 0.95–0.96). RPES is a transparent *ceiling-free aggregation module*, not a signal-discovering model.
2. The original 76.7% PCA reconstruction-accuracy figure could not be reproduced; the full-library value is **61.7%** (corrected).
3. The iterative search loop adds no gain; the "+3.4% over random" is an *initialisation* effect, not an optimisation effect.
4. No biological-activity claims are made; no wet-lab validation is presented (out of scope for a methods/resource paper).

---

## Data availability

The peptide library and large cached arrays are archived on Zenodo (DOI to be assigned at submission). The code, scripts, small JSONs, and `rpes_scores.npz` are released under an MIT licence at `https://github.com/yyx-4113/peptide-rpes` (to be made public upon acceptance). No transcriptomic (GEO) data are used in this manuscript.

## Citation

See `CITATION.cff`. Suggested BibTeX:

```bibtex
@software{yang2026peprpes,
  author = {Yang, Yongxin},
  title = {peptide-rpes: RPES scoring and peptide-representation benchmark},
  year = {2026},
  license = {MIT},
  url = {https://github.com/yyx-4113/peptide-rpes}
}
```

## Licence

MIT — see `LICENSE`.
