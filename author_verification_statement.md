# Author Verification Statement — peptide-rpes

**Manuscript (B-pivot):** *RPES: a rank-percentile ensemble score for ceiling-effect-free peptide bioactivity ranking, with a benchmark of PCA and protein-language-model representations for short food-derived peptides.*

1. **Authorship.** I, Yongxin Yang, am the sole author of this work and of the accompanying code repository `peptide-rpes`. I performed all conceptual design, implementation, data analysis, and manuscript writing. There is no undisclosed ghost, guest, or fabricated authorship. My academic title is **B.M. (Bachelor of Medicine)**; I do not hold an MD, PhD, or other postgraduate research degree, and I have not represented myself as holding one.

2. **Originality and integrity of the code.** The repository contains my own analysis scripts (`b_pivot_*.py`), the original pipeline module (`phase1_pipeline.py`), and `new_methods.py`. All results reported in the manuscript are produced by these scripts from the stated public/derived data; no result was fabricated, rounded to mislead, or copied from an irreproducible source. Every numerical claim in the manuscript is traceable to a file in `output/` (see `supplementary_tables_bpivot.md` and `B_pivot_runs_2026-09-25.md`).

3. **Reproducibility.** All reported figures are deterministic under seed = 42. The RPES distribution was previously non-deterministic (cache max 0.8148 vs headline 0.8386) due to unseeded sampling; both random sources were seeded and the score was recomputed to a single authoritative value (max = 0.8342). The pipeline is re-runnable end-to-end via the commands in `README.md`.

4. **Honesty gates (acknowledged, not concealed).**
   - RPES's components are all derived from physicochemical descriptors; RPES is a near-deterministic function of those descriptors (surrogate R² ≈ 0.95–0.96). It is positioned as a transparent aggregation module, not a signal-discovering model.
   - The original 76.7% PCA reconstruction-accuracy figure could not be reproduced; the corrected full-library value is 61.7%.
   - The latent-space search's iterative loop contributes zero improvement and never exceeds the best natural peptide; the "+3.4% over random" is an initialisation effect, not an optimisation effect. These facts are stated plainly in the manuscript.
   - No biological-activity claims and no wet-lab validation are presented; this is a methods/resource paper and biological validation is explicitly out of scope.

5. **Use of generative AI.** Generative AI tools were used solely as writing/editorial assistants for drafting and formatting; they were not used to generate data, references, figures, or authorship, and no AI-generated content is presented as my own experimental result.

6. **Data availability.** The peptide library and large cached arrays are archived on Zenodo; the code, scripts, and deterministic RPES scores are released under an MIT licence (no "available on request"). No transcriptomic GEO data are used.

*Signed: Yongxin Yang, B.M. — 2026-09-25.*
