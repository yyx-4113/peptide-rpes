# peptide-rpes v1.1.2 — Artifact MANIFEST

Reproducibility manifest for peptide-rpes release **v1.1.2** (aligned with bpivot manuscript **v0.9.2**).
Checksums are SHA-256 of the listed files as committed under tag `v1.1.2`.

| Artifact | Role | SHA-256 |
|----------|------|---------|
| `compute_s36_permutation.py` | §3.6 permutation/FDR (ratio statistic, T1-1 fix) | `526931a65fae8c3b510ad3097ee6debe602ce505d0f3699f58030a414ddbcb11` |
| `phase1_pipeline.py` | main pipeline; `train_rf_classifier` seeded (T2-14) | `e53b6aa099f60c336a43115d3cef93d32b8a8698584f36c0a70fc025eb5bd54d` |
| `output/s36_permutation.json` | **authoritative** §3.6 enrichment + 20-residue BH | `bfeac1a9d69726d0023ec0e8d7f3c7a75cbc34b570885abcd745145162a8a691` |
| `output/s36_sensitivity.json` | §3.6 bandwidth sensitivity sweep (T2-9) | `81b51eea4ad327a03dba3116a73b4c851aeda7b0363cd0782ee126daa9357096` |
| `output/_s36_sensitivity.py` | regenerates `s36_sensitivity.json` | `21a2b705c223e7610dd4de6d9f762d100476d305a83472645b865520256b8f67` |
| `data/training_peptides.csv` | labelled RF training set (73 positives + 300 legacy negatives) | `db4e20dec3dc686497db8ddf436f9769791398e2fc383f6f28adf56ec8760ebd` |

## Non-authoritative / superseded
- `output/c3_s36_analysis.json` — retained for provenance only. It is **internally inconsistent**
  (records both n = 5,945 / 451,785 and n = 5,592 / 403,461) and is **superseded** by
  `output/s36_permutation.json` for the §3.6 enrichment / FDR claim. **Do not cite it for §3.6.**

## Verification
```bash
sha256sum compute_s36_permutation.py phase1_pipeline.py \
  output/s36_permutation.json output/s36_sensitivity.json \
  output/_s36_sensitivity.py data/training_peptides.csv
```
