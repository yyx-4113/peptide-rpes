# peptide-rpes v1.1.2 — Release Notes

> 对应稿件版本：**bpivot 手稿 v0.9.2**（PeerJ, Methods/Resource）。
> 本文件随 v1.1.2 tag 一并提交，供 GitHub Release 页面覆盖自动生成的英文 commit 列表。

## 相对 v1.1.1 的改动（代码层）

1. **`compute_s36_permutation.py` — 修正 §3.6 置换检验统计量（T1-1）。**
   旧版把"原始残基分数"直接对"富集比值"比较，导致每个残基都返回 p ≈ 0。
   新版比较**富集比值** `sf_ratio = sc / sl[:, None] / matched_bg_frac20[None, :]`
   与观测比值 `obs_enrich`，对全部 20 个标准氨基酸做标注置换，并执行 20 检验
   Benjamini–Hochberg 多重校正。已重新生成 `output/s36_permutation.json`。
   - 校正结论：His / Gly / Tyr / Pro 在多重校正后**仍显著**（BH-q ≤ 0.0094）；
     Trp（q = 0.594）与 Ala（q = 0.065）在 α = 0.05 下**不显著**。
   - 该修正确正了 v1.1.1 / 手稿 v0.9.1 在 §3.6 的"全部 p < 0.0001、q < 0.0001"多重校正过度声明。

2. **`phase1_pipeline.py`（`train_rf_classifier`）— 固定 rf 负样本种子（T2-14）。**
   在采样 200 个库来源阴性肽之前新增 `random.seed(42); np.random.seed(42)`，
   使 rf_score 训练在 seed = 42 下完全可复现（与 RPES 其余产物一致）。

3. **新增 `output/_s36_sensitivity.py` + `output/s36_sensitivity.json` — 带宽敏感性证据（T2-9）。**
   扫描长度窗 8–10 / 9–10 × 疏水性 ±0.5 / 1 / 2 SD（n = 2,776–11,875），
   重算匹配富集与 20 残基 BH。结论稳健：H/G/Y/P 在每一带宽下均显著，W/A 均不显著。

## 重要警示（数据可用性声明引用）

- `output/c3_s36_analysis.json` **保留但标记为内部不一致**（同时记录 n = 5,945 / 451,785
  与 n = 5,592 / 403,461），已被 `output/s36_permutation.json` 取代；**§3.6 富集 / FDR 勿引用 c3**。
- 稿件 §3.6 以严格去重库（403,461）/ n = 5,592 为权威来源（`s36_permutation.json`）。
- `README.md` 同步校正：第 6 分量常数 0.50000124、表示学习基准 tie 实差 Δ = 3.6 × 10⁻⁵。

## 产物 MANIFEST（sha256，v1.1.2）

| 文件 | sha256 |
|------|--------|
| `compute_s36_permutation.py` | `526931a65fae8c3b510ad3097ee6debe602ce505d0f3699f58030a414ddbcb11` |
| `phase1_pipeline.py` | `e53b6aa099f60c336a43115d3cef93d32b8a8698584f36c0a70fc025eb5bd54d` |
| `output/s36_permutation.json` | `bfeac1a9d69726d0023ec0e8d7f3c7a75cbc34b570885abcd745145162a8a691` |
| `output/s36_sensitivity.json` | `81b51eea4ad327a03dba3116a73b4c851aeda7b0363cd0782ee126daa9357096` |
| `output/_s36_sensitivity.py` | `21a2b705c223e7610dd4de6d9f762d100476d305a83472645b865520256b8f67` |
| `data/training_peptides.csv` | `db4e20dec3dc686497db8ddf436f9769791398e2fc383f6f28adf56ec8760ebd` |

## 发布状态

- **v1.1.2 tag 已发布（2026-09-27）**，仓库 `yyx-4113/peptide-rpes` 为 **Public**。
- PeerJ 投稿（传 v0.9.2 双 DOCX + 4 PNG + 补充包）仍由作者在主机执行。
