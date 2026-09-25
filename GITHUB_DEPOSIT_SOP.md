# GitHub 发布与 Zenodo 归档操作手册（peptide-rpes）

> 本文件为中文操作手册，配合 `README.md` / `CITATION.cff` 使用。目标：把本地仓库 `D:\projects\peptide-rpes` 发布为公开 GitHub 仓库 `yyx-4113/peptide-rpes` 并打版本 tag，随后在 Zenodo 归档大文件（肽库 CSV、PCA latent、ESM-2 嵌入）。

---

## 一、发布前检查（本地）

1. 确认仓库根目录含以下**已就位且为最新**的文件：
   - `README.md`、`CITATION.cff`、`LICENSE`（MIT）、`.gitignore`
   - `config.py` + 六个 `b_pivot_*.py` + `rpes_full_experiment.py` + `phase1_pipeline.py` + `new_methods.py`
   - `output/*.json`、`output/rpes_scores.npz`、`output/rpes_scores_pre_seedfix.npz`
2. 确认 `.gitignore` 已排除 `data/`、`models/`、所有 `*.npz` / `*.npy`（这些走 Zenodo，不进 git）。
3. 在 PowerShell / Git Bash 进入仓库：
   ```bash
   cd D:\projects\peptide-rpes
   git status        # 确认没有大文件被纳入
   ```

---

## 二、本地提交

```bash
git init            # 若尚未初始化
git add -A
git commit -m "v1.0.0: peptide-rpes reproduction package (RPES + representation benchmark)"
```

> 提交信息统一用 `vX.Y.Z` 语义。本仓库首个发布版为 `v1.0.0`。

---

## 三、推送到 GitHub

> ⚠️ 沙箱限制：当前 WorkBuddy 沙箱对 git 的 https 传输硬拦截（`remote helper 'https' aborted session`），且 `github.com` 主页 HTTPS 亦被 reset（仅 `api.github.com` 放行），**无法在沙箱内完成 push**。请在**你本机**（已开网络/代理、已配置 git 凭据）执行下方命令。本地已配好 `origin` 远程与 `v1.0.0` 标签。

1. 在 github.com 新建仓库 **`yyx-4113/peptide-rpes`**（公开仓库，不要勾选自动生成 README/LICENSE——本地已有）。
2. 关联远程并推送（本机 git 可能走 `127.0.0.1` 代理；若 `Failed to connect to github.com port 443`，先确认 Clash 等代理已开，或清代理直连）：
   ```bash
   cd D:\projects\peptide-rpes
   git remote add origin https://github.com/yyx-4113/peptide-rpes.git   # 若已存在用 set-url
   git push -u origin master --tags
   ```
3. 免交互令牌推送（将 `YOUR_TOKEN` 替换为你的 GitHub Personal Access Token，需 `repo` 权限）：
   ```bash
   git -C D:\projects\peptide-rpes push https://YOUR_TOKEN@github.com/yyx-4113/peptide-rpes.git master --tags
   ```
4. 推送 `v1.0.0` 标签后会触发 `.github/workflows/release.yml` 自动创建 GitHub Release（版本锚点；大文件走 Zenodo，见第四节）。

---

## 四、Zenodo 归档

### 4.1 代码仓库自动归档（GitHub 集成，推荐）
Zenodo 提供 GitHub 集成：在你 Zenodo 账户开启 GitHub 应用并授权后，对 `v1.0.0` 标签发 GitHub Release，Zenodo 会自动抓取 GitHub 仓库（脚本 + 小 JSON + `.zenodo.json` 元数据）生成一个软件归档版本并分配 DOI。无需手动上传代码。

### 4.2 大文件单独归档（手动 deposit）
GitHub 不适合放 >100 MB 的输入文件。在 Zenodo 新建 deposit，上传：
- `data/merged_peptide_library.csv`（451,785 行；可去重后 403,461 行）
- `models/pca_latent.npz`
- `models/esm2_embeddings.npy`

在 deposit 元数据里填写作者（Yongxin Yang, ORCID 0009-0004-9698-6552）、许可证（MIT）、与 GitHub 仓库的关系（"is supplemented by"），并引用 `.zenodo.json` 中的同名元数据以保持一致。发布后拿到 Zenodo DOI（如 `10.5281/zenodo.XXXXXXX`）。

### 4.3 回填
把该 DOI 回填到 `README.md` 的 "Data availability" 段与 `CITATION.cff` 的 `url`/补充字段，并同步到稿件数据可用性声明。

---

## 五、稿件数据可用性声明（回填模板）

> "The food-derived peptide library and large cached embeddings are archived on Zenodo (DOI: 10.5281/zenodo.XXXXXXX). The code, analysis scripts, and deterministic RPES scores are released under an MIT licence at https://github.com/yyx-4113/peptide-rpes (v1.0.0)."

---

## 六、事后核查清单

- [ ] GitHub 仓库公开可访问，含全部脚本 + 小 JSON + `rpes_scores.npz`
- [ ] `v1.0.0` tag 已推送，GitHub Release 已生成
- [ ] Zenodo deposit 已发布，DOI 已回填 README / CITATION.cff / 稿件
- [ ] 稿件数据可用性声明用真实 URL/DOI，无 "available on request"
