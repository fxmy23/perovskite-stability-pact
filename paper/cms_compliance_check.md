# CMS 投稿合规性逐项检查 (对照官方规范全文)

**检查日期**: 2026-06-17
**对照源**: CMS官方Guide for Authors全文 + Elsevier政策

---

## ✅ 已合规 (无需修改)

| 要求 | 我们的状况 | 合规 |
|---|---|---|
| Article type | Full Length Article | ✅ |
| 单作者声明 | Xumingyong Feng, 单作者 | ✅ |
| Abstract ≤250词 | 221词 | ✅ |
| Keywords 1-7个 | 6个 | ✅ |
| Highlights (3-5条, ≤85字符) | 5条, 全部≤85字符, 单独文件 | ✅ |
| Graphical abstract (531×1328px) | 已生成, 单独PDF文件 | ✅ |
| 章节编号 (1, 1.1...) | 有 | ✅ |
| 参考文献 [1]格式 | 有, 40条含DOI | ✅ |
| 单栏布局 (Word) | .docx单栏 | ✅ |
| Times New Roman 12pt双倍行距 | .docx已设 | ✅ |
| 表格可编辑文本+无竖线 | 有 | ✅ |
| 图单独文件+编号+caption | 7张PDF | ✅ |
| 代码公开 (GitHub) | github.com/fxmy23/perovskite-stability-pact | ✅ |
| 数据可访问 (matminer) | wolverton_oxides via matminer | ✅ |
| CRediT作者贡献 | 已写 | ✅ |
| Funding声明 | 已写"无基金" | ✅ |

## ⚠️ 需要修改 (3项)

### 修改1: AI使用声明 — 必须用Elsevier精确格式 ★★★
**规范要求** (原文):
> Title of new section: **Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.**
> Statement: During the preparation of this work the author(s) used [NAME OF TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.

**我们的问题**: 当前标题和措辞不匹配Elsevier精确格式。
**修复**: 改为精确格式 (见下方manuscript更新)。

### 修改2: Declaration of Competing Interests — 必须单独Word文件 ★★★
**规范要求** (原文):
> "The resulting Word document containing your declaration should be uploaded at the 'attach/upload files' step in the submission process. It is important that the Word document is saved in the .doc/.docx file format."

**我们的问题**: 声明在manuscript内, 没有单独的.docx文件。
**修复**: 生成单独的 competing_interests.docx。

### 修改3: Data Availability — Option C精确措辞 ★★
**规范要求** (原文):
> "For this journal, Option C instructions apply. This means that you are required to: Deposit your research data in a relevant data repository. Cite and link to this dataset in your article. If this is not possible, make a statement explaining why research data cannot be shared."

**我们的状况**: GitHub有代码+数据, matminer有原始数据。但Zenodo DOI还没生成。
**处理**: 投稿时用GitHub链接 + 注明"Zenodo DOI will be provided upon acceptance"。Option C允许"make a statement"解释。
