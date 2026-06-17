# Computational Materials Science (Elsevier) — 投稿规范完整清单

**来源**: [CMS Guide for Authors](https://www.sciencedirect.com/journal/computational-materials-science/publish/guide-for-authors) (官方全文) + [Elsevier Artwork Sizing](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing) + [Elsevier Digital Art Guidelines](https://www.elsevier.com/__data/promis_misc/JBCDigitalArtGuidelines.pdf)
**整理日期**: 2026-06-17

---

## ★★★ 最关键: FAIR 数据+代码强制要求

CMS 对数据驱动/ML论文有**特殊强制要求**(从Guide原文):
> "To be considered for publication in Computational Materials Science studies proposing or applying data-driven techniques must exhibit **a high degree of novelty in application and interpretation**, in addition to **providing FAIR-compatible data and code** to support their analysis."
> "Papers that are deemed to be primarily methodological but do not provide FAIR data and code will be **returned without review**."

**我们必须做到**:
1. 代码公开(GitHub, MIT license)
2. 数据可访问(wolverton_oxides via matminer + 处理后特征存Zenodo/Mendeley Data)
3. 论文中有Data Availability Statement
4. **高新颖性**: 不能只是"用标准ML跑标准数据集" → 我们的CQR/AD/LOEO/SR是差异化

**数据政策**: Option C — 必须 (1)存数据到仓库 (2)在论文引用链接 (3)不行则说明原因。

---

## 一、稿件文件格式

| 要求 | 规范 |
|---|---|
| 文件类型 | **可编辑源文件** (.docx 或 .tex); PDF不可作源文件 |
| 布局 | Word: **单栏**; LaTeX: 可双栏 |
| 文字 | 移除删除线/下划线(除非有科学意义); 拼写检查 |
| 语言 | 美式或英式英语(不混用) |
| 推荐 | 阅读 Elsevier Step-by-step guide |

## 二、稿件结构 (按顺序)

### 2.1 Title page (必须)
- **标题**: 简洁信息丰富, 避免缩写/公式(除非广泛认知如DNA)
- **作者**: 名+姓, 顺序与投稿系统一致, 拼写准确
- **单位**: 上标小写字母标记, 含完整地址+国家+邮箱
- **通讯作者**: 明确标注, 邮箱保持更新
- **现/永久地址**: 如有变动用脚注(上标阿拉伯数字)

### 2.2 Abstract (必须, ≤250词)
- 简洁事实性, ≤250词
- 独立可读(常单独展示)
- **避免参考文献**(必要的标作者+年份)
- 避免非标准缩写(必须用则首次定义)

### 2.3 Keywords (必须, 1-7个)
- 英文
- 避免多词组(用"and"/"of")
- 缩写仅限领域确立的

### 2.4 Highlights (必须提交!)
- 单独文件, 文件名含"highlights"
- **3-5个要点, 每个≤85字符(含空格)**
- 捕获新颖结果+新方法
- 提升搜索引擎可发现性

### 2.5 Graphical Abstract (必须提交!)
- 单独文件
- 图像 **531×1328像素(h×w)** 或成比例更大
- 在 **5×13cm** 尺寸可读
- 格式: TIFF/EPS/PDF/MS Office
- 不得用生成式AI制作(Elsevier GenAI政策)

### 2.6 正文分节 (numbered)
- 编号章节 1, 1.1, 1.1.1, 1.2...
- 交叉引用用编号(不只"the text")
- 小节可有简短标题(单独一行)
- **摘要不计入章节编号**

### 2.7 Math formulae
- 可编辑文本(非图片)
- 简单公式行内, 变量斜体
- 用 / 代替分数线(小分式)
- exp 表示 e的幂
- 显示公式单独编号(顺序)

### 2.8 References
- 方括号编号 [1], [2], 顺序按文中出现
- 文中可提作者名但必须给编号: "as shown [3,6]. Smith [8] found..."
- 提交时格式不严格(一致即可), 录用后proof阶段统一
- 必含: 作者名, 期刊/书名, 标题, 年份, 卷, 页/文章号
- 推荐DOI
- 期刊名按LTWA缩写
- 示例: [1] J. van der Geer, T. Handgraaf, R.A. Lupton, The art of writing a scientific article, J. Sci. Commun. 163 (2020) 51–59. https://doi.org/10.1016/j.sc.2020.00372.

### 2.9 标准声明(按顺序, 参考文献前)
- **Acknowledgements** (单独节, 参考文献前, 不在title page)
- **Author contributions (CRediT)**: 14角色(Conceptualization/Data curation/.../Writing-review&editing), 对应作者
- **Declaration of competing interests**: 选"I have nothing to declare"或列出
- **Funding sources**: 标准格式, 无则声明
- **Declaration of generative AI use**: 若用了AI工具必须声明(单独节, 参考文献前); 仅语法检查工具不用声明

## 三、Tables 规范

| 要求 | 规范 |
|---|---|
| 格式 | **可编辑文本(非图片)** |
| 位置 | 相关文本旁或文末单独页 |
| 引用 | 文中必须引用所有表 |
| 编号 | 按出现顺序连续 |
| 标题 | 表必须有caption |
| 表注 | 表体下方 |
| ★ 线条 | **避免竖线和单元格底纹**(三线表) |
| 节制 | 表少用, 不与正文重复 |

## 四、Figures/Artwork 规范 ★★★(画图核心)

### 4.1 通用要求
- **单独文件**, 逻辑命名(Figure_1, Figure_2...)
- 文中必须引用所有图
- 按出现顺序编号
- 每图**必须有caption**(标题+描述, 放图下方, "Fig. 1."开头)
- 图内文字最少化, 符号/缩写需在caption解释
- **不提交**: 分辨率过低的文件; 图像与字体比例失衡(文字不可读)

### 4.2 分辨率(按图类型, 印刷尺寸下) ★
| 图类型 | 最低分辨率 | 单栏像素 | 全页宽像素 |
|---|---|---|---|
| **彩色/灰度照片(halftone)** | **300 dpi** | 1063 px | 2244 px |
| **位图线条画(line art)** | **1000 dpi** | 3543 px | 7480 px |
| **组合(线+半色调)** | **500 dpi** | 1772 px | 3740 px |
| ★ 线条画首选 | **1000–1200 dpi** | — | — |

### 4.3 字体
| 元素 | 印刷后尺寸 |
|---|---|
| **正文文字** | **7 pt** |
| 下标/上标 | ≥6 pt |
| 推荐字体 | Arial, Helvetica, Times New Roman, Symbol |

### 4.4 文件格式
- **矢量图**: EPS 或 PDF (嵌入字体或文字存为graphics)
- **照片(halftone)**: TIFF/JPG/PNG, ≥300 dpi
- **位图线条画**: TIFF/JPG/PNG, ≥1000 dpi
- **组合图**: TIFF/JPG/PNG, ≥500 dpi
- **首选格式: TIFF**; EPS/PDF也可

### 4.5 颜色
- 在线彩色免费
- **必须色盲友好**(对色觉障碍者可访问) — 避免红绿对比, 用viridis/colorblind palette

### 4.6 尺寸
- 单栏宽: ~90mm (Elsevier单栏)
- 全页宽: ~190mm
- 图像不得 disproportionally 大于字体

## 五、我们的画图执行规范(基于CMS要求)

| 图 | 类型 | 分辨率 | 字体 | 格式 |
|---|---|---|---|---|
| F1 工作流图 | 线条画/组合 | 1000 dpi(矢量PDF优先) | Arial 7pt | PDF/EPS |
| F2 Parity | 组合(散点+线) | 500 dpi | Arial 7pt | PDF/TIFF |
| F3 Reliability | 组合 | 500 dpi | Arial 7pt | PDF/TIFF |
| F4 AD可视化 | 组合(散点) | 500 dpi | Arial 7pt | PDF/TIFF |
| F5 LOEO | 组合(热力/条形) | 500 dpi | Arial 7pt | PDF/TIFF |
| F6 SHAP-SR | 组合(条形) | 500 dpi | Arial 7pt | PDF/TIFF |

**统一设置**:
- matplotlib + seaborn/SciencePlots
- 字体: Arial (matplotlib 'font.family':'sans-serif', 'font.sans-serif':['Arial'])
- 字号: 坐标轴标签/图例 7pt(印刷), 刻度 6-7pt
- 线宽: 数据线 ≥1.0pt (印刷), 参考线 0.5-1.0pt 虚线
- 配色: viridis/colorblind palette
- 保存: `savefig(dpi=500, bbox_inches='tight')` → PDF(矢量) + PNG(预览)
- 坐标轴必须有标签+单位

## 六、必须提交的文件清单(CMS)
1. Manuscript (.docx 或 .tex) — 含正文+tables+figure captions
2. **Highlights** (单独文件, 3-5要点每≤85字符)
3. **Graphical Abstract** (531×1328px, 单独文件)
4. Figures (每个单独文件, Figure_1.pdf...)
5. Declaration of competing interests (Word doc)
6. (若用AI) Declaration of generative AI use
7. Funding statement (若无则声明)

## 七、对我们的特殊提示(避坑)
1. **FAIR数据+代码强制**: 不公开代码会被无审稿退回 → 必须GitHub+Zenodo
2. **高新颖性要求**: "用标准ML跑标准数据"会被退 → 强调CQR/AD/LOEO/SR差异化
3. **Highlights必须**: 每要点≤85字符 — 我们的卖点要压缩成5条短句
4. **Graphical abstract必须**: 单独画一张531×1328px的工作流概览
5. **CRediT必须**: 14角色对应(本科生: Software/Visualization/Formal analysis/Writing-original)
6. **AI使用声明**: 我们用了AI辅助 → 必须诚实声明(单独节, 参考文献前)

Sources:
- [CMS Guide for Authors (官方全文)](https://www.sciencedirect.com/journal/computational-materials-science/publish/guide-for-authors)
- [Elsevier Artwork Sizing](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing)
- [Elsevier Artwork Types](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-types)
- [Elsevier Digital Art Guidelines PDF](https://www.elsevier.com/__data/promis_misc/JBCDigitalArtGuidelines.pdf)
- [CMS Template (SciSpace)](https://scispace.com/formats/elsevier/computational-materials-science/0a200ca2df77aee6bc818b1501dff61a)
