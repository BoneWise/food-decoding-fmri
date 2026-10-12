# 主观食物价值的跨被试 fMRI 解码

**Deep Learning-Based Cross-Participant Decoding of Subjective Food Value**  
*论文准备中（Manuscript in preparation）*

全脑 fMRI 中与主观食物价值相关的信息能够泛化到未参与训练的被试。线性 SVM 和 3D-CNN 均能在按被试留出的交叉验证中，以高于机会水平的表现区分高、低即时食用意愿。

本仓库提供跨被试解码与空间分析的代码及固定结果。Layer-CAM 刻画 CNN 的空间归因，显示高、低价值预测均具有突出的枕叶和岛叶—皮层下区域侧重。GLM 和 Searchlight 分析分别考察平均反应差异与局部可解码信息。

[English](README.md) · [图片复现](docs/README.md) · [解码结果](docs/results.md)

## 主要结果

- **跨被试泛化：** 30 名被试采用十折交叉验证，每折留出三名被试；标签为即时食用意愿评分的最低端和最高端（1 和 4）。
- **预测表现：** 单试次输入下，SVM 和 CNN 的被试平均平衡准确率分别为 57.87% 和 53.71%；同一被试、同一评分的六试次平均输入下，分别为 63.33% 和 57.24%。两种模型在 k=1、3、6 三种设置下均高于机会水平。
- **空间归因：** 基于 k=6 CNN 的 Layer-CAM 显示，两类预测共有枕叶和岛叶—皮层下区域侧重，并在后内侧、内侧颞叶等区域呈现不同的相对侧重。

## 最快使用方式

在仓库目录中安装 Python 环境后运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/reproduce_results.py
python scripts/validate_reference.py
python scripts/reproduce_figures.py
```

这套入口直接使用随仓库提供的小型预测表和区域数据，重算论文主要表格并生成统计图。新结果位于 `outputs/`。任务图、模型结构图和脑表面图的原件保存在 `figures/original/`。

## 文件从哪里看

- `results/reference/`：固定的论文结果、预测与来源哈希。
- `configs/decoding/`：k=1/3/6 正式配置和原始参与者分折。
- `src/foodpref/`：CNN、SVM、数据标准化及试次平均代码。
- `analyses/`：GLM、Searchlight、Layer-CAM、被试内 session 留出及补充分析。
- `preprocessing/`：生成 beta 图的 MATLAB/SPM 入口。
- `docs/README.md`：结果图的复现命令、输出文件和原始图片。
- `docs/results.md`：解码任务、评价指标和结果数值。

区域量统一为“平均 Layer-CAM 值（全脑平均值 = 1）”，保留当前论文使用的群体图归一化顺序。统计单位以参与者为主，折均值另外标记。

完整训练和空间分析需要另行准备原始数据、beta 图和相关软件；大型模型权重及特征矩阵未随初版分发。论文写作文件继续保留在独立工作流中，后续可用固定提交或标签与论文版本对应。
