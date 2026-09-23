# 食物偏好的 fMRI 解码与脑区分析

本仓库对应论文：**Decoding Food Preferences from fMRI with Deep Learning: What Drives Prediction Across Individuals?**

主要内容是跨参与者偏好解码，以及 CNN 判断所使用的脑区分布。包含 CNN、匹配的线性 SVM、GLM、Searchlight、k=6 Layer-CAM、脑区平均值和既有补充分析。

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
- `docs/results.md`：各个数值和结果文件的关系。
- `docs/analysis-entrypoints.md`：完整分析的运行顺序。

区域量统一为“平均 Layer-CAM 值（全脑平均值 = 1）”，保留当前论文使用的群体图归一化顺序。统计单位以参与者为主，折均值另外标记。

完整训练和空间分析需要另行准备原始数据、beta 图和相关软件；大型模型权重及特征矩阵未随初版分发。论文写作文件继续保留在独立工作流中，后续可用固定提交或标签与论文版本对应。
