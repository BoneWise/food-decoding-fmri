from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import (
    CODE_ROOT, CONFIG_ROOT,
    PROJECT_ROOT,
    condition_order,
    load_config,
    read_json,
    sha256_file,
    stable_sample_id,
    verify_frozen_static_dependencies,
    write_csv,
    write_json,
)


DISPLAY_NAMES = {
    "full": "Full input",
    "cam_top20": (
        "Training-fold class-balanced joint-CAM top-20% global-scalar replacement"
    ),
    "random_top20_mean": "CAM-top-20%-count-matched random global-scalar replacement",
    "value": "Atlas-defined value-related ROI global-scalar replacement",
    "visual": "Atlas-defined occipital visual ROI global-scalar replacement",
    "union": "Atlas-defined value-related–occipital-visual union global-scalar replacement",
}

CONTRAST_DISPLAY_NAMES = {
    "random_top20_minus_cam_top20": (
        "Random-replacement BA − joint-CAM-replacement BA"
    ),
    "full_minus_value": "Full-input BA − value-ROI-replacement BA",
    "full_minus_visual": "Full-input BA − visual-ROI-replacement BA",
    "full_minus_union": "Full-input BA − union-replacement BA",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the completed RQ3 global-scalar replacement experiment."
    )
    parser.add_argument("--config", default=None)
    return parser


def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    probability_threshold: float,
) -> dict[str, Any]:
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError(f"Invalid metric inputs: y={y.shape}, p={p.shape}")
    pred = (p >= float(probability_threshold)).astype(np.int64)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    recall_1 = float(tn / (tn + fp)) if (tn + fp) else math.nan
    recall_4 = float(tp / (tp + fn)) if (tp + fn) else math.nan
    balanced_accuracy = float(np.nanmean([recall_1, recall_4]))
    accuracy = float((tn + tp) / y.size)
    auc = (
        float(roc_auc_score(y, p))
        if np.unique(y).size == 2
        else math.nan
    )
    return {
        "n_samples": int(y.size),
        "n_rating1": int(np.sum(y == 0)),
        "n_rating4": int(np.sum(y == 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall_rating1": recall_1,
        "recall_rating4": recall_4,
        "roc_auc": auc,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    num_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        int(array.size),
        size=(int(num_bootstrap), int(array.size)),
        endpoint=False,
    )
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_one_sided(
    values: Sequence[float],
    *,
    num_sign_flips: int,
    seed: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Sign-flip inference received no finite values.")
    rng = np.random.default_rng(int(seed))
    observed = float(np.mean(array))
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float64),
        size=(int(num_sign_flips), int(array.size)),
        replace=True,
    )
    null_means = np.mean(signs * array[None, :], axis=1)
    p_value = float(
        (1 + np.sum(null_means >= observed)) / (int(num_sign_flips) + 1)
    )
    return {
        "n": int(array.size),
        "mean_effect": observed,
        "p_raw": p_value,
        "num_sign_flips": int(num_sign_flips),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    total = int(values.size)
    for rank, index in enumerate(order):
        candidate = float((total - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(values)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def summarize_fixed_predictions(
    fixed: pd.DataFrame,
    *,
    probability_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_rows: list[dict[str, Any]] = []
    for (subject_id, fold_index, condition), group in fixed.groupby(
        ["subject_id", "fold_index", "condition"],
        sort=True,
    ):
        metrics = binary_metrics(
            group["target"].to_numpy(),
            group["prob_high"].to_numpy(),
            probability_threshold=probability_threshold,
        )
        participant_rows.append(
            {
                "subject_id": str(subject_id),
                "fold_index": int(fold_index),
                "condition": str(condition),
                **metrics,
            }
        )

    fold_rows: list[dict[str, Any]] = []
    for (fold_index, condition), group in fixed.groupby(
        ["fold_index", "condition"],
        sort=True,
    ):
        metrics = binary_metrics(
            group["target"].to_numpy(),
            group["prob_high"].to_numpy(),
            probability_threshold=probability_threshold,
        )
        fold_rows.append(
            {
                "fold_index": int(fold_index),
                "condition": str(condition),
                **metrics,
            }
        )
    return pd.DataFrame(participant_rows), pd.DataFrame(fold_rows)


def read_random_metrics(
    fold_dirs: Sequence[Path],
    *,
    probability_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participant_replicate_rows: list[dict[str, Any]] = []
    fold_replicate_rows: list[dict[str, Any]] = []
    sample_mean_rows: list[dict[str, Any]] = []

    for fold_dir in fold_dirs:
        with np.load(
            fold_dir / "predictions" / "random_top20_predictions.npz",
            allow_pickle=False,
        ) as payload:
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
            targets = np.asarray(payload["targets"], dtype=np.int64)
            subjects = np.asarray(payload["subject_ids"]).astype(str)
            ratings = np.asarray(payload["ratings"], dtype=np.int64)
            trial_ids = np.asarray(payload["trial_ids"], dtype=np.int64)
            fold_index = int(np.asarray(payload["fold_index"]).reshape(-1)[0])

        for replicate in range(probabilities.shape[0]):
            fold_metrics = binary_metrics(
                targets,
                probabilities[replicate],
                probability_threshold=probability_threshold,
            )
            fold_replicate_rows.append(
                {
                    "fold_index": fold_index,
                    "random_replicate": int(replicate + 1),
                    **fold_metrics,
                }
            )
            for subject in sorted(set(subjects.tolist())):
                keep = subjects == subject
                metrics = binary_metrics(
                    targets[keep],
                    probabilities[replicate, keep],
                    probability_threshold=probability_threshold,
                )
                participant_replicate_rows.append(
                    {
                        "subject_id": str(subject),
                        "fold_index": fold_index,
                        "random_replicate": int(replicate + 1),
                        **metrics,
                    }
                )

        mean_probability = np.mean(probabilities, axis=0)
        for index in range(targets.size):
            sample_mean_rows.append(
                {
                    "sample_id": stable_sample_id(
                        str(subjects[index]),
                        int(ratings[index]),
                        int(trial_ids[index]),
                    ),
                    "fold_index": fold_index,
                    "subject_id": str(subjects[index]),
                    "trial_id": int(trial_ids[index]),
                    "rating": int(ratings[index]),
                    "target": int(targets[index]),
                    "condition": "random_top20_mean_probability",
                    "mean_prob_high_across_random_masks": float(
                        mean_probability[index]
                    ),
                }
            )

    return (
        pd.DataFrame(participant_replicate_rows),
        pd.DataFrame(fold_replicate_rows),
        pd.DataFrame(sample_mean_rows),
    )


def participant_random_means(random_participant: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n_samples",
        "n_rating1",
        "n_rating4",
        "accuracy",
        "balanced_accuracy",
        "recall_rating1",
        "recall_rating4",
        "roc_auc",
    ]
    rows: list[dict[str, Any]] = []
    for (subject_id, fold_index), group in random_participant.groupby(
        ["subject_id", "fold_index"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "subject_id": str(subject_id),
            "fold_index": int(fold_index),
            "condition": "random_top20_mean",
            "num_random_replicates": int(group["random_replicate"].nunique()),
        }
        for column in metric_columns:
            row[column] = float(group[column].mean())
        for column in ("tn", "fp", "fn", "tp"):
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_participant_effects(
    participant_conditions: pd.DataFrame,
) -> pd.DataFrame:
    metric_lookup = participant_conditions.pivot(
        index=["subject_id", "fold_index"],
        columns="condition",
        values=["balanced_accuracy", "roc_auc"],
    )
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("full_minus_cam_top20", "full", "cam_top20"),
        ("full_minus_random_top20", "full", "random_top20_mean"),
        ("random_top20_minus_cam_top20", "random_top20_mean", "cam_top20"),
        ("full_minus_value", "full", "value"),
        ("full_minus_visual", "full", "visual"),
        ("full_minus_union", "full", "union"),
    ]
    for (subject_id, fold_index), values in metric_lookup.iterrows():
        for comparison_id, reference, perturbed in comparisons:
            reference_ba = float(values[("balanced_accuracy", reference)])
            perturbed_ba = float(values[("balanced_accuracy", perturbed)])
            reference_auc = float(values[("roc_auc", reference)])
            perturbed_auc = float(values[("roc_auc", perturbed)])
            rows.append(
                {
                    "subject_id": str(subject_id),
                    "fold_index": int(fold_index),
                    "comparison_id": comparison_id,
                    "reference_condition": reference,
                    "perturbed_condition": perturbed,
                    "reference_ba": reference_ba,
                    "perturbed_ba": perturbed_ba,
                    "ba_effect": reference_ba - perturbed_ba,
                    "reference_auc": reference_auc,
                    "perturbed_auc": perturbed_auc,
                    "auc_effect": reference_auc - perturbed_auc,
                }
            )
    return pd.DataFrame(rows)


def group_condition_summary(
    participant_conditions: pd.DataFrame,
    *,
    num_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(condition_order()):
        group = participant_conditions.loc[
            participant_conditions["condition"] == condition
        ]
        if group.empty:
            raise ValueError(f"Missing participant metrics for {condition}.")
        ba = group["balanced_accuracy"].to_numpy(dtype=np.float64)
        auc = group["roc_auc"].to_numpy(dtype=np.float64)
        ba_ci = bootstrap_mean_ci(
            ba,
            num_bootstrap=num_bootstrap,
            seed=int(seed) + condition_index * 101,
        )
        auc_ci = bootstrap_mean_ci(
            auc,
            num_bootstrap=num_bootstrap,
            seed=int(seed) + condition_index * 101 + 1,
        )
        rows.append(
            {
                "condition": condition,
                "display_name": DISPLAY_NAMES[condition],
                "num_subjects": int(group["subject_id"].nunique()),
                "balanced_accuracy_mean": float(np.mean(ba)),
                "balanced_accuracy_sd_sample": float(np.std(ba, ddof=1)),
                "balanced_accuracy_ci_low": ba_ci[0],
                "balanced_accuracy_ci_high": ba_ci[1],
                "roc_auc_mean": float(np.nanmean(auc)),
                "roc_auc_sd_sample": float(np.nanstd(auc, ddof=1)),
                "roc_auc_ci_low": auc_ci[0],
                "roc_auc_ci_high": auc_ci[1],
            }
        )
    return pd.DataFrame(rows)


def primary_inference(
    effects: pd.DataFrame,
    *,
    primary_contrasts: Sequence[str],
    num_sign_flips: int,
    num_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contrast_index, contrast in enumerate(primary_contrasts):
        values = effects.loc[
            effects["comparison_id"] == contrast,
            "ba_effect",
        ].to_numpy(dtype=np.float64)
        inference = sign_flip_one_sided(
            values,
            num_sign_flips=num_sign_flips,
            seed=int(seed) + contrast_index * 1009,
        )
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            num_bootstrap=num_bootstrap,
            seed=int(seed) + contrast_index * 1009 + 1,
        )
        rows.append(
            {
                "comparison_id": contrast,
                "display_name": CONTRAST_DISPLAY_NAMES[contrast],
                "metric": "balanced_accuracy",
                "analysis_unit": "held_out_participant",
                "n": int(inference["n"]),
                "effect_direction": (
                    "positive means CAM replacement is more harmful than random"
                    if contrast == "random_top20_minus_cam_top20"
                    else "positive means replacement is worse than full input"
                ),
                "mean_effect": float(inference["mean_effect"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "test_name": "Monte_Carlo_participant_sign_flip_one_sided",
                "p_raw": float(inference["p_raw"]),
                "num_sign_flips": int(num_sign_flips),
                "num_bootstrap": int(num_bootstrap),
                "seed": int(seed) + contrast_index * 1009,
            }
        )
    p_adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, adjusted in zip(rows, p_adjusted.tolist()):
        row["p_holm"] = float(adjusted)
        row["significant_holm_0_05"] = bool(adjusted < 0.05)
        row["correction_family"] = "four_primary_balanced_accuracy_contrasts"
    return pd.DataFrame(rows)


def significance_text(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def make_figure(
    group_summary: pd.DataFrame,
    inference: pd.DataFrame,
    output_dir: Path,
) -> None:
    colors = {
        "full": "#3C5488",
        "cam_top20": "#E64B35",
        "random_top20_mean": "#8C8C8C",
        "value": "#00A087",
        "visual": "#F39B7F",
        "union": "#7E6148",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))

    ordered = group_summary.set_index("condition").loc[condition_order()].reset_index()
    y = np.arange(len(ordered))
    means = ordered["balanced_accuracy_mean"].to_numpy()
    lower = means - ordered["balanced_accuracy_ci_low"].to_numpy()
    upper = ordered["balanced_accuracy_ci_high"].to_numpy() - means
    axes[0].errorbar(
        means,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#222222",
        elinewidth=1.3,
        capsize=3,
        zorder=1,
    )
    axes[0].scatter(
        means,
        y,
        s=62,
        c=[colors[condition] for condition in ordered["condition"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    axes[0].axvline(0.5, color="#666666", linestyle="--", linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ordered["display_name"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Participant-level balanced accuracy")
    axes[0].set_title("A  Predictive performance")
    axes[0].grid(axis="x", color="#E6E6E6", linewidth=0.8)

    ordered_inference = inference.set_index("comparison_id").loc[
        [
            "random_top20_minus_cam_top20",
            "full_minus_value",
            "full_minus_visual",
            "full_minus_union",
        ]
    ].reset_index()
    y2 = np.arange(len(ordered_inference))
    effects = ordered_inference["mean_effect"].to_numpy()
    lower2 = effects - ordered_inference["ci_low"].to_numpy()
    upper2 = ordered_inference["ci_high"].to_numpy() - effects
    axes[1].errorbar(
        effects,
        y2,
        xerr=np.vstack([lower2, upper2]),
        fmt="o",
        color="#3C5488",
        ecolor="#222222",
        elinewidth=1.3,
        capsize=3,
        markersize=6,
    )
    axes[1].axvline(0.0, color="#666666", linestyle="--", linewidth=1)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(ordered_inference["display_name"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Balanced-accuracy effect")
    axes[1].set_title("B  Prespecified participant-level contrasts")
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.8)
    x_span = max(
        0.01,
        float(
            ordered_inference["ci_high"].max()
            - ordered_inference["ci_low"].min()
        ),
    )
    for index, row in ordered_inference.iterrows():
        axes[1].text(
            float(row["ci_high"]) + 0.025 * x_span,
            int(index),
            significance_text(float(row["p_holm"])),
            va="center",
            fontsize=9,
        )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "Global-scalar replacement in the fixed k=6 rating-conditioned grouped-input 3D-CNN",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "rq3_global_mean_replacement_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "rq3_global_mean_replacement_performance.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def format_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def format_percentage_points(value: float) -> str:
    return f"{100.0 * float(value):.2f}个百分点"


def format_percentage_points_en(value: float) -> str:
    return f"{100.0 * float(value):.2f} percentage points"


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_paper_summaries(
    *,
    group_summary: pd.DataFrame,
    inference: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    performance = group_summary.set_index("condition")
    tests = inference.set_index("comparison_id")
    n_random = int(config["random_control"]["num_replicates"])
    n_top = int(
        math.ceil(
            float(config["cam"]["top_fraction"])
            * int(config["expected"]["common_mask_voxels"])
        )
    )

    cn = f"""# RQ3 训练折全脑标量均值替代实验：方法与结果

## 方法

每个 $k=6$ 输入由同一参与者、同一真实评分类别内六个单试次 $\beta$ 图
平均构成。本分析复用这些条件层面输入上既有3D-CNN的10折参与者不重叠
验证模型和每折最终epoch checkpoint，不重新训练模型；每折为27名训练
参与者和3名留出参与者。

每折仅使用被模型正确分类的训练输入生成 Layer-CAM：先在参与者×评分
类别内平均，再对该类别至少有一个正确训练输入的贡献参与者等权平均；
rating 1 与 rating 4 的训练折类别图分别除以共同脑掩膜内均值后等权
合并。在固定的 {int(config['expected']['common_mask_voxels']):,} 体素共同
脑掩膜内，按 CAM 值确定性选择最高20%的 {n_top:,} 个体素。

每折另以固定种子从同一共同脑掩膜无放回抽取 {n_random} 个等体素数量的
随机掩膜，该随机条件只作为CAM前20%体素的等体素数量参照。预定义掩膜
来自 Harvard–Oxford cortical/subcortical 25%阈值、2 mm最大概率图谱，
经最近邻重采样至CNN输入网格后与共同脑掩膜相交。价值相关 ROI 包括双侧
Frontal Medial Cortex、双侧 Frontal Orbital Cortex 和 Left/Right
Accumbens；枕叶视觉 ROI 包括配置中九个双侧枕叶标签；联合 ROI 为二者
逻辑并集。九个视觉标签为 Lateral Occipital Cortex（superior/inferior
division）、Intracalcarine Cortex、Cuneal Cortex、Lingual Gyrus、
Temporal Occipital Fusiform Cortex、Occipital Fusiform Gyrus、
Supracalcarine Cortex 和 Occipital Pole。

每折从全部训练输入的实际模型输入空间中、在各输入自身参与者脑掩膜内
汇总所有体素，计算一个全脑单一均值标量。该折任一测试输入中所有被选中
的体素均替换为这个相同标量；不计算或使用逐空间位置均值图。替代后不
重新标准化，也不重新训练模型，并在该折全部测试输入上推断。

主要指标为每名留出参与者的 balanced accuracy。预设的四个比较为：
随机20%减 CAM 前20%、完整输入减价值 ROI、完整输入减视觉 ROI，以及
完整输入减联合 ROI。采用 {int(config['statistics']['num_sign_flips']):,}
次参与者级单侧符号翻转检验和 {int(config['statistics']['num_bootstrap']):,}
次参与者 bootstrap 95% CI，并对四项检验进行 Holm 校正。AUC 作为次要
描述性指标。

## 结果

- 完整输入：BA {format_percent(performance.loc['full', 'balanced_accuracy_mean'])}
  （95% CI {format_percent(performance.loc['full', 'balanced_accuracy_ci_low'])}–
  {format_percent(performance.loc['full', 'balanced_accuracy_ci_high'])}），
  AUC {format_percent(performance.loc['full', 'roc_auc_mean'])}。
- 训练折类别等权联合 CAM 前20%体素的全脑标量均值替代：BA
  {format_percent(performance.loc['cam_top20', 'balanced_accuracy_mean'])}；
  随机20%体素替代的100次平均：BA
  {format_percent(performance.loc['random_top20_mean', 'balanced_accuracy_mean'])}。
- 图谱定义的价值相关 ROI 全脑标量均值替代：BA
  {format_percent(performance.loc['value', 'balanced_accuracy_mean'])}；
  图谱定义的枕叶视觉 ROI 全脑标量均值替代：BA
  {format_percent(performance.loc['visual', 'balanced_accuracy_mean'])}；
  二者逻辑并集 ROI 全脑标量均值替代：BA
  {format_percent(performance.loc['union', 'balanced_accuracy_mean'])}。

等体素随机替代 BA 减训练折类别等权联合 CAM 高排名体素替代 BA 的差值为
{format_percentage_points(tests.loc['random_top20_minus_cam_top20', 'mean_effect'])}
（95% CI
{format_percentage_points(tests.loc['random_top20_minus_cam_top20', 'ci_low'])}–
{format_percentage_points(tests.loc['random_top20_minus_cam_top20', 'ci_high'])}，
Holm $p={format_p(tests.loc['random_top20_minus_cam_top20', 'p_holm'])}$）。
图谱定义的价值相关 ROI、图谱定义的枕叶视觉 ROI 与二者逻辑并集 ROI 的
完整输入减替代效应分别为
{format_percentage_points(tests.loc['full_minus_value', 'mean_effect'])}、
{format_percentage_points(tests.loc['full_minus_visual', 'mean_effect'])} 和
{format_percentage_points(tests.loc['full_minus_union', 'mean_effect'])}；对应 Holm
$p$ 分别为 {format_p(tests.loc['full_minus_value', 'p_holm'])}、
{format_p(tests.loc['full_minus_visual', 'p_holm'])} 和
{format_p(tests.loc['full_minus_union', 'p_holm'])}。

这些效应量化固定 $k=6$ 体积模型对特定空间全脑标量均值替代操作的性能敏感性；
它们不定位或解释主要 $k=1$ 单试次效应，不表示脑区的神经因果必要性，
也不等同于纯粹删除区域信息。由于三组图谱掩膜体素数不同，且只有CAM
前20%具有等体素随机参照，价值、视觉与联合 ROI 的完整输入减替代效应
不能解释为功能系统间的相对贡献、特异依赖或协同整合。
"""

    en = f"""# RQ3 training-fold global-scalar mean-replacement experiment: methods and results

## Methods

Each $k=6$ input was the average of six single-trial beta maps from the same
participant and the same observed rating class. We reused the final-epoch
checkpoints from the existing 10-fold participant-disjoint validation of this
condition-level 3D-CNN, without model retraining; each fold contained 27
training participants and three held-out participants.

Within each fold, Layer-CAM was generated exclusively from correctly classified
training inputs. CAMs were first averaged within participant and rating class,
then equally across contributing participants who had at least one correct
training input in that class. The rating-1 and rating-4 training maps were each
divided by their mean within the common brain mask and combined with equal class
weights. The {n_top:,} voxels with the highest values (20% of the fixed
{int(config['expected']['common_mask_voxels']):,}-voxel common mask) were selected
deterministically.

For each fold, {n_random} random masks containing the same number of voxels
were sampled without replacement from the same common mask using fixed seeds;
this random condition was the voxel-count-matched reference only for the CAM
top-20% condition. Fixed masks were derived from the 2-mm Harvard–Oxford
cortical/subcortical maximum-probability atlas at the 25% threshold, resampled
by nearest neighbour to the CNN input grid, and intersected with the common
brain mask. The value-related set comprised bilateral Frontal Medial Cortex,
bilateral Frontal Orbital Cortex, and Left/Right Accumbens. The occipital visual
set comprised the nine bilateral occipital labels frozen in the configuration;
these were Lateral Occipital Cortex (superior and inferior divisions),
Intracalcarine Cortex, Cuneal Cortex, Lingual Gyrus, Temporal Occipital
Fusiform Cortex, Occipital Fusiform Gyrus, Supracalcarine Cortex, and
Occipital Pole. Their logical union formed the union mask.

For each fold, one scalar global mean was pooled across all voxels in all
training inputs in the model-input space, restricted to each input's own
participant brain mask. Every selected voxel in every held-out input received
that same scalar. No coordinate-specific mean map was computed or used. Inputs
were not renormalized after replacement, and the model was not retrained.

The primary metric was participant-level balanced accuracy. Four prespecified
contrasts were tested with {int(config['statistics']['num_sign_flips']):,}
one-sided participant sign flips, {int(config['statistics']['num_bootstrap']):,}
participant bootstrap resamples for 95% confidence intervals, and Holm
correction: random-20% minus CAM-top-20%, full minus value, full minus visual,
and full minus union. AUC was secondary and descriptive.

## Results

Full-input balanced accuracy was
{format_percent(performance.loc['full', 'balanced_accuracy_mean'])}
(95% CI {format_percent(performance.loc['full', 'balanced_accuracy_ci_low'])}–
{format_percent(performance.loc['full', 'balanced_accuracy_ci_high'])});
AUC was {format_percent(performance.loc['full', 'roc_auc_mean'])}.
Balanced accuracy was
{format_percent(performance.loc['cam_top20', 'balanced_accuracy_mean'])}
after training-fold class-balanced joint-CAM-top-20% global-scalar replacement and
{format_percent(performance.loc['random_top20_mean', 'balanced_accuracy_mean'])}
for the average random-20% global-scalar control. Global-scalar replacement in the atlas-defined
value-related ROI, atlas-defined occipital-visual ROI, and their logical-union
mask yielded
{format_percent(performance.loc['value', 'balanced_accuracy_mean'])},
{format_percent(performance.loc['visual', 'balanced_accuracy_mean'])}, and
{format_percent(performance.loc['union', 'balanced_accuracy_mean'])},
respectively.

The matched-random-minus-CAM replacement contrast was
{format_percentage_points_en(tests.loc['random_top20_minus_cam_top20', 'mean_effect'])}
(95% CI
{format_percentage_points_en(tests.loc['random_top20_minus_cam_top20', 'ci_low'])}–
{format_percentage_points_en(tests.loc['random_top20_minus_cam_top20', 'ci_high'])};
Holm $p={format_p(tests.loc['random_top20_minus_cam_top20', 'p_holm'])}$).
The full-minus-replacement effects for the atlas-defined value-related ROI,
atlas-defined occipital-visual ROI, and their logical-union mask were
{format_percentage_points_en(tests.loc['full_minus_value', 'mean_effect'])},
{format_percentage_points_en(tests.loc['full_minus_visual', 'mean_effect'])}, and
{format_percentage_points_en(tests.loc['full_minus_union', 'mean_effect'])}, respectively
(Holm $p={format_p(tests.loc['full_minus_value', 'p_holm'])}$,
{format_p(tests.loc['full_minus_visual', 'p_holm'])}$, and
{format_p(tests.loc['full_minus_union', 'p_holm'])}$).

These effects quantify the sensitivity of the fixed $k=6$ volumetric model to
specific global-scalar replacement operations. They neither localize nor explain
the primary $k=1$ single-trial effect, establish neural causal necessity, nor
represent pure deletion of regional information. Because the atlas masks differ
in voxel count and only CAM top-20% has a voxel-count-matched random reference,
the value, visual, and union full-minus-replacement effects cannot be interpreted
as relative functional-system contributions, specific dependencies, or
synergistic integration.
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "methods_results_cn.md").write_text(cn, encoding="utf-8")
    (output_dir / "methods_results_en.md").write_text(en, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    raw_validation_path = (
        PROJECT_ROOT / "outputs" / "validation" / "raw_validation_report.json"
    )
    if not raw_validation_path.exists():
        raise FileNotFoundError(
            "Raw fold outputs have not been validated. Run "
            "`python code/validate_results.py --raw-only` first."
        )
    raw_validation = read_json(raw_validation_path)
    if (
        not bool(raw_validation.get("passed", False))
        or raw_validation.get("stage") != "raw_fold_outputs"
        or not raw_validation.get("raw_artifact_sha256")
    ):
        raise ValueError("Raw fold-output validation did not pass.")
    provenance = read_json(PROJECT_ROOT / "inputs" / "provenance.json")
    verify_frozen_static_dependencies(config, provenance)
    if (
        str(raw_validation.get("experiment_config_sha256", ""))
        != sha256_file(config["_config_path"])
        or str(raw_validation.get("roi_definitions_sha256", ""))
        != sha256_file(CONFIG_ROOT / "roi_definitions.json")
    ):
        raise ValueError(
            "Raw validation was produced under different experiment/ROI "
            "definitions."
        )
    for relative_path, expected_hash in raw_validation.get(
        "raw_artifact_sha256",
        {},
    ).items():
        live_path = PROJECT_ROOT / relative_path
        if not live_path.exists() or sha256_file(live_path) != expected_hash:
            raise ValueError(
                f"Raw artifact changed after validation: {relative_path}"
            )
    expected_folds = int(config["expected"]["num_folds"])
    fold_dirs = [
        PROJECT_ROOT / "outputs" / "folds" / f"fold_{fold:02d}"
        for fold in range(1, expected_folds + 1)
    ]
    missing = [
        str(path)
        for path in fold_dirs
        if not (path / "fold_complete.json").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Completed fold outputs are missing: {missing}")

    fixed_frames = [
        pd.read_csv(path / "predictions" / "fixed_condition_predictions.csv")
        for path in fold_dirs
    ]
    fixed = pd.concat(fixed_frames, ignore_index=True)
    computed_sample_ids = [
        stable_sample_id(subject, rating, trial)
        for subject, rating, trial in zip(
            fixed["subject_id"],
            fixed["rating"],
            fixed["trial_id"],
        )
    ]
    if "sample_id" not in fixed.columns:
        fixed.insert(0, "sample_id", computed_sample_ids)
    else:
        missing_sample_id = fixed["sample_id"].isna() | (
            fixed["sample_id"].astype(str).str.strip() == ""
        )
        fixed.loc[missing_sample_id, "sample_id"] = np.asarray(
            computed_sample_ids,
            dtype=object,
        )[missing_sample_id.to_numpy()]
    if fixed["sample_id"].astype(str).tolist() != computed_sample_ids:
        raise ValueError("Fixed-prediction sample IDs are inconsistent.")
    fixed = fixed.sort_values(
        ["fold_index", "subject_id", "trial_id", "condition"]
    ).reset_index(drop=True)
    probability_threshold = float(config["inference"]["probability_threshold"])
    fixed_participant, fixed_fold = summarize_fixed_predictions(
        fixed,
        probability_threshold=probability_threshold,
    )
    random_participant, random_fold, random_sample_mean = read_random_metrics(
        fold_dirs,
        probability_threshold=probability_threshold,
    )
    random_mean = participant_random_means(random_participant)
    participant_conditions = pd.concat(
        [fixed_participant, random_mean],
        ignore_index=True,
        sort=False,
    )
    participant_conditions["condition_order"] = participant_conditions[
        "condition"
    ].map({value: index for index, value in enumerate(condition_order())})
    participant_conditions = participant_conditions.sort_values(
        ["subject_id", "condition_order"]
    ).drop(columns=["condition_order"])
    effects = build_participant_effects(participant_conditions)

    statistics = config["statistics"]
    group_summary = group_condition_summary(
        participant_conditions,
        num_bootstrap=int(statistics["num_bootstrap"]),
        seed=int(statistics["seed"]),
    )
    inference = primary_inference(
        effects,
        primary_contrasts=list(statistics["primary_contrasts"]),
        num_sign_flips=int(statistics["num_sign_flips"]),
        num_bootstrap=int(statistics["num_bootstrap"]),
        seed=int(statistics["seed"]),
    )

    mask_rows: list[dict[str, Any]] = []
    for fold_dir in fold_dirs:
        complete = read_json(fold_dir / "fold_complete.json")
        training = complete["training_artifacts"]
        random_metadata = complete["random_masks"]
        mask_rows.append(
            {
                "fold_index": int(complete["fold_index"]),
                "num_train_subjects": int(len(complete["train_subjects"])),
                "num_test_subjects": int(len(complete["test_subjects"])),
                "num_train_inputs": int(complete["num_train_inputs"]),
                "num_test_inputs": int(complete["num_test_inputs"]),
                "num_correct_training_inputs_for_cam": int(
                    training["cam"]["num_correct_training_inputs"]
                ),
                "num_correct_training_rating1": int(
                    training["cam"]["num_correct_rating1"]
                ),
                "num_correct_training_rating4": int(
                    training["cam"]["num_correct_rating4"]
                ),
                "num_contributing_subjects_rating1": int(
                    training["cam"]["num_subjects_with_correct_rating1"]
                ),
                "num_contributing_subjects_rating4": int(
                    training["cam"]["num_subjects_with_correct_rating4"]
                ),
                "cam_top20_voxels": int(
                    training["top20_selection"]["num_selected_voxels"]
                ),
                "cam_top20_minimum_selected_value": float(
                    training["top20_selection"]["minimum_selected_cam"]
                ),
                "random_replicates": int(
                    random_metadata["num_replicates"]
                ),
                "random_voxels_per_mask": int(
                    random_metadata["num_voxels_per_mask"]
                ),
                "random_fold_seed": int(random_metadata["fold_seed"]),
                "value_roi_voxels": 4092,
                "visual_roi_voxels": 27651,
                "union_roi_voxels": 31743,
            }
        )
    mask_summary = pd.DataFrame(mask_rows).sort_values("fold_index")

    random_fold_mean = (
        random_fold.groupby("fold_index", as_index=False)
        .agg(
            n_samples=("n_samples", "first"),
            n_rating1=("n_rating1", "first"),
            n_rating4=("n_rating4", "first"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            recall_rating1=("recall_rating1", "mean"),
            recall_rating4=("recall_rating4", "mean"),
            roc_auc=("roc_auc", "mean"),
        )
    )
    random_fold_mean["condition"] = "random_top20_mean"
    fold_conditions = pd.concat(
        [fixed_fold, random_fold_mean],
        ignore_index=True,
        sort=False,
    )

    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(tables_dir / "sample_predictions_fixed_conditions.csv", index=False)
    random_sample_mean.to_csv(
        tables_dir / "sample_random_mean_probability.csv",
        index=False,
    )
    random_participant.to_csv(
        tables_dir / "participant_random_replicate_metrics.csv",
        index=False,
    )
    random_fold.to_csv(
        tables_dir / "fold_random_replicate_metrics.csv",
        index=False,
    )
    participant_conditions.to_csv(
        tables_dir / "participant_condition_metrics.csv",
        index=False,
    )
    effects.to_csv(tables_dir / "participant_effects.csv", index=False)
    fold_conditions.to_csv(
        tables_dir / "fold_condition_metrics.csv",
        index=False,
    )
    group_summary.to_csv(
        tables_dir / "group_condition_summary.csv",
        index=False,
    )
    inference.to_csv(tables_dir / "primary_inference.csv", index=False)
    mask_summary.to_csv(
        tables_dir / "fold_mask_and_cam_summary.csv",
        index=False,
    )

    make_figure(
        group_summary,
        inference,
        PROJECT_ROOT / "outputs" / "figures",
    )
    write_paper_summaries(
        group_summary=group_summary,
        inference=inference,
        config=config,
        output_dir=PROJECT_ROOT / "outputs" / "paper",
    )

    summary = {
        "experiment_id": config["experiment_id"],
        "num_folds": expected_folds,
        "num_subjects": int(participant_conditions["subject_id"].nunique()),
        "num_samples": int(
            fixed.loc[fixed["condition"] == "full"].shape[0]
        ),
        "num_random_replicates_per_fold": int(
            random_participant["random_replicate"].nunique()
        ),
        "group_condition_summary": group_summary.to_dict(orient="records"),
        "primary_inference": inference.to_dict(orient="records"),
        "interpretation_boundary": (
            "Performance sensitivity of the trained volumetric model to global-scalar "
            "mean replacement; not neural causal necessity."
        ),
        "files": {
            path.name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in sorted(tables_dir.glob("*.csv"))
        },
    }
    write_json(PROJECT_ROOT / "outputs" / "results_summary.json", summary)
    print(group_summary.to_string(index=False), flush=True)
    print(inference.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
