#!/usr/bin/env python3
"""Render the verified, explicitly selected ordinary efficiency summary."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(args.summary.read_text())
    groups = [group for group in summary["comparison_groups"] if len(group["arms"]) > 1]
    if not groups:
        raise ValueError("No verified timing cohorts")
    labels = {
        "control": "Control: all checkpointed",
        "fa4": "FA4*",
        "compiled": "Compiled SwiGLU (rounded)",
        "compiled-checkpoint-alternating": "Compiled + alternate",
        "compiled-checkpoint-none": "Compiled + no checkpoint",
        "checkpoint-alternating": "Alternate checkpointing",
        "checkpoint-none": "No checkpointing",
        "fa4-compiled": "FA4 + compiled*",
        "fa4-compiled-checkpoint-alternating": "FA4 + compiled + alternate*",
        "fa4-compiled-checkpoint-none": "FA4 + compiled + no checkpoint*",
    }
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5.5), squeeze=False)
    for ax, group in zip(axes[0], groups):
        arms = [name for name in labels if name in group["arms"]]
        rows = [group["arms"][name] for name in arms]
        values = np.array([row["median_input_tokens_per_second"] / 1000 for row in rows])
        lower = np.array([row["minimum_input_tokens_per_second"] / 1000 for row in rows])
        upper = np.array([row["maximum_input_tokens_per_second"] / 1000 for row in rows])
        bars = ax.barh(range(len(arms)), values, xerr=[values-lower, upper-values],
                       color=["#536d82" if name == "control" else "#438f83" for name in arms], capsize=3)
        ax.set_yticks(range(len(arms)), [labels[name] for name in arms])
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5, fontsize=9)
        ax.set_xlim(0, max(upper) * 1.22)
        ax.set_xlabel("Input tokens / second (thousands)")
        ax.set_title(f"B{group['batch_size']} / T{group['length']}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pretrained OLMo-1B: complete optimizer-step throughput", fontsize=14)
    fig.text(.02, .025, "H100 80GB · BF16 mixed · CUDA graphs · full CE · five timed updates/run\n"
             "Whiskers: range of run medians, not confidence intervals. *See retained numerical qualifications; no quality claim.",
             fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .93))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_dir / ("throughput." + suffix), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
