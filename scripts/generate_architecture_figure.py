from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "latex" / "figures" / "pipeline_architecture.pdf"


def add_box(ax, xy, width, height, title, lines, facecolor):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#222222",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + 0.03 * width,
        y + height - 0.12 * height,
        title,
        fontsize=9.3,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
    )
    ax.text(
        x + 0.03 * width,
        y + height - 0.28 * height,
        "\n".join(lines),
        fontsize=7.2,
        ha="left",
        va="top",
        color="#111111",
        linespacing=1.35,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color="#333333",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.35, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = 0.08
    w = 0.84
    h = 0.16
    ys = [0.78, 0.56, 0.34, 0.12]

    add_box(
        ax,
        (x, ys[0]),
        w,
        h,
        "Prompt Interface",
        [
            "Free-form request",
            "task, labels, priority,",
            "environment constraints",
            "Example: low light + speed",
        ],
        "#F6E7CB",
    )
    add_box(
        ax,
        (x, ys[1]),
        w,
        h,
        "Parser + Resolver",
        [
            "Typed schema extraction",
            "synonym / label grounding",
            "augmentation policy",
            "model-family selection",
            "Output: pipeline_config.json",
        ],
        "#D9E8F5",
    )
    add_box(
        ax,
        (x, ys[2]),
        w,
        h,
        "Training Factory",
        [
            "Classification or detection",
            "dataset loaders + transforms",
            "ImageNet initialization",
            "checkpoint + metrics",
            "Output: best_model.pt",
        ],
        "#DCEED7",
    )
    add_box(
        ax,
        (x, ys[3]),
        w,
        h,
        "Evaluation + Export",
        [
            "clean / corrupted benchmarks",
            "CPU latency, smoke tests",
            "ONNX conversion + metadata",
            "Markdown report generation",
            "Outputs: model.onnx, report",
        ],
        "#F2D9D5",
    )

    for i in range(3):
        start = (0.5, ys[i] - 0.01)
        end = (0.5, ys[i + 1] + h + 0.01)
        add_arrow(ax, start, end)

    ax.text(
        0.5,
        0.04,
        "Prompt-conditioned handoff chain implemented end to end.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#222222",
    )

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
