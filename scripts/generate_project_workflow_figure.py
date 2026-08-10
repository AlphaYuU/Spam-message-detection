from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figure" / "figure_3_x_overall_project_workflow.png"

GOLD = "#D9A400"
RED = "#F02B20"
BLUE = "#2B49D8"
DARK = "#181818"


def draw_top_group(ax, x, title, items, color):
    spine_y = 0.48

    ax.plot([x, x], [spine_y, 0.555], color=color, linewidth=2.6)
    ax.scatter([x], [spine_y], s=96, color=color, zorder=5)

    ax.text(
        x,
        0.586,
        title,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="medium",
        color=DARK,
    )
    ax.plot([x - 0.065, x + 0.065], [0.572, 0.572], color=color, linewidth=2.2)

    top = 0.925
    bottom = 0.64
    ax.plot([x, x], [bottom, top], color=color, linewidth=1.8)

    if len(items) == 1:
        ys = [(top + bottom) / 2]
    else:
        step = (top - bottom) / (len(items) - 1)
        ys = [top - i * step for i in range(len(items))]

    for y, item in zip(ys, items):
        ax.plot([x, x + 0.031], [y - 0.021, y], color=color, linewidth=1.55)
        ax.text(
            x + 0.039,
            y,
            item,
            ha="left",
            va="center",
            fontsize=10.7,
            color=DARK,
            linespacing=1.05,
        )


def draw_bottom_group(ax, x, title, items, color):
    spine_y = 0.48

    ax.plot([x, x], [spine_y, 0.415], color=color, linewidth=2.6)
    ax.scatter([x], [spine_y], s=96, color=color, zorder=5)

    ax.text(
        x,
        0.386,
        title,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="medium",
        color=DARK,
    )
    ax.plot([x - 0.073, x + 0.073], [0.352, 0.352], color=color, linewidth=2.2)

    top = 0.315
    bottom = 0.055
    ax.plot([x, x], [top, bottom], color=color, linewidth=1.8)

    if len(items) == 1:
        ys = [(top + bottom) / 2]
    else:
        step = (top - bottom) / (len(items) - 1)
        ys = [top - i * step for i in range(len(items))]

    for y, item in zip(ys, items):
        ax.plot([x, x + 0.031], [y + 0.021, y], color=color, linewidth=1.55)
        ax.text(
            x + 0.039,
            y,
            item,
            ha="left",
            va="center",
            fontsize=10.7,
            color=DARK,
            linespacing=1.05,
        )


def main():
    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    spine_y = 0.48
    label_box = FancyBboxPatch(
        (0.026, spine_y - 0.055),
        0.19,
        0.11,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        facecolor="#111111",
    )
    ax.add_patch(label_box)
    ax.text(
        0.121,
        spine_y,
        "SMS Spam Classification\nProject Workflow",
        ha="center",
        va="center",
        color="white",
        fontsize=13.7,
        fontweight="bold",
        linespacing=1.25,
    )

    ax.plot([0.216, 0.955], [spine_y, spine_y], color=GOLD, linewidth=2.8)

    draw_top_group(
        ax,
        0.300,
        "Data Preparation",
        [
            "English and multilingual datasets",
            "Data cleaning and standardisation",
            "Separate dataset splitting",
            "TF-IDF feature extraction",
            "XLM-RoBERTa tokenisation",
        ],
        RED,
    )

    draw_bottom_group(
        ax,
        0.415,
        "Model Training and Selection",
        [
            "Support Vector Machine",
            "Logistic Regression",
            "English XLM-RoBERTa",
            "Multilingual XLM-RoBERTa",
            "F1-based checkpoint selection",
            "Combination-weight grid search",
        ],
        GOLD,
    )

    draw_top_group(
        ax,
        0.565,
        "Model Testing",
        [
            "Held-out English test set",
            "Held-out multilingual test set",
            "Stratified five-fold cross-validation",
            "SVM and Logistic Regression only",
        ],
        BLUE,
    )

    draw_bottom_group(
        ax,
        0.680,
        "Model Evaluation",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 score",
            "Confusion matrices",
            "Language-specific performance",
        ],
        RED,
    )

    draw_top_group(
        ax,
        0.815,
        "System Integration",
        [
            "Desktop user interface",
            "Model loading and selection",
            "SMS classification",
            "Confidence-score output",
            "Feature-contribution explanation",
        ],
        GOLD,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
