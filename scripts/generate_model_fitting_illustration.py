from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "figure"
PNG_PATH = OUTPUT_DIR / "figure_2_x_model_fitting_illustration.png"

RANDOM_STATE = 42


def build_classifier(gamma: float, regularisation_strength: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    gamma=gamma,
                    C=regularisation_strength,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    features, labels = make_moons(
        n_samples=120,
        noise=0.24,
        random_state=RANDOM_STATE,
    )

    # A small number of deliberately noisy observations makes the difference
    # between an appropriate boundary and an overly flexible boundary visible.
    noisy_indices = np.array([7, 19, 36, 58, 82, 105])
    labels[noisy_indices] = 1 - labels[noisy_indices]

    panel_specs = [
        ("Underfitting", "low complexity", 0.04, 0.5),
        ("Appropriate fit", "moderate complexity", 1.5, 5.0),
        ("Overfitting", "high complexity", 35.0, 100.0),
    ]

    x_min, x_max = features[:, 0].min() - 0.45, features[:, 0].max() + 0.45
    y_min, y_max = features[:, 1].min() - 0.40, features[:, 1].max() + 0.40
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 280),
        np.linspace(y_min, y_max, 210),
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    region_colours = ListedColormap(["#E7ECFF", "#FFF0EE"])
    class_colours = {0: "#2632E6", 1: "#E51C23"}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.35), sharex=True, sharey=True)

    for axis, (fit_label, complexity_label, gamma, c_value) in zip(axes, panel_specs):
        classifier = build_classifier(gamma, c_value)
        classifier.fit(features, labels)
        region_predictions = classifier.predict(grid).reshape(grid_x.shape)
        decision_scores = classifier.decision_function(grid).reshape(grid_x.shape)

        axis.contourf(
            grid_x,
            grid_y,
            region_predictions,
            levels=[-0.5, 0.5, 1.5],
            cmap=region_colours,
            alpha=1.0,
        )
        axis.contour(
            grid_x,
            grid_y,
            decision_scores,
            levels=[0.0],
            colors=["#4B5563"],
            linewidths=1.8,
        )

        class_zero = labels == 0
        class_one = labels == 1
        axis.scatter(
            features[class_zero, 0],
            features[class_zero, 1],
            c=class_colours[0],
            marker="o",
            s=34,
            edgecolors="white",
            linewidths=0.45,
            label="Class 0",
            zorder=3,
        )
        axis.scatter(
            features[class_one, 0],
            features[class_one, 1],
            c=class_colours[1],
            marker="^",
            s=40,
            edgecolors="white",
            linewidths=0.45,
            label="Class 1",
            zorder=3,
        )

        axis.set_title(f"{fit_label}\n({complexity_label})", pad=10)
        axis.set_xlabel("Feature 1")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.grid(color="#CBD0D6", linewidth=0.7, alpha=0.55)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Feature 2")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()
