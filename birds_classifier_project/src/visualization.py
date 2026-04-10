from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .constants import CLASS_LABEL_COLUMN


def _display_feature_name(feature_name: str) -> str:
    if feature_name == "gender":
        return "gender (female=0, male=1)"
    return feature_name


def create_decision_boundary_figure(result) -> Figure:
    split = result.split
    feature_1, feature_2 = split.feature_names
    class_1, class_2 = split.class_pair

    figure = Figure(figsize=(7, 5), dpi=100)
    axis = figure.add_subplot(111)

    train_df = split.train_df
    test_df = split.test_df

    # Plot training points.
    class_1_train = train_df[train_df[CLASS_LABEL_COLUMN] == class_1]
    class_2_train = train_df[train_df[CLASS_LABEL_COLUMN] == class_2]
    class_1_test = test_df[test_df[CLASS_LABEL_COLUMN] == class_1]
    class_2_test = test_df[test_df[CLASS_LABEL_COLUMN] == class_2]

    axis.scatter(
        class_1_train[feature_1],
        class_1_train[feature_2],
        label=f"{class_1} train",
        alpha=0.85,
        marker="o",
    )
    axis.scatter(
        class_2_train[feature_1],
        class_2_train[feature_2],
        label=f"{class_2} train",
        alpha=0.85,
        marker="o",
    )
    axis.scatter(
        class_1_test[feature_1],
        class_1_test[feature_2],
        label=f"{class_1} test",
        alpha=0.85,
        marker="x",
    )
    axis.scatter(
        class_2_test[feature_1],
        class_2_test[feature_2],
        label=f"{class_2} test",
        alpha=0.85,
        marker="x",
    )

    weights = result.model.weights
    bias = result.model.bias

    x_values = np.linspace(
        min(train_df[feature_1].min(), test_df[feature_1].min()) - 1,
        max(train_df[feature_1].max(), test_df[feature_1].max()) + 1,
        200,
    )

    if abs(weights[1]) > 1e-12:
        y_values = -(weights[0] * x_values + bias) / weights[1]
        axis.plot(x_values, y_values, linewidth=2, label="Decision boundary")
    elif abs(weights[0]) > 1e-12:
        x_constant = -bias / weights[0]
        axis.axvline(x=x_constant, linewidth=2, label="Decision boundary")

    axis.set_title(
        f"{result.model.__class__.__name__}: {class_1} vs {class_2}",
        pad=12,
    )
    axis.set_xlabel(_display_feature_name(feature_1))
    axis.set_ylabel(_display_feature_name(feature_2))
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    return figure
