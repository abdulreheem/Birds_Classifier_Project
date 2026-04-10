from __future__ import annotations

from typing import Dict, List

import numpy as np


def binary_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    negative_label: int = -1,
    positive_label: int = 1,
) -> Dict[str, int | List[List[int]]]:
    tn = fp = fn = tp = 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == negative_label and predicted == negative_label:
            tn += 1
        elif actual == negative_label and predicted == positive_label:
            fp += 1
        elif actual == positive_label and predicted == negative_label:
            fn += 1
        elif actual == positive_label and predicted == positive_label:
            tp += 1

    matrix = [
        [tn, fp],  # Actual negative row
        [fn, tp],  # Actual positive row
    ]

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "matrix": matrix,
    }


def accuracy_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    correct = int(np.sum(y_true == y_pred))
    return correct / len(y_true)


def format_confusion_matrix(
    matrix: List[List[int]],
    class_names: tuple[str, str],
) -> str:
    class_1, class_2 = class_names
    return (
        f"{'':18s}Predicted {class_1:>6s}   Predicted {class_2:>6s}\n"
        f"Actual {class_1:>7s}{matrix[0][0]:>16d}{matrix[0][1]:>18d}\n"
        f"Actual {class_2:>7s}{matrix[1][0]:>16d}{matrix[1][1]:>18d}"
    )
