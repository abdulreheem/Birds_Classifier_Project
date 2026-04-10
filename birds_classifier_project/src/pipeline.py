from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .adaline import AdalineClassifier
from .data_loader import BirdDataset, DatasetSplit
from .metrics import accuracy_score_manual, binary_confusion_matrix
from .perceptron import PerceptronClassifier


@dataclass
class ExperimentResult:
    model: object
    split: DatasetSplit
    confusion_matrix: list[list[int]]
    accuracy: float
    decision_equation: str
    epochs_completed: int


def format_decision_equation(
    weights,
    bias: float,
    feature_names: List[str],
) -> str:
    return (
        f"{weights[0]:.6f} * {feature_names[0]} + "
        f"{weights[1]:.6f} * {feature_names[1]} + "
        f"{bias:.6f} = 0"
    )


def run_experiment(
    csv_path: str,
    selected_features: List[str],
    class_pair: Tuple[str, str],
    algorithm: str,
    eta: float,
    epochs: int,
    mse_threshold: float,
    use_bias: bool,
    seed: int = 42,
) -> ExperimentResult:
    dataset = BirdDataset(csv_path)
    split = dataset.prepare_binary_split(
        selected_features=selected_features,
        class_pair=class_pair,
        train_per_class=30,
        seed=seed,
    )

    algorithm_normalized = algorithm.strip().lower()
    if algorithm_normalized == "perceptron":
        model = PerceptronClassifier(
            eta=eta,
            epochs=epochs,
            use_bias=use_bias,
            random_state=seed,
        )
    elif algorithm_normalized == "adaline":
        model = AdalineClassifier(
            eta=eta,
            epochs=epochs,
            mse_threshold=mse_threshold,
            use_bias=use_bias,
            random_state=seed,
        )
    else:
        raise ValueError("Algorithm must be either 'perceptron' or 'adaline'.")

    model.fit(split.X_train, split.y_train)
    y_pred = model.predict(split.X_test)

    confusion = binary_confusion_matrix(split.y_test, y_pred)
    accuracy = accuracy_score_manual(split.y_test, y_pred)

    return ExperimentResult(
        model=model,
        split=split,
        confusion_matrix=confusion["matrix"],
        accuracy=accuracy,
        decision_equation=format_decision_equation(
            model.weights,
            model.bias,
            split.feature_names,
        ),
        epochs_completed=model.epochs_completed,
    )
