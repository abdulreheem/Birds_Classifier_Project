from __future__ import annotations

import numpy as np


class PerceptronClassifier:
    def __init__(
        self,
        eta: float = 0.01,
        epochs: int = 100,
        use_bias: bool = True,
        random_state: int = 42,
    ) -> None:
        self.eta = eta
        self.epochs = epochs
        self.use_bias = use_bias
        self.random_state = random_state

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.history: list[float] = []
        self.epochs_completed: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronClassifier":
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = float(rng.normal(loc=0.0, scale=0.01)) if self.use_bias else 0.0

        self.history = []

        for epoch in range(self.epochs):
            misclassified = 0

            for xi, target in zip(X, y):
                net = self.net_input(xi)
                output = self.signum(net)
                error = target - output

                if error != 0:
                    self.weights = self.weights + self.eta * error * xi
                    if self.use_bias:
                        self.bias = self.bias + self.eta * error
                    misclassified += 1

            self.history.append(float(misclassified))
            self.epochs_completed = epoch + 1

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("The model must be trained before use.")
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def signum(value: np.ndarray | float) -> np.ndarray | int:
        if isinstance(value, np.ndarray):
            return np.where(value >= 0, 1, -1)
        return 1 if value >= 0 else -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        net = self.net_input(X)
        return self.signum(net)
