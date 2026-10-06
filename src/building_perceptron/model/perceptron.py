from pathlib import Path

import numpy as np
import pandas as pd

from building_perceptron.model.linear_classifier import LinearClassifier

class Perceptron(LinearClassifier):

    def __init__(self, n_features: int, learning_rate: float = 0.01) -> None:
        if n_features <= 0:
            raise ValueError("n_features doit être > 0")
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0


    def __str__(self) -> str:
        return (
            f"Perceptron(n_features={self.n_features}, "
            f"learning_rate={self.learning_rate}, "
            f"bias={self.bias}, weights={self.weights.tolist()})"
        )