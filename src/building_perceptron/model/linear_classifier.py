from pathlib import Path

import numpy as np
import pandas as pd


class LinearClassifier:
    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    def __str__(self) -> str:
        return f"LinearClassifier(n_features={self.n_features})"