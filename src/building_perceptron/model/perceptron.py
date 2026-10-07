from pathlib import Path

import numpy as np
import pandas as pd

class Perceptron:

    def __init__(self, lr: float = 0.01, epochs: int = 100) -> None:
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def activation_function(self, z: float | np.ndarray) -> np.ndarray:
        """Fonction d'activation Heaviside : renvoie 1 si z >= 0, sinon 0."""
        return np.where(z >= 0, 1, 0)

    # Le modèle va apprendre la valeur de chaque poids :
    # - Si une caractéristique est très utile, son poids w_j deviendra grand.
    # - Si une caractéristique est inutile/inutilement complexe, son poids w_j restera très proche de 0.
    # def fit_perceptron(X, y, lr=0.1, epochs=100):
    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        w = np.zeros(n, dtype=float)
        b = 0.0
        lr = self.lr
        margin = 1e-9

        for epoch in range(self.epochs):
            errors_in_epoch = 0
            
            for i in range(m):
                # Combinaison linéaire (z)
                z_i = np.dot(X[i], w) + b
                
                # Prédiction binaire stricte (0 ou 1)
                y_hat_i = self.activation_function(z_i)
                
                # Calcul de l'erreur brute : vaut -1, 0, ou 1
                # Le modèle s'est trompé en prédisant 1 : error = -1
                if y[i] == 1 and z_i <= margin:
                    error = 1.0
                # Le modèle s'est trompé en prédisant 0 : error = +1
                elif y[i] == 0 and z_i >= -margin:
                    error = -1.0
                else:
                    error = 0.0
                
                # Mise à jour directe si erreur
                # Le modèle s'est trompé en prédisant 0 : error = +1
                # Le modèle s'est trompé en prédisant 1 : error = -1
                if error != 0:
                    # si error > 0, w augmente, ce qui pousse la prédiction vers 1
                    # si error < 0, w diminue, ce qui pousse la prédiction vers 0
                    w += lr * error * X[i]
                    b += lr * error
                    errors_in_epoch += 1
                    
            # Arrêt précoce si aucune erreur sur l'ensemble du dataset
            if errors_in_epoch == 0:
                print(f"Convergence atteinte à l'époque {epoch + 1} !")
                break

        self.w = w
        self.b = b

        return self


    def fit_classique(self, X: np.ndarray, y: np.ndarray):
        """
        Perceptron de Rosenblatt : prédiction y_hat = 1 si z > 0, sinon 0.
        Un exemple sur la frontière (z = 0) est classé 0 : il n'est donc corrigé que s'il est de classe 1.
        """
        m, n = X.shape
        self.w = np.zeros(n, dtype=float)
        self.b = 0.0

        for epoch in range(self.epochs):
            erreurs_epoque = 0
            for i in range(m):  # ordre fixe, pour comparer à fit_strict
                z_i = np.dot(X[i], self.w) + self.b
                y_hat_i = self.activation_function(z_i)
                erreur = y[i] - y_hat_i  # -1, 0 ou +1

                if erreur != 0:
                    self.w += self.lr * erreur * X[i]
                    self.b += self.lr * erreur
                    erreurs_epoque += 1

            if erreurs_epoque == 0:
                print(f"Convergence atteinte à l'époque {epoch + 1} !")
                break

        return self

    def fit_strict(self, X: np.ndarray, y: np.ndarray):
        """
        Perceptron à séparation stricte : un exemple est considéré comme mal classé
        s'il est du mauvais côté de la frontière OU exactement dessus (z ≈ 0).
        """
        m, n = X.shape
        w = np.zeros(n, dtype=float)
        b = 0.0
        lr = self.lr

        # marge fonctionnelle : un exemple n'est "bien classé" que si
        #   y=1 -> z > marge  et  y=0 -> z < -marge.
        #   Un z dans [-marge, marge] est traité comme "sur la frontière" (donc erreur),
        #   même si son signe est correct.
        marge = 1e-9

        for epoch in range(self.epochs):
            erreurs_epoque = 0

            for i in range(m):
                # Combinaison linéaire : z = w·x + b
                z_i = np.dot(X[i], w) + b

                # Signe de la correction à appliquer :
                #   +1 -> l'exemple est de classe 1 mais z n'est pas assez positif : on pousse z vers le haut
                #   -1 -> l'exemple est de classe 0 mais z n'est pas assez négatif : on pousse z vers le bas
                #    0 -> exemple correctement classé, strictement du bon côté : rien à faire
                if y[i] == 1 and z_i <= marge:
                    erreur = 1.0
                elif y[i] == 0 and z_i >= -marge:
                    erreur = -1.0
                else:
                    erreur = 0.0

                # Règle du perceptron : on déplace la frontière dans la direction de x
                #   erreur = +1 -> w augmente dans la direction de X[i], z croît
                #   erreur = -1 -> w diminue dans la direction de X[i], z décroît
                if erreur != 0:
                    w += lr * erreur * X[i]
                    b += lr * erreur
                    erreurs_epoque += 1

            # Arrêt anticipé : une epoch complète sans correction signifie que tous
            # les exemples sont bien classés (garanti si les données sont linéairement séparables)
            if erreurs_epoque == 0:
                print(f"Convergence atteinte à l'époque {epoch + 1} !")
                break

        self.w = w
        self.b = b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.b is None:
            raise ValueError("Il faut entraîner le modèle avec .fit() avant de prédire.")
        z = np.dot(X, self.w) + self.b
        return self.activation_function(z)

    def __str__(self) -> str:
        weights_list = self.w.tolist() if self.w is not None else "Non entraîné"
        return (
            f"Perceptron(learning_rate={self.lr}, "
            f"epochs={self.epochs}, "
            f"bias={self.b}, weights={weights_list})"
        )