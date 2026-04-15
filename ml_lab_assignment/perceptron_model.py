"""Perceptron model implementation from scratch."""

import numpy as np


class PerceptronFromScratch:
    """Binary Perceptron classifier implemented from scratch."""

    def __init__(self, learning_rate=0.01, n_iters=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.train_misclassification_history_ = []
        self.val_misclassification_history_ = []

    def decision_function(self, X):
        """Return raw linear scores before thresholding."""
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        """Predict binary labels (0/1)."""
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train Perceptron and store misclassification history per iteration."""
        n_samples, n_features = X_train.shape
        rng = np.random.default_rng(self.random_state)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.train_misclassification_history_ = []
        self.val_misclassification_history_ = []

        y_train_pm = np.where(y_train == 1, 1, -1)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = X_train[i]
                y_i = y_train_pm[i]

                score = np.dot(x_i, self.weights_) + self.bias_
                y_hat = 1 if score >= 0.0 else -1

                if y_hat != y_i:
                    update = self.learning_rate * y_i
                    self.weights_ += update * x_i
                    self.bias_ += update

            train_preds = self.predict(X_train)
            self.train_misclassification_history_.append(np.mean(train_preds != y_train))

            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                self.val_misclassification_history_.append(np.mean(val_preds != y_val))

        return self
