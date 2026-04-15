"""Logistic Regression model implementation from scratch."""

import numpy as np

from metrics_utils import sigmoid, log_loss_binary, misclassification_error


class LogisticRegressionFromScratch:
    """Binary Logistic Regression trained with vectorized gradient descent."""

    def __init__(self, learning_rate=0.08, n_iters=10000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None

        self.train_misclassification_history_ = []
        self.val_misclassification_history_ = []
        self.train_log_loss_history_ = []
        self.val_log_loss_history_ = []

    def predict_proba(self, X):
        """Return predicted probabilities for class 1."""
        linear_scores = np.dot(X, self.weights_) + self.bias_
        return sigmoid(linear_scores)

    def predict(self, X, threshold=0.5):
        """Return binary predictions (0/1)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train model with one loop over iterations and vectorized updates."""
        n_samples, n_features = X_train.shape
        rng = np.random.default_rng(self.random_state)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0

        self.train_misclassification_history_ = []
        self.val_misclassification_history_ = []
        self.train_log_loss_history_ = []
        self.val_log_loss_history_ = []

        for _ in range(self.n_iters):
            train_probs = sigmoid(np.dot(X_train, self.weights_) + self.bias_)
            train_errors = train_probs - y_train

            grad_w = np.dot(X_train.T, train_errors) / n_samples
            grad_b = np.mean(train_errors)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            updated_train_probs = sigmoid(np.dot(X_train, self.weights_) + self.bias_)
            self.train_misclassification_history_.append(
                misclassification_error(y_train, updated_train_probs)
            )
            self.train_log_loss_history_.append(
                log_loss_binary(y_train, updated_train_probs)
            )

            if X_val is not None and y_val is not None:
                val_probs = sigmoid(np.dot(X_val, self.weights_) + self.bias_)
                self.val_misclassification_history_.append(
                    misclassification_error(y_val, val_probs)
                )
                self.val_log_loss_history_.append(log_loss_binary(y_val, val_probs))

        return self
