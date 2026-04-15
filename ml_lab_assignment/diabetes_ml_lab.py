"""
ML Lab Skeleton: Binary Classification on Pima Indians Diabetes Dataset

This file provides reusable data, preprocessing, metrics, and plotting utilities.
Model implementations are intentionally left as TODOs for later steps.
"""

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB


# =========================
# Data Loading
# =========================
def load_diabetes_data():
    """Load the Pima Indians Diabetes dataset as a pandas DataFrame.

    Expected local filename: diabetes.csv
    """
    candidate_paths = [
        "diabetes.csv",
        "pima-indians-diabetes.csv",
        "pima_diabetes.csv",
    ]

    for path in candidate_paths:
        try:
            data = pd.read_csv(path)
            return data
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Dataset file not found. Place one of these files in the project root: "
        "diabetes.csv, pima-indians-diabetes.csv, or pima_diabetes.csv"
    )


# =========================
# Preprocessing Utilities
# =========================
def split_features_target(data):
    """Split DataFrame into features matrix X and target vector y.

    Assumes the target is in the last column.
    """
    X = data.iloc[:, :-1].to_numpy(dtype=float)
    y = data.iloc[:, -1].to_numpy(dtype=int)
    return X, y


def standardize_train_val(X_train, X_val):
    """Standardize train and validation data using train-set mean/std only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Avoid divide-by-zero for constant columns.
    std_safe = np.where(std == 0, 1.0, std)

    X_train_std = (X_train - mean) / std_safe
    X_val_std = (X_val - mean) / std_safe
    return X_train_std, X_val_std


def add_bias_column(X):
    """Add a leading bias (ones) column to feature matrix X."""
    bias = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack((bias, X))


# =========================
# Metric and Loss Helpers
# =========================
def sigmoid(z):
    """Compute sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))


def log_loss_binary(y_true, y_pred_probs):
    """Compute binary log loss with probability clipping for stability."""
    eps = 1e-12
    p = np.clip(y_pred_probs, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def misclassification_error(y_true, y_pred_probs, threshold=0.5):
    """Compute misclassification rate from predicted probabilities."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_pred != y_true)


def evaluate_classification(y_true, y_pred_labels):
    """Return a dictionary of standard binary classification metrics."""
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred_labels),
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "precision": precision_score(y_true, y_pred_labels, zero_division=0),
        "recall": recall_score(y_true, y_pred_labels, zero_division=0),
        "f1_score": f1_score(y_true, y_pred_labels, zero_division=0),
    }


# =========================
# Plotting Helpers
# =========================
def plot_single_curve(values, xlabel, ylabel, title):
    """Plot a single line curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_two_curves(train_values, val_values, xlabel, ylabel, title, label1, label2):
    """Plot two curves for train/validation comparison."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=label1)
    plt.plot(val_values, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================
# Perceptron (From Scratch)
# =========================
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

        # Convert labels from {0,1} to {-1,+1} for perceptron updates.
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
            train_error = np.mean(train_preds != y_train)
            self.train_misclassification_history_.append(train_error)

            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                val_error = np.mean(val_preds != y_val)
                self.val_misclassification_history_.append(val_error)

        return self


def run_perceptron_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    learning_rate,
    n_iters,
):
    """Train, evaluate, print metrics, and plot Perceptron misclassification curves."""
    perceptron_model = PerceptronFromScratch(
        learning_rate=learning_rate,
        n_iters=n_iters,
        random_state=42,
    )
    perceptron_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_val_pred = perceptron_model.predict(X_val)
    metrics = evaluate_classification(y_val, y_val_pred)

    print("\nPerceptron (From Scratch) - Validation Metrics")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    if len(perceptron_model.val_misclassification_history_) > 0:
        plot_two_curves(
            perceptron_model.train_misclassification_history_,
            perceptron_model.val_misclassification_history_,
            xlabel="Iteration",
            ylabel="Misclassification Rate",
            title="Perceptron Misclassification vs Iteration",
            label1="Train Misclassification",
            label2="Validation Misclassification",
        )
    else:
        plot_single_curve(
            perceptron_model.train_misclassification_history_,
            xlabel="Iteration",
            ylabel="Misclassification Rate",
            title="Perceptron Train Misclassification vs Iteration",
        )

    return perceptron_model, metrics


# =========================
# Main Execution Skeleton
# =========================
def main():
    # 1. Load data
    data = load_diabetes_data()

    # 2. Split into features and target
    X, y = split_features_target(data)

    # 3. Train/validation split (single fold)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Standardize inputs
    X_train_std, X_val_std = standardize_train_val(X_train, X_val)

    # 5. Optional bias-expanded versions for custom linear models
    X_train_bias = add_bias_column(X_train_std)
    X_val_bias = add_bias_column(X_val_std)

    # 6. Train and evaluate Perceptron from scratch
    perceptron_model, perceptron_metrics = run_perceptron_experiment(
        X_train=X_train_std,
        X_val=X_val_std,
        y_train=y_train,
        y_val=y_val,
        learning_rate=0.01,
        n_iters=100,
    )

    # 7. TODO: Train and evaluate Logistic Regression from scratch
    # logistic_model = ...
    # logistic_val_probs = ...
    # logistic_val_preds = ...
    # logistic_metrics = evaluate_classification(y_val, logistic_val_preds)

    # 8. TODO: Train and evaluate Gaussian Naive Bayes (library allowed)
    # nb_model = GaussianNB()
    # nb_model.fit(X_train_std, y_train)
    # nb_val_preds = nb_model.predict(X_val_std)
    # nb_metrics = evaluate_classification(y_val, nb_val_preds)

    # 9. TODO: Plot learning curves / error curves when model training histories are available

    # Keep these variables referenced to avoid accidental removal during refactors.
    _ = (
        X_train_bias,
        X_val_bias,
        y_train,
        y_val,
        GaussianNB,
        perceptron_model,
        perceptron_metrics,
    )


if __name__ == "__main__":
    main()
