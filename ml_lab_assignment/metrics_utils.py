"""Metrics and loss helpers for binary classification."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def sigmoid(z):
    """Compute sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))


def log_loss_binary(y_true, y_pred_probs):
    """Binary log loss with clipping for numerical stability."""
    eps = 1e-15
    p = np.clip(y_pred_probs, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def misclassification_error(y_true, y_pred_probs, threshold=0.5):
    """Misclassification rate from predicted probabilities."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    return np.mean(y_pred != y_true)


def evaluate_classification(y_true, y_pred_labels):
    """Return confusion matrix and common binary metrics."""
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred_labels),
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "precision": precision_score(y_true, y_pred_labels, zero_division=0),
        "recall": recall_score(y_true, y_pred_labels, zero_division=0),
        "f1_score": f1_score(y_true, y_pred_labels, zero_division=0),
    }


def select_best_threshold_by_accuracy(y_true, scores_or_probs, candidate_thresholds):
    """Select threshold that maximizes accuracy and return full metric payload."""
    best_result = None
    best_accuracy = float("-inf")

    for threshold in candidate_thresholds:
        y_pred = (scores_or_probs >= threshold).astype(int)
        metrics = evaluate_classification(y_true, y_pred)
        accuracy = metrics["accuracy"]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_result = {
                "threshold": float(threshold),
                "confusion_matrix": metrics["confusion_matrix"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }

    return best_result
