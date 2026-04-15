"""Experiment helpers for model training, evaluation, and plotting."""

import numpy as np
from sklearn.naive_bayes import GaussianNB

from metrics_utils import select_best_threshold_by_accuracy
from plot_utils import plot_single_curve, plot_two_curves
from perceptron_model import PerceptronFromScratch
from logistic_regression_model import LogisticRegressionFromScratch


def _print_metrics(title, metrics, threshold=None):
    print("\n" + "=" * 76)
    print(f"{title} - Validation Results")
    print("=" * 76)
    if threshold is not None:
        print(f"Chosen Threshold: {threshold:.2f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")


def _plot_history(train_values, val_values, xlabel, ylabel, title, train_label, val_label):
    """Plot train-only or train-vs-validation curves."""
    if val_values:
        plot_two_curves(
            train_values,
            val_values,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            label1=train_label,
            label2=val_label,
        )
    else:
        plot_single_curve(
            train_values,
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{title} (Train)",
        )


def run_perceptron_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    learning_rate,
    n_iters,
    random_state=42,
):
    """Train, evaluate, print metrics, and plot Perceptron misclassification curves."""
    model = PerceptronFromScratch(
        learning_rate=learning_rate,
        n_iters=n_iters,
        random_state=random_state,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    val_scores = model.decision_function(X_val)
    candidate_thresholds = np.linspace(-0.5, 0.5, 41)
    tuned_metrics = select_best_threshold_by_accuracy(
        y_true=y_val,
        scores_or_probs=val_scores,
        candidate_thresholds=candidate_thresholds,
    )
    threshold = tuned_metrics.pop("threshold")

    _print_metrics("Perceptron (From Scratch)", tuned_metrics, threshold=threshold)

    _plot_history(
        train_values=model.train_misclassification_history_,
        val_values=model.val_misclassification_history_,
        xlabel="Iteration",
        ylabel="Misclassification Rate",
        title="Perceptron Misclassification vs Iteration",
        train_label="Train Misclassification",
        val_label="Validation Misclassification",
    )

    return model, tuned_metrics


def run_logistic_regression_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    learning_rate,
    n_iters,
    random_state=42,
):
    """Train, evaluate, print metrics, and plot Logistic Regression curves."""
    model = LogisticRegressionFromScratch(
        learning_rate=learning_rate,
        n_iters=n_iters,
        random_state=random_state,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_val_probs = model.predict_proba(X_val)
    candidate_thresholds = np.linspace(0.30, 0.70, 41)
    tuned_metrics = select_best_threshold_by_accuracy(
        y_true=y_val,
        scores_or_probs=y_val_probs,
        candidate_thresholds=candidate_thresholds,
    )
    threshold = tuned_metrics.pop("threshold")

    _print_metrics("Logistic Regression (From Scratch)", tuned_metrics, threshold=threshold)

    _plot_history(
        train_values=model.train_misclassification_history_,
        val_values=model.val_misclassification_history_,
        xlabel="Iteration",
        ylabel="Misclassification Rate",
        title="Logistic Regression Misclassification vs Iteration",
        train_label="Train Misclassification",
        val_label="Validation Misclassification",
    )

    _plot_history(
        train_values=model.train_log_loss_history_,
        val_values=model.val_log_loss_history_,
        xlabel="Iteration",
        ylabel="Log Loss",
        title="Logistic Regression Log Loss vs Iteration",
        train_label="Train Log Loss",
        val_label="Validation Log Loss",
    )

    return model, tuned_metrics


def run_naive_bayes_experiment(X_train, X_val, y_train, y_val):
    """Train and evaluate Gaussian Naive Bayes on validation data."""
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_val_probs = model.predict_proba(X_val)[:, 1]
    candidate_thresholds = np.linspace(0.30, 0.70, 41)
    tuned_metrics = select_best_threshold_by_accuracy(
        y_true=y_val,
        scores_or_probs=y_val_probs,
        candidate_thresholds=candidate_thresholds,
    )
    threshold = tuned_metrics.pop("threshold")

    _print_metrics("Gaussian Naive Bayes", tuned_metrics, threshold=threshold)

    return model, tuned_metrics, y_val_probs
