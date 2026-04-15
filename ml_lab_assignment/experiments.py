"""Experiment helpers for model training, evaluation, and plotting."""

from metrics_utils import evaluate_classification
from plot_utils import plot_single_curve, plot_two_curves
from perceptron_model import PerceptronFromScratch
from logistic_regression_model import LogisticRegressionFromScratch


def _print_metrics(title, metrics):
    print(f"\n{title} - Validation Metrics")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")


def run_perceptron_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    learning_rate,
    n_iters,
):
    """Train, evaluate, print metrics, and plot Perceptron misclassification curves."""
    model = PerceptronFromScratch(
        learning_rate=learning_rate,
        n_iters=n_iters,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_val_pred = model.predict(X_val)
    metrics = evaluate_classification(y_val, y_val_pred)
    _print_metrics("Perceptron (From Scratch)", metrics)

    if model.val_misclassification_history_:
        plot_two_curves(
            model.train_misclassification_history_,
            model.val_misclassification_history_,
            xlabel="Iteration",
            ylabel="Misclassification Rate",
            title="Perceptron Misclassification vs Iteration",
            label1="Train Misclassification",
            label2="Validation Misclassification",
        )
    else:
        plot_single_curve(
            model.train_misclassification_history_,
            xlabel="Iteration",
            ylabel="Misclassification Rate",
            title="Perceptron Train Misclassification vs Iteration",
        )

    return model, metrics


def run_logistic_regression_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    learning_rate,
    n_iters,
):
    """Train, evaluate, print metrics, and plot Logistic Regression curves."""
    model = LogisticRegressionFromScratch(
        learning_rate=learning_rate,
        n_iters=n_iters,
        random_state=42,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_val_pred = model.predict(X_val)
    metrics = evaluate_classification(y_val, y_val_pred)
    _print_metrics("Logistic Regression (From Scratch)", metrics)

    if model.val_misclassification_history_:
        plot_two_curves(
            model.train_misclassification_history_,
            model.val_misclassification_history_,
            xlabel="Iteration",
            ylabel="Misclassification Rate",
            title="Logistic Regression Misclassification vs Iteration",
            label1="Train Misclassification",
            label2="Validation Misclassification",
        )
    else:
        plot_single_curve(
            model.train_misclassification_history_,
            xlabel="Iteration",
            ylabel="Misclassification Rate",
            title="Logistic Regression Train Misclassification vs Iteration",
        )

    if model.val_log_loss_history_:
        plot_two_curves(
            model.train_log_loss_history_,
            model.val_log_loss_history_,
            xlabel="Iteration",
            ylabel="Log Loss",
            title="Logistic Regression Log Loss vs Iteration",
            label1="Train Log Loss",
            label2="Validation Log Loss",
        )
    else:
        plot_single_curve(
            model.train_log_loss_history_,
            xlabel="Iteration",
            ylabel="Log Loss",
            title="Logistic Regression Train Log Loss vs Iteration",
        )

    return model, metrics
