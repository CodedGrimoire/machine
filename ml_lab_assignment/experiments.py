"""Experiment helpers for model training, evaluation, and plotting."""

from sklearn.naive_bayes import GaussianNB

from metrics_utils import evaluate_classification
from plot_utils import plot_single_curve, plot_two_curves
from perceptron_model import PerceptronFromScratch
from logistic_regression_model import LogisticRegressionFromScratch


def _print_metrics(title, metrics):
    print("\n" + "=" * 76)
    print(f"{title} - Validation Results")
    print("=" * 76)
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


def _select_naive_bayes_threshold(y_true, y_pred_probs, thresholds, recall_target=0.70):
    """Select threshold: first meeting recall target, else highest recall."""
    best_metrics = None
    best_threshold = None

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        metrics = evaluate_classification(y_true, y_pred)
        if metrics["recall"] >= recall_target:
            return threshold, metrics

        if best_metrics is None or metrics["recall"] > best_metrics["recall"]:
            best_metrics = metrics
            best_threshold = threshold

    return best_threshold, best_metrics


def _select_perceptron_threshold(y_true, scores, thresholds):
    """Select threshold using filtered candidates and F1 maximization."""
    filtered_candidates = []
    all_candidates = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metrics = evaluate_classification(y_true, y_pred)
        all_candidates.append((threshold, metrics))

        # Filter out obviously degenerate behavior before selecting by F1.
        if metrics["precision"] >= 0.45 and metrics["accuracy"] >= 0.60:
            filtered_candidates.append((threshold, metrics))

    candidate_pool = filtered_candidates if filtered_candidates else all_candidates
    return max(candidate_pool, key=lambda item: item[1]["f1_score"])


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
    candidate_thresholds = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25]
    chosen_threshold, metrics = _select_perceptron_threshold(
        y_true=y_val,
        scores=val_scores,
        thresholds=candidate_thresholds,
    )

    print("\n" + "=" * 76)
    print("Perceptron (From Scratch) - Validation Results")
    print("=" * 76)
    print(f"Chosen Threshold: {chosen_threshold:.2f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    _plot_history(
        train_values=model.train_misclassification_history_,
        val_values=model.val_misclassification_history_,
        xlabel="Iteration",
        ylabel="Misclassification Rate",
        title="Perceptron Misclassification vs Iteration",
        train_label="Train Misclassification",
        val_label="Validation Misclassification",
    )

    return model, metrics


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
    candidate_thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
    chosen_threshold = None
    metrics = None

    for threshold in candidate_thresholds:
        y_val_pred = (y_val_probs >= threshold).astype(int)
        current_metrics = evaluate_classification(y_val, y_val_pred)
        if current_metrics["recall"] >= 0.70:
            chosen_threshold = threshold
            metrics = current_metrics
            break
        if metrics is None or current_metrics["recall"] > metrics["recall"]:
            chosen_threshold = threshold
            metrics = current_metrics

    print("\n" + "=" * 76)
    print("Logistic Regression (From Scratch) - Validation Results")
    print("=" * 76)
    print(f"Chosen Threshold: {chosen_threshold:.2f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

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

    return model, metrics


def run_naive_bayes_experiment(X_train, X_val, y_train, y_val):
    """Train and evaluate Gaussian Naive Bayes on validation data."""
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_val_probs = model.predict_proba(X_val)[:, 1]
    candidate_thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
    chosen_threshold, metrics = _select_naive_bayes_threshold(
        y_true=y_val,
        y_pred_probs=y_val_probs,
        thresholds=candidate_thresholds,
        recall_target=0.70,
    )

    print("\n" + "=" * 76)
    print("Gaussian Naive Bayes - Validation Results")
    print("=" * 76)
    print(f"Chosen Threshold: {chosen_threshold:.2f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    return model, metrics, y_val_probs
