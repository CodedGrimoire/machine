"""Main script for binary classification ML lab on diabetes dataset."""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from data_utils import (
    load_diabetes_data,
    split_features_target,
    standardize_train_val,
)
from experiments import (
    run_perceptron_experiment,
    run_logistic_regression_experiment,
    run_naive_bayes_experiment,
)


def _ensure_binary_zero_one(y):
    """Ensure labels are binary and encoded as 0/1."""
    unique_vals = sorted(set(y.tolist()))
    if len(unique_vals) != 2:
        raise ValueError("Binary classification requires exactly 2 label classes.")
    if unique_vals == [0, 1]:
        return y.astype(int)
    return (y == unique_vals[1]).astype(int)


def _print_comparison_table(summary_metrics):
    """Print a neat model comparison table."""
    print("\n" + "=" * 76)
    print("\nFinal Validation Metrics Comparison")
    print(f"{'Model':<34}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("-" * 76)
    for model_name, metrics in summary_metrics.items():
        print(
            f"{model_name:<34}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['f1_score']:>10.4f}"
        )


def _plot_comparison_chart(summary_metrics):
    """Plot grouped bar chart for final model metric comparison."""
    model_names = list(summary_metrics.keys())
    accuracy_list = [summary_metrics[name]["accuracy"] for name in model_names]
    precision_list = [summary_metrics[name]["precision"] for name in model_names]
    recall_list = [summary_metrics[name]["recall"] for name in model_names]
    f1_list = [summary_metrics[name]["f1_score"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_acc = ax.bar(x - 1.5 * width, accuracy_list, width, label="Accuracy")
    bars_pre = ax.bar(x - 0.5 * width, precision_list, width, label="Precision")
    bars_rec = ax.bar(x + 0.5 * width, recall_list, width, label="Recall")
    bars_f1 = ax.bar(x + 1.5 * width, f1_list, width, label="F1")

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar_group in (bars_acc, bars_pre, bars_rec, bars_f1):
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    plt.show()


def run_all_models(
    data,
    test_size=0.2,
    random_state=42,
    standardize=True,
    learning_rate=0.01,
    n_iters=None,
):
    """Run Perceptron, Logistic Regression, and Naive Bayes with one split."""
    if n_iters is None:
        n_iters = {
            "perceptron": 100,
            "logistic_regression": 1000,
        }

    # 1. Split into features and target
    X, y = split_features_target(data)
    y = _ensure_binary_zero_one(y)

    # 2. Single-fold train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3. Optional standardization using only train stats
    if standardize:
        X_train_ready, X_val_ready = standardize_train_val(X_train, X_val)
    else:
        X_train_ready, X_val_ready = X_train, X_val

    print("\n[1/4] Running Perceptron...")
    perceptron_model, perceptron_metrics = run_perceptron_experiment(
        X_train=X_train_ready,
        X_val=X_val_ready,
        y_train=y_train,
        y_val=y_val,
        learning_rate=learning_rate,
        n_iters=n_iters["perceptron"],
        random_state=random_state,
    )

    print("\n[2/4] Running Logistic Regression...")
    logistic_model, logistic_metrics = run_logistic_regression_experiment(
        X_train=X_train_ready,
        X_val=X_val_ready,
        y_train=y_train,
        y_val=y_val,
        learning_rate=learning_rate,
        n_iters=n_iters["logistic_regression"],
        random_state=random_state,
    )

    print("\n[3/4] Running Gaussian Naive Bayes...")
    nb_model, nb_metrics, nb_val_probs = run_naive_bayes_experiment(
        X_train=X_train_ready,
        X_val=X_val_ready,
        y_train=y_train,
        y_val=y_val,
    )

    summary_metrics = {
        "Perceptron (From Scratch)": perceptron_metrics,
        "Logistic Regression (From Scratch)": logistic_metrics,
        "Gaussian Naive Bayes": nb_metrics,
    }

    print("\n[4/4] Building final comparison summary...")
    _print_comparison_table(summary_metrics)
    _plot_comparison_chart(summary_metrics)

    return {
        "summary_metrics": summary_metrics,
        "models": {
            "perceptron": perceptron_model,
            "logistic_regression": logistic_model,
            "naive_bayes": nb_model,
        },
        "naive_bayes_val_probs": nb_val_probs,
        "split_data": {
            "X_train": X_train_ready,
            "X_val": X_val_ready,
            "y_train": y_train,
            "y_val": y_val,
        },
    }


def main():
    # Load data and run all models through one clean pipeline.
    data = load_diabetes_data()

    # =========================
    # Configuration
    # =========================
    test_size = 0.2
    random_state = 42
    learning_rate = 0.01
    n_iters = {
        "perceptron": 100,
        "logistic_regression": 1000,
    }

    results = run_all_models(
        data=data,
        test_size=test_size,
        random_state=random_state,
        standardize=True,
        learning_rate=learning_rate,
        n_iters=n_iters,
    )

    _ = results


if __name__ == "__main__":
    # Final submission filename format: <roll>_<name>.<ext>
    main()
