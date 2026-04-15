"""Main script for binary classification ML lab on diabetes dataset."""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from data_utils import (
    load_diabetes_data,
    split_features_target,
    standardize_train_val,
    add_bias_column,
)
from experiments import (
    run_perceptron_experiment,
    run_logistic_regression_experiment,
)


def main():
    # 1. Load data
    data = load_diabetes_data()

    # 2. Split into features and target
    X, y = split_features_target(data)

    # 3. Single-fold train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Standardize inputs using train stats only
    X_train_std, X_val_std = standardize_train_val(X_train, X_val)

    # 5. Optional bias-expanded versions for custom linear models
    X_train_bias = add_bias_column(X_train_std)
    X_val_bias = add_bias_column(X_val_std)

    # 6. Perceptron experiment
    perceptron_model, perceptron_metrics = run_perceptron_experiment(
        X_train=X_train_std,
        X_val=X_val_std,
        y_train=y_train,
        y_val=y_val,
        learning_rate=0.01,
        n_iters=100,
    )

    # 7. Logistic Regression experiment
    logistic_model, logistic_metrics = run_logistic_regression_experiment(
        X_train=X_train_std,
        X_val=X_val_std,
        y_train=y_train,
        y_val=y_val,
        learning_rate=0.01,
        n_iters=1000,
    )

    # 8. TODO: Train and evaluate Gaussian Naive Bayes (library allowed)
    # nb_model = GaussianNB()
    # nb_model.fit(X_train_std, y_train)
    # nb_val_preds = nb_model.predict(X_val_std)

    _ = (
        X_train_bias,
        X_val_bias,
        perceptron_model,
        perceptron_metrics,
        logistic_model,
        logistic_metrics,
        GaussianNB,
    )


if __name__ == "__main__":
    main()
