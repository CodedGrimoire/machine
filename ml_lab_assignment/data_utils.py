"""Data loading and preprocessing utilities for the diabetes ML lab."""

import numpy as np
import pandas as pd


def load_diabetes_data():
    """Load the Pima Indians Diabetes dataset as a pandas DataFrame."""
    candidate_paths = [
        "diabetes.csv",
        "pima-indians-diabetes.csv",
        "pima_diabetes.csv",
    ]

    for path in candidate_paths:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Dataset file not found. Place one of these files in the project root: "
        "diabetes.csv, pima-indians-diabetes.csv, or pima_diabetes.csv"
    )


def split_features_target(data):
    """Split DataFrame into features matrix X and target vector y."""
    X = data.iloc[:, :-1].to_numpy(dtype=float)
    y = data.iloc[:, -1].to_numpy(dtype=int)
    return X, y


def standardize_train_val(X_train, X_val):
    """Standardize using only training mean and std."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)

    X_train_std = (X_train - mean) / std_safe
    X_val_std = (X_val - mean) / std_safe
    return X_train_std, X_val_std


def add_bias_column(X):
    """Add a leading bias column of ones."""
    bias = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack((bias, X))
