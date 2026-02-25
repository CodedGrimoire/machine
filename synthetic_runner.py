# synthetic_runner.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import process_data, train, evaluate


def generate_synthetic_data():
    """
    Generates synthetic data:
    y = 3 + 5x + Gaussian noise
    """
    np.random.seed(42)

    x = np.arange(1, 101)  # 1 to 100
    noise = np.random.normal(0, 1, 100)
    y = 3 + 5 * x + noise

    # Save for future use
    data = pd.DataFrame({'x': x, 'y': y})
    data.to_csv("lab01_data.csv", index=False)

    return x, y


if __name__ == "__main__":

    # ---------------------------
    # 1. Generate Data
    # ---------------------------
    x, y = generate_synthetic_data()

    # ---------------------------
    # 2. Plot ONLY Data Points (as required by lab)
    # ---------------------------
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Data Points")
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 3. Process Data
    # ---------------------------
    X, y = process_data(x, y, feature_scaling=False)

    # ---------------------------
    # 4. Train Model
    # ---------------------------
    theta, cost_history = train(
        X,
        y,
        alpha=0.0001,
        iterations=1000000
    )

    # Evaluate model
    evaluate(X, y, theta)

    # ---------------------------
    # 5. Plot Training Error Curve
    # ---------------------------
    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training Error Curve (Synthetic Data)")
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 6. Plot Regression Line
    # ---------------------------
    plt.figure()
    plt.scatter(x, y, label="Data Points")

    predictions = X @ theta
    plt.plot(x, predictions, color='red', label="Regression Line")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit (Synthetic Data)")
    plt.legend()
    plt.grid(True)
    plt.show()