import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# 1. Load Data
# ==============================
def load_data(file_path=None, synthetic=True):
    """
    Loads data.
    If synthetic=True, generates y = 3 + 5x + noise.
    Otherwise loads from CSV file.
    """
    if synthetic:
        np.random.seed(42)
        noise = np.random.normal(0, 1, 100)
        y = 3 + 5 * x + noise

        data = pd.DataFrame({'x': x, 'y': y})
        data.to_csv("lab01_data.csv", index=False)

        return data
    else:
        data = pd.read_csv(file_path)
        return data


# ==============================
# 2. Process Data
# ==============================
def process_data(data, feature_scaling=False):
    """
    Adds dummy feature x0=1.
    Optionally applies feature scaling.
    Returns X matrix and y vector.
    """
    X = data[['x']].values
    y = data[['y']].values

    if feature_scaling:
        X = (X - np.mean(X)) / np.std(X)

    # Add dummy feature (bias term)
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    return X, y


# ==============================
# 3. Compute Cost
# ==============================
def compute_cost(X, y, theta):
    """
    Computes Mean Squared Error cost.
    """
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# ==============================
# 4. Gradient Descent
# ==============================
def gradient_descent(X, y, theta, alpha, iterations):
    """
    Performs gradient descent.
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = X @ theta
        gradient = (1 / m) * (X.T @ (predictions - y))
        theta = theta - alpha * gradient

        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


# ==============================
# 5. Train Model
# ==============================
def train(X, y, alpha=0.0001, iterations=1000):
    """
    Trains linear regression model.
    """
    theta = np.zeros((X.shape[1], 1))
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    return theta, cost_history


# ==============================
# 6. Evaluate Model
# ==============================
def evaluate(X, y, theta):
    """
    Evaluates model performance.
    """
    final_cost = compute_cost(X, y, theta)
    print("Final Cost:", final_cost)
    print("Learned Parameters (theta):")
    print(theta)
    return final_cost


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":

    # ---- Synthetic Data ----
    data = load_data(synthetic=True)
    X, y = process_data(data, feature_scaling=False)

    theta, cost_history = train(X, y, alpha=0.0001, iterations=2000)

    evaluate(X, y, theta)

    # ---- Plot Training Error Curve ----
    plt.figure()
    plt.plot(cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training Error Curve")
    plt.show()

    # ---- Plot Regression Line ----
    plt.figure()
    plt.scatter(data['x'], data['y'], label="Data Points")
    plt.plot(data['x'], (X @ theta), color='red', label="Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()

    # ---- Real Data (if needed) ----
    # real_data = load_data(file_path="data_01.csv", synthetic=False)
    # X_real, y_real = process_data(real_data)
    # theta_real, cost_real = train(X_real, y_real)
    # evaluate(X_real, y_real, theta_real)