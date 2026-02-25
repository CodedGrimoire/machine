import numpy as np


# --------------------------------------------------
# 1. Data Processing
# --------------------------------------------------
def process_data(X_raw, y_raw, feature_scaling=False):

    X = X_raw.reshape(-1, 1)
    y = y_raw.reshape(-1, 1)

    if feature_scaling:
        mean = np.mean(X)
        std = np.std(X)
        X = (X - mean) / std

    # Add bias column
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    return X, y


# --------------------------------------------------
# 2. Cost Function
# --------------------------------------------------
def compute_cost(X, y, theta):

    m = len(y)
    predictions = X @ theta
    error = predictions - y

    cost = (1 / (2 * m)) * np.sum(error ** 2)

    return cost


# --------------------------------------------------
# 3. Gradient Descent
# --------------------------------------------------
def gradient_descent(X, y, theta, alpha, iterations):

    m = len(y)
    cost_history = []

    for i in range(iterations):

        predictions = X @ theta
        error = predictions - y

        gradient = (1 / m) * (X.T @ error)

        theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


# --------------------------------------------------
# 4. Train Wrapper
# --------------------------------------------------
def train(X, y, alpha=0.0001, iterations=2000):

    theta = np.zeros((X.shape[1], 1))

    theta, cost_history = gradient_descent(
        X, y, theta, alpha, iterations
    )

    return theta, cost_history


# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------
def evaluate(X, y, theta):

    final_cost = compute_cost(X, y, theta)

    print("\nFinal Cost:", final_cost)
    print("Learned Parameters (theta):")
    print(theta)

    return final_cost

def normal_equation(X, y):
    """
    Computes theta using the closed-form normal equation:
    theta = (X^T X)^(-1) X^T y
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y
def train_gradient_descent(X, y, alpha=0.001, iterations=20000, print_every=None):
    """
    Gradient Descent with optional progress printing.
    Returns:
        theta
        cost_history
        alpha_used
    """

    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []

    # If alpha is None, pick a safe default
    if alpha is None:
        alpha = 0.001

    for i in range(1, iterations + 1):

        predictions = X @ theta
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient

        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

        if print_every and i % print_every == 0:
            print(f"iter {i:6d} | cost = {cost:.6f}")

    return theta, cost_history, alpha