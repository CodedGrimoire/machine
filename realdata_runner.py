import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import process_data, normal_equation, train_gradient_descent


def load_real_data(file_path):
    data = pd.read_csv(file_path, header=None)

    # As you confirmed: col0 = x, col1 = y
    x = data.iloc[:, 0].to_numpy(dtype=np.float64)
    y = data.iloc[:, 1].to_numpy(dtype=np.float64)

    # Sort by x (clean plots + consistent)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    corr = pd.Series(x).corr(pd.Series(y))
    print("Correlation:", corr)

    # Quick sanity preview
    print("First 5 (x, y):")
    for i in range(5):
        print(f"  {x[i]:.4f}, {y[i]:.4f}")

    return x, y


if __name__ == "__main__":
    x, y = load_real_data("data_01.csv")

    X, y_col = process_data(x, y)

    # ----- Ideal theta (closed-form) -----
    theta_ne = normal_equation(X, y_col)
    print("\nNormal Equation theta (ideal):")
    print(theta_ne)

    # ----- Gradient Descent theta -----
    theta_gd, cost_history, alpha_used = train_gradient_descent(
        X, y_col,
        alpha=0.002,          
        iterations=100000,
        print_every=8000
    )

    print("\nGradient Descent alpha used:", alpha_used)
    print("Gradient Descent theta:")
    print(theta_gd)

    # ----- Plot cost history (sampled) -----
    plt.figure()
    plt.plot(np.arange(len(cost_history)) * 50 + 1, cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training Error Curve (Real Data)")
    plt.grid(True)
    plt.show()

    # ----- Plot regression line (use GD or NE; both should overlap) -----
    y_pred_gd = (X @ theta_gd).reshape(-1)
    y_pred_ne = (X @ theta_ne).reshape(-1)

    plt.figure()
    plt.scatter(x, y, s=10, label="Data Points")
    plt.plot(x, y_pred_gd, linewidth=2, label="Regression Line (GD)")
    plt.plot(x, y_pred_ne, linewidth=2, linestyle="--", label="Regression Line (Normal Eq)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit (Real Data)")
    plt.legend()
    plt.grid(True)
    plt.show()