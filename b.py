import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# print numbers normally
np.set_printoptions(suppress=True, precision=4)

# -----------------------------
# Load dataset
# -----------------------------
def load_dataset(path):

    data = pd.read_excel(path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1,1)

    return X,y


# -----------------------------
# Normalize
# -----------------------------
def normalize(X):

    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    return (X-mean)/std


# -----------------------------
# Cost Function
# -----------------------------
def compute_cost(X,y,theta):

    m=len(y)

    predictions = X @ theta
    error = predictions - y

    return (1/(2*m))*np.sum(error**2)


# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(X_train,y_train,X_val,y_val,theta,lr,iterations):

    m=len(y_train)

    train_errors=[]
    val_errors=[]

    for i in range(iterations):

        gradient = (1/m) * X_train.T @ (X_train @ theta - y_train)

        theta = theta - lr*gradient

        train_errors.append(compute_cost(X_train,y_train,theta))
        val_errors.append(compute_cost(X_val,y_val,theta))

    return theta,train_errors,val_errors


# -----------------------------
# Cross Validation
# -----------------------------
def cross_validation(X,y,k=5):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_error=float("inf")
    best_theta=None

    train_curves=[]
    val_curves=[]

    for train_idx,val_idx in kf.split(X):

        X_train,X_val = X[train_idx],X[val_idx]
        y_train,y_val = y[train_idx],y[val_idx]

        theta = np.zeros((X.shape[1],1))

        theta,train_curve,val_curve = gradient_descent(
            X_train,y_train,X_val,y_val,theta,0.01,10000
        )

        val_error = compute_cost(X_val,y_val,theta)

        train_curves.append(train_curve)
        val_curves.append(val_curve)

        if val_error < best_error:

            best_error = val_error
            best_theta = theta

    return best_theta,best_error,train_curves,val_curves


# -----------------------------
# MAIN
# -----------------------------

# Load data
X,y = load_dataset("CCPP/Folds5x2_pp.xlsx")

# Normalize features
X = normalize(X)

# Add bias column
X = np.c_[np.ones(len(X)),X]

# Run 5-fold cross validation
theta_cv,error_cv,train_curves,val_curves = cross_validation(X,y)

print("\nBest Parameters from 5-Fold Cross Validation:\n")

for i,val in enumerate(theta_cv):
    print(f"Theta{i}: {val[0]:.4f}")

print(f"\nBest Validation Error: {error_cv:.4f}")


# -----------------------------
# Plot Training Curves
# -----------------------------
plt.figure(figsize=(8,5))

for i,curve in enumerate(train_curves):
    plt.plot(curve,label=f"Train Fold {i+1}")

plt.title("Gradient Descent Training Error (Each Fold)")
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# -----------------------------
# Plot Training vs Validation
# -----------------------------
plt.figure(figsize=(8,5))

for i in range(len(train_curves)):
    plt.plot(train_curves[i], label=f"Train Fold {i+1}")
    plt.plot(val_curves[i], linestyle="--", label=f"Val Fold {i+1}")

plt.title("Training vs Validation Error")
plt.xlabel("Iterations")
plt.ylabel("Cost")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()