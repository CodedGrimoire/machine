import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Print numbers normally instead of scientific notation
np.set_printoptions(suppress=True, precision=4)

# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset(path):

    data = pd.read_excel(path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1,1)

    return X,y


"""
In gradient descent, the model updates its parameters using the feature values.
If different features have very different scales, the updates become unbalanced.
Features with large values dominate the gradient and cause very large parameter
updates, while smaller features barely affect the update. This makes gradient
descent unstable and slow to converge.

Normalization rescales all features to a similar range so that each feature
contributes equally to the gradient. As a result, the optimization becomes
more stable and the algorithm converges faster.
"""


# -----------------------------
# Normalize Features
# -----------------------------
def normalize(X):

    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    return (X-mean)/std


# -----------------------------
# Cost Function
# -----------------------------
def compute_cost(X,y,theta):

    m = len(y)

    predictions = X @ theta
    error = predictions - y

    return (1/(2*m))*np.sum(error**2)


# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(X_train, y_train, X_val, y_val, theta, lr, iterations):

    m = len(y_train)

    train_errors = []
    val_errors = []

    for _ in range(iterations):

        gradient = (1/m) * X_train.T @ (X_train @ theta - y_train)

        theta = theta - lr * gradient

        train_error = compute_cost(X_train, y_train, theta)
        val_error = compute_cost(X_val, y_val, theta)

        # stop if exploding
        if np.isnan(train_error) or np.isinf(train_error):
            break

        train_errors.append(train_error)
        val_errors.append(val_error)

    return theta, train_errors, val_errors


# -----------------------------
# Train Validation Split
# -----------------------------
def split_data(X,y,ratio=0.8):

    m=len(X)
    split=int(m*ratio)

    X_train=X[:split]
    X_val=X[split:]

    y_train=y[:split]
    y_val=y[split:]

    return X_train,X_val,y_train,y_val


# -----------------------------
# Plot Feature vs Target
# -----------------------------
def plot_features(X,y):

    for i in range(X.shape[1]):

        plt.figure()
        plt.scatter(X[:,i],y,s=5)
        plt.xlabel(f"Feature {i+1}")
        plt.ylabel("Target")
        plt.title(f"Feature {i+1} vs Target")
        plt.grid(True)
        plt.show()


# -----------------------------
# Train Model
# -----------------------------
def train_model(X,y):

    X=np.c_[np.ones(len(X)),X]

    X_train,X_val,y_train,y_val = split_data(X,y)

    theta=np.zeros((X.shape[1],1))

    theta,train_errors,val_errors = gradient_descent(
        X_train,y_train,X_val,y_val,theta,0.01,100000
    )

    best_train=min(train_errors)
    best_val=min(val_errors)

    return theta,train_errors,val_errors,best_train,best_val


# -----------------------------
# MAIN
# -----------------------------

X,y = load_dataset("CCPP/Folds5x2_pp.xlsx")


# 1️⃣ Plot features
plot_features(X,y)


# =============================
# WITHOUT NORMALIZATION
# =============================
print("\n---- WITHOUT NORMALIZATION ----")

theta,train_errors,val_errors,best_train,best_val = train_model(X,y)

print(f"Best Training Error: {best_train:.4f}")
print(f"Best Validation Error: {best_val:.4f}")

print("\nLearned Parameters:")
for i, val in enumerate(theta):
    print(f"Theta{i}: {val[0]:.4f}")


plt.figure(figsize=(8,5))

plt.plot(train_errors,label="Training Error")
plt.plot(val_errors,label="Validation Error")

plt.title("Error Curves (Without Normalization)")
plt.xlabel("Iterations")
plt.ylabel("Error")

plt.grid(True)
plt.legend()

plt.show()


# =============================
# WITH NORMALIZATION
# =============================
print("\n---- WITH NORMALIZATION ----")

X_norm = normalize(X)

theta2,train_errors2,val_errors2,best_train2,best_val2 = train_model(X_norm,y)

print(f"Best Training Error: {best_train2:.4f}")
print(f"Best Validation Error: {best_val2:.4f}")

print("\nLearned Parameters:")
for i, val in enumerate(theta2):
    print(f"Theta{i}: {val[0]:.4f}")


plt.figure(figsize=(8,5))

plt.plot(train_errors2,label="Training Error")
plt.plot(val_errors2,label="Validation Error")

plt.title("Error Curves (With Normalization)")
plt.xlabel("Iterations")
plt.ylabel("Error")

plt.grid(True)
plt.legend()

plt.show()