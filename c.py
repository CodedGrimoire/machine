import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)


# ------------------------------
# Load dataset
# ------------------------------
def load_data(path):

    data = pd.read_csv(path)

    X = data.iloc[:,0].values.reshape(-1,1)
    y = data.iloc[:,1].values.reshape(-1,1)

    return X,y


# ------------------------------
# Normalize feature
# ------------------------------
def normalize(X):

    mean = np.mean(X)
    std = np.std(X)

    X_norm = (X - mean) / std

    return X_norm, mean, std


# ------------------------------
# Polynomial Features
# ------------------------------
def polynomial_features(X,degree):

    X_poly = np.ones((len(X),1))

    for i in range(1,degree+1):
        X_poly = np.c_[X_poly, X**i]

    return X_poly


# ------------------------------
# Cost Function
# ------------------------------
def compute_cost(X,y,theta):

    m=len(y)

    predictions = X @ theta
    error = predictions - y

    return (1/(2*m))*np.sum(error**2)


# ------------------------------
# Gradient Descent
# ------------------------------
def gradient_descent(X_train,y_train,X_val,y_val,theta,lr,iterations):

    m=len(y_train)

    train_errors=[]
    val_errors=[]

    for i in range(iterations):

        gradient = (1/m)*X_train.T @ (X_train@theta - y_train)

        theta = theta - lr*gradient

        train_errors.append(compute_cost(X_train,y_train,theta))
        val_errors.append(compute_cost(X_val,y_val,theta))

        if i % 2000 == 0:
            print(f"Iteration {i} | Train Error: {train_errors[-1]:.4f} | Val Error: {val_errors[-1]:.4f}")

    return theta,train_errors,val_errors


# ------------------------------
# Train Model
# ------------------------------
def train_poly(X_train,y_train,X_val,y_val,degree):

    print(f"\nTraining Polynomial Model with Degree d = {degree}")

    X_train_poly = polynomial_features(X_train,degree)
    X_val_poly = polynomial_features(X_val,degree)

    theta=np.zeros((X_train_poly.shape[1],1))

    theta,train_errors,val_errors = gradient_descent(
        X_train_poly,y_train,X_val_poly,y_val,theta,0.001,10000
    )

    return theta,train_errors,val_errors


# ==============================
# MAIN PROGRAM
# ==============================

print("Loading dataset...")

X,y = load_data("data_02b.csv")


# ------------------------------
# Normalize Feature
# ------------------------------

X, X_mean, X_std = normalize(X)


# ------------------------------
# Plot dataset
# ------------------------------

plt.scatter(X.flatten(), y.flatten())
plt.title("Feature vs Target (Normalized)")
plt.xlabel("Normalized Feature")
plt.ylabel("Target")
plt.grid(True)
plt.show()


# ------------------------------
# Train / Validation split
# ------------------------------

split = int(0.8*len(X))

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]


# polynomial degrees
degrees=[1,2,3,4]

validation_errors=[]
thetas=[]
predictions=[]


# ------------------------------
# Train models
# ------------------------------

for d in degrees:

    print(f"\n========== DEGREE d = {d} ==========")

    theta,train_errors,val_errors = train_poly(X_train,y_train,X_val,y_val,d)

    validation_errors.append(val_errors[-1])
    thetas.append(theta)

    print("\nLearned Parameters:")
    for i,val in enumerate(theta):
        print(f"Theta{i}: {val[0]:.4f}")

    # prediction for plotting
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
    X_plot_poly = polynomial_features(X_plot,d)
    y_pred = X_plot_poly @ theta

    predictions.append((d,X_plot,y_pred))


    # ------------------------------
    # Error curves
    # ------------------------------

    plt.figure()

    plt.plot(train_errors,label="Training Error")
    plt.plot(val_errors,label="Validation Error")

    plt.title(f"Training vs Validation Error (d = {d})")
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    plt.legend()
    plt.grid(True)
    plt.show()


# ------------------------------
# Final regression comparison
# ------------------------------

plt.figure()

plt.scatter(X.flatten(), y.flatten(), label="Data")

for d,X_plot,y_pred in predictions:
    plt.plot(X_plot.flatten(),y_pred.flatten(),label=f"d = {d}")

plt.title("Polynomial Regression Fits")
plt.xlabel("Normalized Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------
# Validation bar plot
# ------------------------------

plt.bar(["d=1","d=2","d=3","d=4"],validation_errors)

plt.title("Validation Errors")
plt.ylabel("Validation Error")
plt.grid(True)

plt.show()


# ------------------------------
# Best Model
# ------------------------------

best_index = np.argmin(validation_errors)

best_degree = degrees[best_index]
best_theta = thetas[best_index]

print("\n==============================")
print("Best Polynomial Degree:",best_degree)
print(f"Best Validation Error: {validation_errors[best_index]:.4f}")

print("\nLearned Parameters of Best Model:")
for i,val in enumerate(best_theta):
    print(f"Theta{i}: {val[0]:.4f}")

print("==============================")