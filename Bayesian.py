import pandas as pd
import numpy as np
from math import pi, exp, sqrt

# Load large MNIST-like dataset
print("Loading data...")
df = pd.read_csv("mnist_large.csv")  # Assumes 'label' is first column
X = df.iloc[:, 1:].values  # Features: pixels
y = df.iloc[:, 0].values   # Labels: digits

# Split into train/test sets
train_size = 50000
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

classes = np.unique(y_train)
n_features = X_train.shape[1]

# 1. Compute prior probabilities
print("Computing priors...")
priors = {c: np.mean(y_train == c) for c in classes}

# 2. Compute mean and variance for each class and feature
print("Computing likelihoods (mean and variance)...")
means = {}
variances = {}
epsilon = 1e-6  # Smoothing for variance

for c in classes:
    X_c = X_train[y_train == c]
    means[c] = np.mean(X_c, axis=0)
    variances[c] = np.var(X_c, axis=0) + epsilon

# 3. Define Gaussian PDF using vectorized operations
def gaussian_log_pdf(x, mu, var):
    return -0.5 * np.sum(np.log(2. * pi * var) + ((x - mu) ** 2) / var)

# 4. Predict using log-space to avoid underflow
def predict(x):
    log_probs = {}
    for c in classes:
        log_prior = np.log(priors[c])
        log_likelihood = gaussian_log_pdf(x, means[c], variances[c])
        log_probs[c] = log_prior + log_likelihood
    return max(log_probs, key=log_probs.get)

# 5. Test the model
print("Predicting test data...")
correct = 0
total = len(X_test)

for i in range(total):
    pred = predict(X_test[i])
    if pred == y_test[i]:
        correct += 1
    if i % 500 == 0:
        print(f"Tested {i}/{total} samples...")

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
