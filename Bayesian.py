import pandas as pd
import numpy as np
from math import pi, exp, sqrt
from sklearn.model_selections import train_test_split

# Load large MNIST-like dataset
print("Loading data...")
X =  np.load('X_train.npy')  # Features: pixels
y = np.load('y_train.npy')  # Labels: digits

# Split into train/test sets

X_train,  X_test,y_train, y_test = train_test_split(X,y,train_size = 0.8,random_state=12)

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
