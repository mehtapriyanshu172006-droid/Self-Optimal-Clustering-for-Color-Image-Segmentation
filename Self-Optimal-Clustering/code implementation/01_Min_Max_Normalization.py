def min_max_normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    denominator = X_max - X_min
    denominator[denominator == 0] = 1e-10
    return (X - X_min) / denominator
