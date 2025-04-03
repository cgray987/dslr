import numpy as np
import utils.dslr_math as dslr


def fitter(X):
    """Scales columns to unit variance
    mean becomes = 0, variance becomes = 1"""
    X = np.array(X)
    n_features = X.shape[1]
    fitted = np.zeros_like(X)

    for i in range(n_features):
        col = X[:, i]
        mean = dslr.mean(col[~np.isnan(col)])
        std = dslr.std(col[~np.isnan(col)])

        if std == 0:
            fitted[:, i] = 0
        else:
            fitted[:, i] = (col - mean) / std

    return fitted


def sigmoid(guess):
    """returns probability (array) between 0-1 using
    sigmoid function"""
    guess = np.clip(guess, -100, 100)
    return (1 / (1 + np.exp(-guess)))
