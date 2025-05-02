import numpy as np
import utils.dslr_math as dslr


def fitter(features):
    """Scales columns to unit variance
    mean becomes = 0, variance becomes = 1"""
    n_features = features.shape[1]  # 13
    fitted = np.zeros_like(features)  # output array with shape (13 on 1600)
    for i in range(n_features):
        col = features[:, i]  # all rows of specific (i) feature
        mean = dslr.mean(col)
        std = dslr.std(col)
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
