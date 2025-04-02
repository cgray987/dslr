import numpy as np


def count(arr):
    """len of given array-like"""
    try:
        arr = arr.astype('float')
        arr = arr[~np.isnan(arr)]
        return len(arr)
    except Exception:
        return len(arr)


def sum(arr):
    """sum of given array-like"""
    sum = 0
    for x in arr:
        if not np.isnan(x):
            sum += x
    return sum


def mean(arr):
    """mean (avg) of given array-like"""
    n = count(arr)
    if n == 0:
        return np.nan
    return sum(arr) / len(arr)


def variance(arr):
    """variance of given array-like"""
    avg = mean(arr)
    return sum([(x - avg) ** 2 for x in arr]) / len(arr)


def std(arr):
    """standard deviation of given array-like"""
    return (variance(arr) ** 0.5)


def min(arr):
    """minimum of arr-like"""
    min_value = arr[0]
    for x in arr:
        if x < min_value:
            min_value = x
    return min_value


def max(arr):
    """maximum of arr-like"""
    min_value = arr[0]
    for x in arr:
        if x > min_value:
            min_value = x
    return min_value


def quantile(arr, p):
    """returns the 'Pth' percentile for given array-like"""
    arr = arr[~np.isnan(arr)]  # Remove NaN values
    arr = sorted(arr)
    position = (len(arr) - 1) * (p / 100)
    lower = int(np.floor(position))
    higher = int(np.ceil(position))

    # if position is not an int, linearly interpolate from data
    if lower == higher:
        return arr[lower]
    fraction = position - lower
    ret = arr[lower] * (1 - fraction) + arr[higher] * fraction
    return ret
