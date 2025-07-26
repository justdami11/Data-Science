import numpy as np

def mean(x):
    total = np.sum(x)
    count = x.size
    return (total/count)


def population_variance(x):
    mean = np.mean(x)
    squared_diff = (x - mean) ** 2
    variance = np.sum(squared_diff) / x.size
    return variance

def sample_variance(x):
    mean = np.mean(x)
    squared_diffs = (x - mean) ** 2
    variance = np.sum(squared_diffs) / (x.size - 1)
    return variance

