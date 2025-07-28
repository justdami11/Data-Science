import numpy as np

def mean(x):
    total = np.sum(x)
    count = x.size
    return (total/count)


def mode(x):
    value, count = np.unique(x, return_counts=True)
    mode = np.argmax(count)
    return value[mode]

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

def standard_deviation(x):
    mean = np.mean(x)
    return np.sqrt(np.mean((x - mean) ** 2))

###

def z_scores(x):
    mean = np.mean(x)
    std = standard_deviation(x)
    return (x - mean) / std

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def correlation_matrix(x1, x2):
    mean1, mean2 = np.mean(x1), np.mean(x2)
    std1, std2 = standard_deviation(x1), standard_deviation(x2)
    cov = np.mean((x1 - mean1) * (x2 - mean2))
    corr = cov / (std1 * std2)
    return np.xay([[1, corr], [corr, 1]])

def quantiles(x, q=[0.25, 0.5, 0.75]):
    sorted_x = np.sort(x)
    return np.percentile(sorted_x, np.xay(q)*100)

def skewness(x):
    mean = np.mean(x)
    std = standard_deviation(x)
    n = len(x)
    return (np.sum((x - mean) ** 3) / n) / (std ** 3)

def kurtosis(x):
    mean = np.mean(x)
    std = standard_deviation(x)
    n = len(x)
    return (np.sum((x - mean) ** 4) / n) / (std ** 4)

