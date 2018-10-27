"""
This module contains functions required for descriptive statistics.
To use it in your code, enter one of the following lines:
    >>> from stapy import descriptive  # to import entire module
    >>> from stapy.descriptive import func_name  # to import a specific function of this module
"""

def get_mean(distribution):
    """
    Mean is defined as the sum of all the elements of the distribution over its size.
    Parameter: a list containing the distribution of the sample or population
    Returns: the mean of the distribution
    """

    return sum(distribution) / len(distribution)

def get_median(distribution):
    """
    Median lies exactly midway of your distribution, when arraged in an order.
    Parameter: a list containing the distribution of the sample or population
    Returns: the median of the distribution
    """

    n = len(distribution) # distribution size

    # for median, first sort the list
    distribution.sort()
    # next, compute the median based on `n`
    mid = int(n/2)
    if not n%2: median = (distribution[mid] + distribution[mid-1]) / 2
    else: median = distribution[mid]
    
    return median


def get_mode(distribution):
    """
    Mode of your distribution is its highest occuring element.
    Parameter: a list containing the distribution of the sample or population
    Returns: the mode of the distribution. If multiple modes exist, it will pick one of them at random.
    """

    # for mode, first remove duplicates
    unique = list(set(distribution))
    # next, calculate the frequency of each element in distribution
    freq = dict()  # key:value = frequency of element : the element
    for elem in unique:
        freq[distribution.count(elem)] = elem
    # finally, get the element with maximum frequency
    max_freq = max(freq.keys())
    mode = freq[max_freq]

    return mode


def get_central(distribution):
    """
    This function is used for getting the central tendencies (mean, median and mode) of a distribution.
    Parameter: a list containing the distribution of the sample or population
    Returns: a dictionary with keys "mean", "median" and "mode" and values as the corresponding mean, median and mode
    """

    # central tendencies
    mean = get_mean(distribution)
    median = get_median(distribution)
    mode = get_mode(distribution)

    return {"mean": mean, "median": median, "mode": mode}


def get_variance(distribution):
    """
    Get the variance of a population.
    Parameter: a list containing the distribution of the sample or population
    Returns: the variance = sum (squared(xi-mean)) for i = 0 to n-1
    """

    mean = get_mean(distribution)
    deviations = [xi-mean for xi in distribution]
    dev_squared = [deviation**2 for deviation in deviations]
    
    return sum(dev_squared) / len(dev_squared)


def get_SD(distribution):
    """
    Get the standard distribution of a population.
    arameter: a list containing the distribution of the sample or population
    Returns: SD = sqrt(variance)
    """

    variance = get_variance(distribution)
    return variance ** 0.5
    

    

    

