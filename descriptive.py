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


def get_variance(distribution, bessel = False, mean = None):
    """
    Get the variance of a population.
    Parameters
    ----------
    * distribution: a list containing the distribution of the sample or population.
    * [optional] bessel: a boolean that computes the sample variance if True (that is, with divides by n-1 instead of n if True).
    * [optional] mean: the average of the distribution; it will be computed if not provided.

    Returns
    -------
    The variance = sum (squared(xi-mean)) / n for i = 0 to n-1
    (When bessel is set, variance = sum (squared(xi-mean)) / (n-1) for i = 0 to n-1)
    """

    # calculate the mean if not already supplied
    if not mean:
        mean = get_mean(distribution)
    # now, the deviations from the mean, and their squares
    deviations = [xi-mean for xi in distribution]
    dev_squared = [deviation**2 for deviation in deviations]
    
    n = len(dev_squared)
    if bessel: n = n-1  # for Bessel corrected variance
    return sum(dev_squared) / n  # variance


def get_SD(distribution, variance = None):
    """
    Get the standard distribution of a population.

    Parameters
    ----------
    * distribution: a list containing the distribution of the sample or population
    * [optional] variance: the variance pre-supplied (None by default)
    Note: If you are supplying your own variance, set the first parameter (that is, the distribution) to None.

    Returns
    -------
    Standard deviation, either of the distribution or from the variance given.
    """

    if not variance:
        variance = get_variance(distribution)
    return variance ** 0.5
    

def bessel_correction(distribution):
    """
    Get the Bessel corrected variance and distribution of a sample.
    Parameter: a list containing the distribution of the sample.
    Returns: A dictionary with keys "Sample variance" and "Sample SD", and their corresponding values.
    """

    sample_variance = get_variance(distribution, bessel=True)
    sample_SD = get_SD(None, sample_variance)
    
    return {"Sample variance": sample_variance, "Sample SD": sample_SD}
    

def get_Z_scores(distribution, mean = None, SD = None):
    """
    Get the Z-scores of a distribution, by taking away the mean from each element, and dividing by the standard deviation.

    Parameters
    ----------
    * distribution: a list containing the distribution of the sample or population.
    * [optional] mean: the mean of the distribution (will be calculated if not supplied)
    * [optional] SD: standard deviation of the distribution (will be calculated for the population if not supplied)

    Returns
    -------
    A list of numbers containing the Z-scores of the distribution
    """

    # compute mean if not supplied
    if not mean:
        mean = get_mean(distribution)
    
    # compute SD if not supplied
    if not SD:
        variance = get_variance(distribution, mean = mean)
        SD = get_SD(distribution = None, variance = variance)
    
    # finally, compute the Z-scores
    z_scores = [(xi - mean) / SD for xi in distribution]

    return z_scores
    

