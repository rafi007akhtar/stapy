"""
This module contains functions required for descriptive statistics.
To use it in your code, enter one of the following lines:
    >>> from stapy import descriptive  # to import entire module
    >>> from stapy.descriptive import func_name  # to import a specific function of this module
"""

### INTRO TO STATISTICS ###

def get_sampling_error(mu, xbar):
    """
    Sampling error is defined as the difference between mean of the population and mean of the sample.

    Parameters
    ----------
    mu: mean of the parameter
    xbar: mean of the sample

    Returns
    -------
    The sampling error mu-xbar.
    """

    return mu - xbar


### DATA VISUALIZATION ###

def visualize(sample, name = "Data"):
    """
    This function takes a sample of categorical data as a list, and visualizes its elements in the form of a table, containing:
        * name of the sample
        * frequency of each sample element
        * relative frequencies (or proportions) of the sample
        * corresponding percentages.
    """ 

    n = len(sample)  # sample size

    # At first, get the frequency of each element
    unique = list(set(sample))
    freq = [sample.count(xi) for xi in unique]

    # Next, get the relative frequencies and percentages
    rel_freq = [f/n for f in freq]
    percentages = [rf*100 for rf in rel_freq]

    # Shorten the element names (for more readable printing)
    m = len(unique)
    for i in range(m):
        if len(unique[i]) > 3:
            unique[i] = unique[i][:3] + "..."

    # Finally, write them down in a table
    print(f'{name} \tFrequency \tProportion \tPercentage (%)')
    print("-----------------------------------------------------------")
    for i in range(m):
        print(f'{unique[i]} \t\t{freq[i]} \t\t{rel_freq[i]} \t\t{percentages[i]}')


### CENTRAL TENDENCIES ###

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


### VARIABILITY ###

def get_range(distribution):
    """ Returns the range of a distribution """
    return max(distribution) - min(distribution)


def get_quartiles(distribution):
    """
    Parameter: the list containing the sample or population distribution.
    Returns: a dictionary with keys "Q1", "Q2", "Q3" and corresponding values as first, second and third quartiles.
    """

    q2 = get_median(distribution)  # second quartile

    n = len(distribution)
    m = int(n/2)
    first_half = distribution[:m]
    if n%2 == 1: second_half = distribution[m+1:]
    else: second_half = distribution[m:]

    q1 = get_median(first_half)  # first quartile
    q3 = get_median(second_half)  # second quartile

    return {"Q1": q1, "Q2": q2, "Q3": q3}


def get_IQR(distribution, quartiles = None):
    """
    Parameters
    -----------
    * distribution: the list containing the sample or population distribution
    * [optional] quartiles: the quartiles dictionary pre-supplied.
    If quartiles is supplied, set distribution to False.

    Returns
    -------
    the inter-quartile range (Q3-Q1) of a distribution.
    """

    if  not quartiles:
        quartiles = get_quartiles(distribution)
    return quartiles["Q3"] - quartiles["Q1"]


def is_outlier(val, distribution):
    """
    Checks if val is an outlier in the distribution.
    Parameter: the list containing the sample or population distribution.
    Returns: true if val is an outlier; false otherwise.
    """
    quartiles = get_quartiles(distribution)

    q1 = quartiles["Q1"]
    q3 = quartiles["Q3"]

    if val < (2.5*q1 - 1.5*q3) or val > (2.5*q3 - 1.5*q1):
        return True  # outlier it is
    else: return False  # not an outlier


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
    

