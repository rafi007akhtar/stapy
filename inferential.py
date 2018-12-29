"""
This module contains functions required for inferential statistics.
To use it in your code, enter one of the following lines:
    >>> from inferential import *  # to import entire module
    >>> from inferential import func_name  # to import a specific function of this module
"""

### ESTIMATION ###

def error_margin_approx(z_star, sigma, n):
	"""
	Get the approximate margin of error for a given critical score.

	Parameters
	----------
	> z_star: the critical score of the confidence level
	> sigma: the standard deviation of the population
	> n: the size of the sample

	Returns
	-------
	The approximate margin of error, given by z_star(sigma/root(n)).
	"""
	return z_star * (sigma / (n ** 0.5))

def get_confidence_interval(xbar, sigma, n, z_star):
	"""
	Get the confidence interval of the mean paramater of a distribution.

	Parameters
	----------
	> xbar: mean of the sample
	> sigma: standard deviation of the population
	> n: size of the sample
	> z_star: critical score of the confidence level

	Returns
	-------
	A tuple denoting the range of values, both exclusive, for the mean of the population to lie in it.
	"""

	# calculate the approximate margin of error
	error = error_margin_approx(z_star, sigma, n)
	
	low = xbar - error  # lower range val
	up = xbar + error  # upper range val

	return (low, up)



