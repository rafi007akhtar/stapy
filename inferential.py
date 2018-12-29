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
	> The approximate margin of error, given by z_star(sigma/n).
	"""
	return z_star * (sigma / n)
