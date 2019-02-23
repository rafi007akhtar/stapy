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

def error_margin_sample(z_star, SE):
	"""
	Get the margin of error of a sampling distribution.

	Parameters
	----------
	> z_star: the critical score of the confidence level
	> SE: the standard error of the sample

	Returns
	-------
	The margin of error, given by z_star/SE.
	"""
	return z_star / SE

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


### HYPOTHESIS TESTING ###

def H0(mean_tuple, z_tuple = None, significance = 5):
	"""
	Accept or reject the null hypothesis H0

	Parameters
	----------
	> mean_tuple: a tuple containing means before and after intervention (mu, muI)
	> z_tuple: a tuple contianing values of (z, z_star)
	Provide any one of the required tuples. 
	> significance: the minimum difference between the means for accepting the null

	Returns
	-------
	True -> if it accepts the null
	False -> if it rejects the null
	"""
	
	if z_tuple:
		z, z_star = z_tuple
		if abs(z) > abs(z_star):
			# mean lies within the critical region
			return False  # reject the null
		return True  # accept the null
	
	# else, for mean-tuple
	mean, meanI = mean_tuple
	if abs(mean-meanI) > 5:
		return True  # accept the null
	return False  # reject the null


def HA(mean_tuple, z_tuple = None, significance = 5):
	"""
	Accept or reject the alternative hypothesis HA

	Parameters
	----------
	> significance: the minimum difference between the means for rejecting HA
	> mean_tuple: a tuple containing means before and after intervention (mu, muI)
	> z_tuple: a tuple contianing values of (z, z_star)
	Provide any one of the required tuples. 

	Returns
	-------
	True -> if it accepts HA
	False -> if it rejects HA
	"""

	# accepting the alternative hypothesis means rejecting the null
	if not H0(mean_tuple, z_tuple, significance):
		return True
	return False

def get_dof(n):
	"""
	Get the degrees of freedom from sample size n.
	"""
	return n - 1




