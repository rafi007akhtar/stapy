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


### T-TESTS ###

def get_t_stat(xbar, mu0, SE, s = None, n = None):
	"""
	Get the t-statistic or simply, t for a t-distribution.

	Parameters
	----------
	> xbar: mean of sample from the population
	> mu0: mean of the current population
	> SE: the standard error of the t-distribution; 
		  if supplied as None, it will be calculated from the below two params
	> s (optional): standard deviation of the sample, obtained from bessel's correction
	> n (optional): size of the sample

	Returns
	-------
	The t-statistic, given by (xbar - mu0) / SE where SE = s/sqrt(n)
	"""

	if not SE:
		SE = s / (n**0.5)
	
	return (xbar - mu0) / SE

def get_dof(n):
	"""
	Get the degrees of freedom from sample size n.

	Parameter
	---------
	> n: the sample size

	Returns
	-------
	The degrees of freedom for sample size n
	"""
	return n - 1

from constants import t_table
from utilities import Table
def get_t_critical(dof, alpha, tails=1):
	"""
	Perform one- or two-tailed t-test on a sample, and get the t-critical value.

	Parameters
	----------
	> dof: degrees of freedom
	> alpha: the alpha level, or tail-probability; MUST be one of the following values:
		[.25, .20, .15, .10, .05, .025, .02, .01, .005, .0025, .001, .0005]
	> tails: by default 1, for 1-tailed t-test; supply 2 for 2-tailed t-test

	Returns
	--------
	The t-critical value for the given parameters, or -1 in case of an error.
	"""

	# perform sanity checks
	if alpha > 1 or alpha < 0 or (str(tails) not in "12"):
		return -1
	
	# halve the probability for two-tailed test
	if tails == 2:
		alpha /= 2

	# at first, find the tuple with the given dof
	obj = Table([], [])
	dof_row = obj.select(t_table, "dof", dof)

	# now, get the column with probability alpha, and extract the value from it
	t_critical = obj.project(dof_row, alpha)[1][0]

	# finally, return the critical value
	return t_critical

def t_test(t_statistic, t_critical):
	"""
	Accept or reject the null hypothesis

	Parameters
	----------
	> t_statistic: the t value for the distribution
	> t_critical: the t* or t-critical value

	Returns
	-------
	`True` if null is accepted; `False` if rejected
	"""
	# when t is +ve
	if  t_statistic >= 0:
		return t_critical <= t_statistic
	
	# when t is -ve
	return t_statistic > t_critical

def cohens_d(xbar, mu, s):
	"""
	Returns Cohen's d for a sample.

	Parameters
	----------
	> xbar: mean of the sample
	> mu: mean of the population
	> s: standard distribution of the sample

	Returns
	-------
	Cohen's d, or the number of standard deviations the sample mean is away from the population mean
	"""

	return (xbar - mu) / s