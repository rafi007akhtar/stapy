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
	Get the Cohen's d for a sample.

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

def get_CI_percent(alpha):
	"""
	Get the confidence interval in percentage

	Parameter
	----------
	> alpha: the alpha level for the test (in percentage)
	
	Returns
	-------
	The confidence interval in percentage
	"""
	
	return 100 - alpha

def get_CI_for_t_distribution(xbar, t_critical, s, n):
	"""
	Get the confidence interval for a t-distribution

	Parameters
	----------
	> xbar: mean of the sample
	> t_critical: t-critical value for the distribution
	> s: sample standard distribution
	> n: size of the sample

	Returns
	-------
	The confidence interval of the t-distribution in absolute value (not percentage)
	"""

	low = xbar - (t_critical * s) / (n ** 0.5)
	up = xbar + (t_critical * s) / (n ** 0.5)
	return (low, up)

def get_margin_of_error(CI, t_critical=None, s=None, n=None):
	"""
	Get the margin of error for a t-distribution, directly or from the CI

	Parameters
	----------
	> CI: confidence interval of the t-distribution (if this is supplied, skip the rest)
	> t_critical: t-critical value for the distribution (supply if CI is set to `None`)
	> s: sample standard deviation (supply if CI is set to `None`)
	> n: size of the sample (supply if CI is set to `None`)

	Returns
	-------
	The margin of error of the t-distribution
	"""

	if not CI:
		return t_critical * s / (n ** 0.5)

	# if CI is supplied, compute its absolute half
	low, up = CI
	return (up - low) / 2

def get_r_squared(t, dof):
	"""
	Get the r-squared value for effective size measure.

	Parameters
	----------
	> t: the t-statistic for the distribution
	> dof: the degrees of freedom of the sample

	Returns
	-------
	The r_squared value for effective size measure
	"""

	t_squared = t * t
	return t_squared / (t_squared + dof)

from descriptive import get_mean
def get_dependent_stats(x1, x2):
	"""
	Get the following statistics for two dependent distributions.
	1. First sample and sample mean
	2. Second sample and sample mean
	3. Difference of both the samples and difference mean

	Parameters
	----------
	> x1: an array containing elements of the first distribution
	> x2: an array containing elements of the second distribution

	Returns a dictionary with the following key-value pairs.
	{
		'first_sample': x1,  # the array `x1`
		'first_sample_mean': mean1,  # mean of the first sample
		'second_sample': x2,  # the array `x2`
		'second_sample_mean': mean2,  # mean of the second sample
		'difference': D,  # an array containing the differences between each corresponding element of `x1` and `x2`
		'mean_difference': mean_diff  # mean of the above `difference` array
	}
	"""

	D = []
	l = len(x1)
	for i in range(l):
		D.append(x2[i] - x1[i])

	mean1 = get_mean(x1)
	mean2 = get_mean(x2)
	mean_diff = get_mean(D)

	return ({
		"first_sample": x1,
		"first_sample_mean": mean1,
		"second_sample": x2,
		"second_sample_mean": mean2,
		"difference": D,
		"mean_difference": mean_diff
	})

from descriptive import bessel_correction
class IndependentSamples:
	"""
	This class is created to provide static methods that perform calculations needed for independent samples.
	"""

	@staticmethod
	def get_sample_SD(distribution):
		"""
		Get the standard deviation of an independent sample.
		NOTE: This is a sample, so Bessel's correction is applied.

		Parameters
		----------
		> distribution: the array containing all the values of the sample distribution.

		Returns
		-------
		The sample SD with Bessel's correction applied
		"""
		bessel = bessel_correction(distribution)
		return bessel["Sample SD"]
	
	@staticmethod
	def get_sample_variance(distribution):
		"""
		Get the variance of an independent sample.
		NOTE: This is a sample, so Bessel's correction is applied.

		Parameters
		----------
		> distribution: the array containing all the values of the sample distribution.

		Returns
		-------
		The sample variance with Bessel's correction applied
		"""

		bessel = bessel_correction(distribution)
		return bessel["Sample variance"]

	@staticmethod
	def get_t(xbar1, xbar2, SE, mu_diff=0):
		"""
		Get the t-statistic for the independent samples

		Parameters
		----------
		> xbar1: mean of the first distribution sample
		> xbar2: mean of the second distribution sample
		> SE: sample error of both the independent samples
		> mu_difference: the difference in population parameters, expected to be 0 by

		Returns
		-------
		The t-statistic of the independent samples
		"""

		return (xbar1 - xbar2 - mu_diff) / SE
	
	@staticmethod
	def get_dof(n1, n2):
		"""
		Get the degress of freedom of the independent samples

		Paraneters
		----------
		> n1: number of items in the first sample
		> n2: number of items in the second sample

		Returns
		-------
		The combined degrees of freedom of both the independent samples
		"""

		return n1 + n2 - 2

	@staticmethod
	def get_samples_SD(s1, s2):
		"""
		Get the combined standard deviation of both the independent samples

		Parameters
		----------
		> s1: the standard deviation of the first independent sample
		> x2: the standard deviation of the second independent sample

		Returns
		-------
		The combined standard deviation of both the independent samples
		"""

		return (s1*s1 + s2*s2) ** 0.5
	
	@staticmethod
	def get_standard_error(s1, s2, n1, n2):
		"""
		Get the standard error of the independent samples

		Parameters
		----------
		> s1: the standard deviation of the first sample
		> s2: the standard deviation of the second sample
		> n1: the size of the first sample
		> n2: the size of the second sample

		Returns
		-------
		The standard error of the independent samples
		"""

		SE1 = (s1 * s1) / n1
		SE2 = (s2 * s2) / n2
		return (SE1 + SE2) ** 0.5

	@staticmethod
	def get_confidence_interval(xbar1, xbar2, t_critical, SE):
		"""
		Get the confidence interval of independent samples.

		Parameters
		----------
		> xbar1: mean of the first sample
		> xbar2: mean of the second sample
		> t_critical: the t-critical value of the t-test
		> SE: the standard error of the samples

		Returns
		-------
		The confidence interval in a tuple `(down, up)` where `down` is the lower-limit, and `up` is the upper-limit.
		"""
		
		if xbar1 < xbar2:
			xbar1, xbar2 = xbar2, xbar1
		
		xdiff = xbar1 - xbar2

		down = xdiff - (t_critical * SE)
		up = xdiff + (t_critical * SE)
		
		return (down, up)
	
	@staticmethod
	def pooled_variance(distribution1, distribution2, verbose=False):
		"""
		Get the pooled variance of the distribution, where the sample sizes are not similar.

		Parameters
		----------
		> distribution1: an array of integers containing the distribution values of the first sample
		> distribution2: an array of integers containing the distribution values of the second sample
		> verbose (optional): a boolean that prints means and sum of squares of the samples before returning pooled variance if `True`; 
							print nothing if `False`

		Returns
		-------
		The pooled variance of the samples
		"""

		xbar1 = get_mean(distribution1)
		squares1 = [(xi - xbar1)**2 for xi in distribution1]
		ssx = sum(squares1)

		xbar2 = get_mean(distribution2)
		squares2 = [(xi - xbar2)**2 for xi in distribution2]
		ssy = sum(squares2)

		n1 = len(distribution1)
		n2 = len(distribution2)

		if verbose:
			print(f"Mean of sample 1: {xbar1}")
			print(f"Sum of squares for sample 1: {ssx}")
			print(f"Mean of sample 2: {xbar2}")
			print(f"Sum of squares for sample 2: {ssy}")

		return (ssx + ssy) / (get_dof(n1) + get_dof(n2))
	
	@staticmethod
	def corrected_SE(sp2, n1, n2):
		"""
		Get the corrected standard error of the independent samples, with the help of the pooled variance.

		Parameters
		----------
		> sp2: the pooled variance; in other words, the square of the pooled standard deviation
		> n1: size of the first sample
		> n2: size of the second sample

		Returns
		-------
		The corrected standard error of the samples.
		"""
		inverse = (n1 + n2) / (n1 * n2)
		return (inverse * sp2) ** 0.5

def number_of_tests_for_comparison(ns):
	"""
	Get the number of t-tests required to compare `ns` number of samples.

	Parameter
	---------
	> ns: number of samples

	Returns
	-------
	The number of t-tests required to compare those many samples
	"""

	return (ns * (ns -1)) / 2

def get_grand_mean(samples):
	"""
	Get the grand mean for a number of samples, or a number of means.

	Parameter
	---------
	> `samples`: A tuple containing a list of samples, or a list of means.
		- If the lists are samples, send them like `get_grand_mean([1,2,3,...], [3,4,5,...], [4,5,6,6,7,....], ...)`
		Here, each list contains all the values of that very sample.

		- If the lists are the means of the samples, send them like `get_grand_mean([3], [2], [12], ....)`
		Where each list should contain only one value, and that's the mean of its corresponding sample.
	
	Returns
	-------
	The grand mean for the means or the samples.
	"""

	N = 0
	grand_sum = 0
	for sample in samples:
		grand_sum += sum(sample)
		N += len(sample)
	return grand_sum / N

def sum_squared_between(samples):
	"""
	Get the sum of squares for between-group variability of the samples.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The sum of squares for between-group variability.
	"""

	xbarG = get_grand_mean(samples)  # grand mean
	ss = 0  # sum of squares for between-group variability
	for sample in samples:
		xbarK = get_mean(sample)
		n = len(sample)
		ss += n * ((xbarK - xbarG) ** 2)
	return ss

def dof_between(samples):
	"""
	Get the degrees of freedom for between-group variability.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The degrees of freedom for between-group variability.
	"""

	return len(samples) - 1

def ms_between(samnples):
	"""
	Get the mean squared value for betweem-group variability of the samples.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The mean squared value for between-group variability.
	"""

	ss_bet = sum_squared_between(samnples)  # sum of squares
	dof = dof_between(samnples)  # degrees of freedom
	return ss_bet / dof

from descriptive import get_SD, get_variance
def sum_squared_within(samples):
	"""
	Get the sum of squares for within-group variability of the samples.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The sum of squares for within-group variability.
	"""

	ss = 0  # sum of squares for within-group variability
	for sample in samples:
		n = len(sample)
		var = get_variance(sample, bessel=True)
		ss += (n - 1) * var
	return ss

def dof_within(samples):
	"""
	Get the degrees of freedom for within-group variability.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The degrees of freedom for within-group variability.
	"""

	k = len(samples)  # number of samples
	N = sum([len(sample) for sample in samples])
	return N - k

def ms_within(samples):
	"""
	Get the mean squared value for within-group variability of the samples.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The mean squared value for within-group variability.
	"""

	ss_with = sum_squared_within(samples)
	dof = dof_within(samples)
	return ss_with / dof

def get_f_statistic(samples):
	"""
	Get the f-statistic for the samples.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The f-statisitc for the samples.
	"""

	ms_bet = ms_between(samples)
	ms_with = ms_within(samples)
	return ms_bet / ms_with
