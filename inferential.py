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

def create_ANOVA_table(samples):
	"""
	Print the ANOVA table for a group of samples, in the following format:
		SS (between): xx      dof (between): xx        MS (between): xx     F: xx
		SS (within): xx       dof (within): xx         MS (within): xx
	where 'xx' is the placeholder value for the variable on left-hand side.

	Parameter
	---------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	`void`
	"""

	ss_bet = sum_squared_between(samples)
	ss_with = sum_squared_within(samples)
	dof_bet = dof_between(samples)
	dof_with = dof_within(samples)
	ms_bet = ms_between(samples)
	ms_with = ms_within(samples)
	f = get_f_statistic(samples)

	results_row1 = {
		"SS (between)": ss_bet,
		"\tdof (between)": dof_bet, 
		"\tMS (between)": ms_bet, 
		"\tF": f
	}

	results_row2 = {
		"SS (within)": ss_with,
		"\tdof (within)": dof_with, 
		"\t\tMS (within)": ms_with 
	}

	for item in results_row1.items():
		print(f"{item[0]}: {item[1]}", end="")
	print()
	for item in results_row2.items():
		print(f"{item[0]}: {item[1]}", end="")
	print()

def tukey_HSD(q_critical, ms_with, n):
	"""
	Get the Tukey's Honestly Significance Difference, given a few parameters.
	Assumption: All samples have the same size.

	Parameters
	----------
	> `q_critical`: The Studentized Range Statistic at a certain alpha level
	> 'ms_with`: The mean squared value for within-group variability
	> `n`: number of items in each sample

	Returns
	-------
	The Tukey's HSD for the given parameters.
	"""

	return q_critical * ((ms_with / n) ** 0.5)

def honestly_significant_samples(samples, q_critical, verbose=True):
	"""
	Get / print the honestly significant samples among the tuple of samples.
	Assumption: All samples have the same size.

	Parameters
	----------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample
	> `q_critical`: The Studentized Range Statistic at a certain alpha level
	> `verbose`: a `bool` that governs whether or not the indices of significantly different samples be printed (defaulted to `True`)

	Returns
	-------
	A list tuples where each tuple contains a pair of honestly significant means.
	"""

	ms_with = ms_within(samples)
	n = len(samples[0])
	# all samples must have the same size
	k = len(samples)
	for i in range(1, k):
		if not len(samples[i]) == n:
			raise "Samples do not have the same size"
	THSD = tukey_HSD(q_critical, ms_with, n)  # Tukey's HSD

	means = [get_mean(sample) for sample in samples]
	significantly_different_means = []

	for i in range(k - 1):
		m1 = means[i]
		for j in range(i+1, k):
			m2 = means[j]
			diff = m1 - m2
			if diff < 0: diff = -1 * diff  # difference should always be +ve
			if diff > THSD:
				significantly_different_means.append((m1, m2))
				if verbose:
					print(f"Means of samples indexed {i} and {j} are honestly significantly different")
	
	return significantly_different_means

def cohens_d_multiple(xbar1, xbar2, ms_with):
	"""
	Get the Cohen's-d value for a multiple comparison test.

	Parameters
	----------
	> `xbar1`: the mean of the one of the samples in the test.
	> `xbar2`: the mean of another of the samples in the test.
	> `ms_with`: the mean-squared variability of the samples

	Returns
	-------
	The Cohen's-d value for both the samples in the multiple comparison test.
	"""

	return (xbar1 - xbar2) / (ms_with ** 0.5)

def get_eta_squared(samples):
	"""
	Get the eta-squared value of the samples (the explained variance)

	Parameters
	----------
	> `samples`: a tuple of lists, where each list is a sample containing all the values of that sample

	Returns
	-------
	The eta-squared value of the samples
	"""

	ss_bet = sum_squared_between(samples)
	ss_with = sum_squared_within(samples)
	ss_total = ss_bet + ss_with
	return ss_bet / ss_total

def get_slope(r, sy, sx):
	"""
	Get the slope for a regression line having given parameters.

	Parameters
	----------
	> `r`: regrwssion coefficient of the line
	> `sy` sample standard deviation of y distribution
	> `sx`: sample standard deviation of x distribution

	Returns
	-------
	The slope of the given regression line with the above parameters.
	"""
	
	return r * (sy / sx)

def get_y_intercept(x_dist, y_dist, r):
	"""
	y = mx + c => c = y - mx = ybar - r(sy/sx)xbar
	"""

	ybar = get_mean(y_dist)
	xbar = get_mean(x_dist)
	sy = bessel_correction(y_dist)['Sample SD']
	sx = bessel_correction(x_dist)['Sample SD']
	m = get_slope(r, sy, sx)

	return ybar - m * xbar

def predict_y(x0, m, c):
	"""
	Predict the value yhat for a regression line with the given parameters.

	Parameters
	----------
	> `x0`: the value of predictor
	> `m`: slope of the regression line
	> `c`: y-intercept of the regression line

	Returns
	-------
	The predicted value of y for the above given parameters.
	"""

	return (m * x0) + c\

def calculate_x(y0, c, m):
	"""
	Calculate the expected value of x given the following paramters.

	Parameters
	----------
	> `y0`: the predicted value of y
	> `c`: y-intercept of the regression line
	> `m`: slope of the regression line

	Returns
	-------
	The expected value x0 for the above given parameters.
	"""

	return (y0 - c) / m

def confidence_interval_for_regression_line(yhat, error):
	"""
	Get the confidence interval for the predicted value of outcome.

	> `yhat`: the predicted value of y
	> `erroe`: the standard error of estimate

	Returns
	-------
	The confidence interval for the predicted value yhat.
	"""

	low = yhat - error
	high = yhat + error
	return (low, high)

def chi_squared(frequencies):
	"""
	Find the chi-squared statistic for the given frequencies.

	Parameter
	---------
	> `frequencies`: an array of dictionaries, where each dictionary has two key-value pairs, 
		the first being observed frequency, while the second being expected frequency
		Example:
		[
			{ "fo": 41, "fe": 33 },
			{ "fo": 59, "fe": 67 }
		]
		Make sure you got the key names ("fo" and "fe") right.
	
	Returns
	-------
	The chi-squared value of the given frequencies.
	"""

	k2 = 0

	for freqs in frequencies:
		fo, fe = freqs["fo"], freqs["fe"]
		k2 += ((fo - fe) ** 2) / fe
	
	return k2

def get_expected_value_r(sum_fo_r, sum_fo, fg):
	"""
	Get the expected frequency from the given parameters.

	Parameters
	----------
	> `sum_fo_r`: sum of frequencies for a particular observed response (row sum)
	> `sum_fo`: sum of all observed frequencies
	> `fg`: frequency of an observed group / category

	Returns
	-------
	The expected frquency value from the given parameters.
	"""

	return (sum_fo_r / sum_fo) * fg

def get_expected_frequencies(observed_frequencies):
	"""
	Get all the expected frequencies from an matrix of observed frequencies.

	Parameters
	----------
	> `observed_frequencies`: the matrix of observed frequencies, where each row is a list containing observed frequencies for a particular response

	Returns
	-------
	A matrix containing the correponding lists of expected frequencies.
	"""

	all_obs = []  # array of all observed frequencies expanded into one array
	for obs in observed_frequencies:
		all_obs.extend(obs)
	sum_fo = sum(all_obs)  # sum of all observed frequencies

	r, c = len(observed_frequencies), len(observed_frequencies[0])  # number of rows and columns in the observed_frequencies array
	fgs = []  # will store the frequency of each group / category
	for j in range(c):
		f = 0
		for i in range(r):
			f += observed_frequencies[i][j]
		fgs.append(f)
	
	expected_frequencies = []  # will store the expected frequency arrays
	for obs in observed_frequencies:
		sum_fo_r = sum(obs)
		exp = []
		i = 0
		for ob in obs:
			fg = fgs[i]
			i += 1
			exp.append(get_expected_value_r(sum_fo_r, sum_fo, fg))
		expected_frequencies.append(exp)
	
	return expected_frequencies

def cramers_v(chi_squared_val, N, k, rc=None):
	"""
	Get the Cramer's coefficient for a chi-squared test.

	Parameters
	--------
	> `chi_squared_val`: the chi-squared value for the test
	> `N`: the total number of participants, i.e. the sum of participants from each category
	> `k`: (optional) the minimum value between number of responses (r) and the number of categories (c)
	> `rc`: a tuple containing (number_of_rows, number_of_categories), in that order
	Note: Pass `k` as `None` if passing `rc`

	Returns
	-------
	The Cramer's coefficient for the given chi-squared test.
	"""

	if not k:
		r, c = rc
		k = min(r, c)
	
	return (chi_squared_val / (N * (k - 1))) ** 0.5

def get_cramers_v_strength(v, k):
	"""
	Get the strength of a relationship corresponding to Cramer's coefficient.

	Parameters
	----------
	> `v`: Cramer's V
	> `k`: the minimum value between number of responses (r) and the number of categories (c)

	Returns
	-------
	A string describing how strong the relationsip is if the values are valid; otherwise an error string.
	"""

	strength_table = {
		2: [0.1, 0.3, 0.5],
		3: [0.07, 0.21, 0.35,],
		4: [0.5, 0.35, 0.29]
	}

	kmin = list(strength_table.keys())[0]
	kmax = list(strength_table.keys())[-1]

	if k < kmin or k > kmax:
		return f"cannot be determined; k needs to be between {kmin} and {kmax}, both inclusive"

	small_message = f"Cramer's V of {v} has small effect"
	medium_message = f"Cramer's V of {v} has medium effect"
	large_message = f"Cramer's V of {v} has large effect"
	
	row = strength_table[k]
	if v <= row[0]:
		return small_message
	if v < row[1]:
		return f"Cramer's V of {v} has effect somewhere between small and medium"
	if v == row[1]:
		return medium_message
	if v > row[1] and v < row[2]:
		return f"Cramer's V of {v} has effect somewhere between medium and large"
	if v >= row[2]:
		return large_message
