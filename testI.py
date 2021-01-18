from inferential import *

# testing estimation

# approximate margin of error
critical = 2
SD = 16.04
n = 35
print(f"Margin of error for critical score {critical}, SD {SD} and size {n} is {error_margin_approx(critical, SD, n)} approximately")

print()

# confidence interval
xbar = 40
CI = get_confidence_interval(xbar, SD, n, critical)
print(f"Confidence interval for this score: {CI}")

critical = 1.96
CI = get_confidence_interval(xbar, SD, n, critical)
print(f"Exact confidence interval: {CI}")

n = 250  # number of users increased
CI = get_confidence_interval(xbar, SD, n, critical)
print(f"New confidence interval: {CI}")

### 98 % confidence level
critical = 2.33
CI = get_confidence_interval(xbar, SD, n, critical)
print(f"Confidence interval for 98%: {CI}")

print()

# hypothesis testing

## null hypothesis
from descriptive import get_z_sample, get_SE
n = 50
xbar = 8.3
mu = 7.47
sigma = 2.41
z = get_z_sample(xbar, mu, get_SE(sigma, n))
z_star = 1.96

if HA(mean_tuple = None, z_tuple=(z, z_star)):
    print("Null is rejected")
else:
    print("Null is accepted")

print()

# t-tests

## t-statistic
xbar = 6.47
s = 0.4
n = 500
mu0 = 6.07
t = get_t_stat(xbar, mu0, None, s, n)
print(f"t-statistic for these parameters is {t}")

## t-critical
alpha = 0.05
dof = 12
n = 30
t_critical = get_t_critical(get_dof(n), alpha, tails=2)
print(f"t-critical value for alpha level {alpha} and sample size {n} = {t_critical}")

## t-test
if t_test(t, t_critical):
    print(f"t-test for t value {t} and critical value {t_critical} has accepted the null")
else:
    print(f"t-test for t value {t} and critical value {t_critical} has rejected the null")

## cohen's d
from descriptive import get_mean, shuffle, get_SD
population = [
    38946, 43420, 49191, 50432, 50557, 
    52580, 53595, 54135, 60181, 10000000
]
shuffle(population)
sample = population[:3]
xbar = get_mean(sample)
mu = get_mean(population)
s = get_SD(sample)
d = cohens_d(xbar, mu, s)
print(f"The sample with mean {xbar} and SD {s} is {d} SD's away from population mean {mu}")

## margin of error
alpha = 5  # in percent
print(f"The CI for alpha rate of {alpha}% is {get_CI_percent(alpha)}%")
alpha = alpha / 100  # absolute value
xbar = 1700
s = 200
n = 100
t_critical = 1.984
CI = get_CI_for_t_distribution(xbar, t_critical, s, n)
print(f"The CI for the given t-distribution is {CI}")
print(f"margin of error for this CI = {get_margin_of_error(CI)} from the CI")
print(f"margin of error for this CI = {get_margin_of_error(None, t_critical, s, n)} when computed directly")

## effective size measure
t, n = -2.5, 25
dof = get_dof(n)
r_squared = get_r_squared(t, dof)
print(f"The effective measure as r-squared for t-statisitc of {t} having {dof} degrees of freedom is {r_squared}")
