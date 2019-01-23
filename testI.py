from inferential import *

# testing estimation

## approximate margin of error
critical = 2
SD = 16.04
n = 35
print(f"Margin of error for critical score {critical}, SD {SD} and size {n} is {error_margin_approx(critical, SD, n)} approximately")

print()

## confidence interval
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