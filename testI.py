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
print(f"Exact confidence interval = {CI}")
