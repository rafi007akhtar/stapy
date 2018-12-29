from inferential import *

critical = 2
SD = 16.04
n = 35
print(f"Margin of error for critical score {critical}, SD {SD} and size {n} is {error_margin_approx(critical, SD, n)} approximately")
