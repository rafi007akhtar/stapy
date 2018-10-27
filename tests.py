from descriptive import *

# testing central tendencies

## even sample size
sample = [6,1,2,2,5,6,6,5]
print(f'Central tendencies of sample {sample}: \n {get_central(sample)}')
## odd sample size
sample = [12,19,7,1,7]
print(f'Central tendencies of sample {sample}: \n {get_central(sample)}')

print()

# testing standard deviation

## of population
population = [2,4,21,32,423,12]
print(f'For population {population}:\n Variance = {get_variance(population)} \n SD = {get_SD(population)}')

## of sample
sample = [2,12,423]
print(f'For sample {sample}:\n {bessel_correction(sample)}')
