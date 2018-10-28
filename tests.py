from descriptive import *

# visualizing data
countries = [
    "US", "China", "US", "Sweden", "China", 
    "Canada", "China", "Japan", "Mexico", "US",
    "China", "Germany", "India", "India", "Japan",
    "US", "US", "US", "China", "China", 
    "India", "Japan", "England", "India", "Japan",
    "England", "India", "China", "Mexico", "US",
    "Mexico", "US", "Canada", "Pakistan", "India",
    "Japan", "China", "US", "Japan", "Germany",
    "China", "India", "India", "China", "China",
    "Germany", "Japan", "China", "US", "Japan"
]
visualize(sample=countries, name="Country")

print()

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

print()

# testing standardization

## z-scores
sample = [205,137,20,90]
print(f'For sample {sample} \n Z-scores: {get_Z_scores(sample, mean = 120, SD = 40)}')
population = [99,102,95,110]
print(f'For population {population} \n Z-scores: {get_Z_scores(population)}')

print()

# testing variability

## quartiles
sample = [
    38946, 43420, 49191, 50432, 50557, 
    52580, 53595, 54135, 60181, 10000000
]
quartiles = get_quartiles(sample)
print(f'For sample {sample} \n {quartiles} \n IQR = {get_IQR(distribution=None, quartiles=quartiles)}')

sample = [1,2,3,4,5]
quartiles = get_quartiles(sample)
print(f'For sample {sample} \n {quartiles} \n IQR = {get_IQR(distribution=None, quartiles=quartiles)}')

print()

## outliers
sample = [
    0.00005, 38946, 43420, 49191, 50432, 50557, 
    52580, 53595, 54135, 60181, 10000000
]
val = 10000000
print(f"For sample {sample}")
print (f" It is {is_outlier(val, sample)} that {val} is an outlier.")
val = 60181
print (f" It is {is_outlier(val, sample)} that {val} is an outlier.")
print (f"After eliminating outliers, the sample is {eleminate_outliers(sample)}")

print()

sample = [
    38946, 43420, 49191, 50432, 50557, 
    52580, 53595, 54135, 60181, 10000000
]
print(f"Sample: {sample}")
boxplot_summary(sample)
