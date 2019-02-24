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

## proportions
print(f"Area for z<=3.423: {get_area(z=3.42)}")
print(f"Area for z<=-1.5: {get_area(z=-1.5)}")
print(f"Area for z<=5: {get_area(z=5)}")

print()

## probability
x = 95
mu = get_mean(population)
sigma = get_SD(population)
print(f"Probability to select {x} with mean {mu} and SD {sigma}: {get_probability(x, mu, sigma)}")

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

print()


# test sampling distributions

population = [1,2,3,4]
n = 2
print (f"For population {population}")

## samples
dist = get_samples(population, n)
print (f" Samples of size {n}: {dist}")

print()

## sampling distribution

samples = [
    [1,1], [1,2], [1,3], [1,4],
    [2,1], [2,2], [2,3], [2,4],
    [3,1], [3,2], [3,3], [3,4],
    [4,1], [4,2], [4,3], [4,4]
]
print (f"For samples {samples}")
samp_dist = get_sampling_distribution(samples)
print (f" Sampling distribution: {samp_dist}")
print (f" Mean: {get_mean(samp_dist)}")

print()

## standard error
print (f" SE = {get_SE(sigma=None, n=n, population=population)}")

print()

## sample z-score

"""
Question: A normally distributed population has a mean of mu=100 and a standard deviation of sigma=20.
What is the probability of randomly selecting a sample of size 4 that has a mean greater than 110?

Question source: [Intro to Statisitcs | Udacity] (https://classroom.udacity.com/courses/st095/lessons/116588932/concepts/2518247670923)
"""
# first, get the SE
SE = get_SE(sigma=20, n=4)
# now, get the z-score of this sample
z_sample = get_z_sample(xbar=100, mu=110, SE=SE)
# now, get the proportion of data less or equal to this z
area_less = get_area(z_sample)
# area more than this is 1-area_less
area_more = 1-area_less
print(f"Answer: {area_less}")

