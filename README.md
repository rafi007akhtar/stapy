# stapy

Python package containing functions implemented for descriptive and inferential statistics.

Presently, this package is a work in progress. I will add to this package as I further my study in statistics.

As of now, I have more-or-less covered all topics on Descriptive Statistics, though room for improvement is ever-existent. I have finally touched upon Inferential Statistics, and currently am working to finish it soon.

## Modules in this package

### [`descriptive.py`](https://github.com/rafi007akhtar/stapy/blob/master/descriptive.py)
Contains functions related to Descriptive Statistics, comprising:
- visualizing categorical data into a table
- one function each for mean, median and mode
- one function for all central tendencies 
- population variance
- population standard deviation
- one function of sample variance and sample SD
- range, quartiles, IQR of a distribution
- outlier checking and elimination in a distribution
- boxplot summary printing
- Z-scores of a distribution
- proportion using z-table
- randomly selecting all the samples from a population
- sampling distrubution of a population
- standard error of a population
- sample z-score.

In order to use this, you may import the entire module:
```py
from stapy import descriptive
```
Or, you may include a particular function:
```py
from stapy.descriptive import get_Z_scores
```
And use them as you like:
```py
>>> sample = [205,137,20,90]
>>> get_Z_scores(sample, mean=120, SD=40)
[2.125, 0.425, -2.5, -0.75]
```

More on the use of this module is provided in the [testsD.py](https://github.com/rafi007akhtar/stapy/blob/master/testsD.py) file.

### [`inferential.py`](https://github.com/rafi007akhtar/stapy/blob/master/inferential.py)
Contains functions related to Inferential Statistics, comprising:
- approximate margin of error
- confidence intervals
- hypothses testing
- t-statistics
- t-critical value
- Cohen's d
- margin of error
- t-tests
- r<sup>2</sup> effective size measure

### Usage
Import entire module
```py
from inferential import *
```
Or, a function in the module
```py
from inferential import get_confidence_interval
```
And use by calling the functions.
```py
>>> xbar, sigma, n, z_star = 40, 16.04, 35, 1.96
>>> get_confidence_interval(xbar, sigma, n, z_star)
(34.6859404956286, 45.3140595043714)
```

More on the use of this module is provided in the [testI.py](https://github.com/rafi007akhtar/stapy/blob/master/testI.py) file.

## License
[MIT License](https://github.com/rafi007akhtar/stapy/blob/master/LICENSE)
