# stapy

Python package containing functions implemented for descriptive and inferential statistics.

Presently, this package is a work in progress. I will add to this package as I further my study in statistics.

As of now, I have only touched topics on Descriptive Statistics. Once that is done, I hope to cover Inferential as well.

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

More on the use of this module is provided in the [tests.py](https://github.com/rafi007akhtar/stapy/blob/master/tests.py) file.

### `inferential.py`
Coming soon.

## License
[MIT License](https://github.com/rafi007akhtar/stapy/blob/master/LICENSE)
