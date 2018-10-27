# stapy

Python package containing functions implemented for descriptive and inferential statistics.

Presently, this package is a work in progress. I will add to this package as I further my study in statistics.

As of now, I have only touched topics on Descriptive Statistics. Once that is done, I hope to cover Inferential as well.

## Modules in this package

### `descriptive.py`
Contains functions related to Descriptive Statistics, comprising:
- one function each for mean, median and mode
- one function for all central tendencies 
- population variance
- population standard deviation
- one function of sample variance and sample SD
- Z-scores of a distribution.

In order to use this, you may import the entire module:
```
from stapy import descriptive
```
Or, you may include a particular function:
```
from stapy.descriptive import get_Z_scores
```
And use them as you like:
```
>>> sample = [205,137,20,90]
>>> get_Z_scores(sample, mean=120, SD=40)
[2.125, 0.425, -2.5, -0.75]
```

More on the use of this module is provided in the [tests.py](https://github.com/rafi007akhtar/stapy/blob/master/tests.py) file.

### `inferential.py`
Coming soon.

## LICENSE
[MIT License](https://github.com/rafi007akhtar/stapy/blob/master/LICENSE)
