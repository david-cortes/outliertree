# OutlierTree

Explainable outlier/anomaly detection based on smart decision tree grouping, similar in spirit to the GritBot software developed by RuleQuest research. Written in C++ with interfaces for R and Python (additional Ruby wrapper can be found [here](https://github.com/ankane/outliertree/)). Supports columns of types numeric, categorical, binary/boolean, and ordinal, and can handle missing values in all of them. Ideal as a sanity checker in exploratory data analysis.

# Example outputs

Example outliers from the [hypothyroid dataset](http://archive.ics.uci.edu/ml/datasets/thyroid+disease):
```
row [1138] - suspicious column: [age] - suspicious value: [75.00]
	distribution: 95.122% <= 42.00 - [mean: 31.46] - [sd: 5.28] - [norm. obs: 39]
	given:
		[pregnant] = [TRUE]


row [2230] - suspicious column: [T3] - suspicious value: [10.60]
	distribution: 99.951% <= 7.10 - [mean: 1.98] - [sd: 0.75] - [norm. obs: 2050]
	given:
		[query.hyperthyroid] = [FALSE]

row [745] - suspicious column: [TT4] - suspicious value: [239.00]
	distribution: 98.571% <= 177.00 - [mean: 135.23] - [sd: 12.57] - [norm. obs: 69]
	given:
		[FTI] between (97.96, 128.12] (value: 112.74)
		[T4U] > [1.12] (value: 2.12)
		[age] > [55.00] (value: 87.00)
```
(i.e. it's saying that it's abnormal to be pregnant at the age of 75, or to not be classified as hyperthyroidal when having very high thyroid hormone levels)
(this dataset is also bundled into the R package - e.g. `data(hypothyroid)`)


Example outliers from the [Titanic dataset](https://www.kaggle.com/c/titanic):
```
row [1147] - suspicious column: [Fare] - suspicious value: [29.12]
	distribution: 97.849% <= 15.50 - [mean: 7.89] - [sd: 1.17] - [norm. obs: 91]
	given:
		[Pclass] = [3]
		[SibSp] = [0]
		[Embarked] = [Q]

row [897] - suspicious column: [Fare] - suspicious value: [0.00]
	distribution: 99.216% >= 3.17 - [mean: 9.68] - [sd: 6.98] - [norm. obs: 506]
	given:
		[Pclass] = [3]
		[SibSp] = [0]
```
(i.e. it's saying that the the first person paid too much for the kind of accomodation he had, and the second person should not have gotten it for free)

_Note that it can also produce other types of conditions such as 'between' (for numeric intervals) or 'in' (for categorical subsets)_

# How it works

Will try to fit decision trees that try to "predict" values for each column based on the values of each other column. Along the way, each time a split is evaluated, it will take the observations that fall into each branch as a homogeneous cluster in which it will search for outliers in the 1-d distribution of the column being predicted. Outliers are determined according to confidence intervals on this 1-d distribution, and need to have a large gap with respect to the next observation in sorted order to be flagged as outliers. Since outliers are searched for in a decision tree branch, it will know the conditions that make it a rare observation compared to others that meet the same conditions, and the conditions will always be correlated with the target variable (as it's being predicted from them).

As such, it will only be able to detect outliers that can be described through a decision tree logic, and unlike other methods such as [Isolation Forests](https://github.com/david-cortes/isotree), will not be able to assign an outlier score to each observation, nor to detect outliers that are just overall rare, but will always provide a human-readable justification when it flags an outlier.

Procedure is described in more detail in [Explainable outlier detection through decision tree conditioning](http://arxiv.org/abs/2001.00636).

# Installation

* For R:
```r
install.packages("outliertree")
```


* For Python:
```
pip install outliertree
```
or if that fails:
```
pip install --no-use-pep517 outliertree
```
** *

**Note for macOS users:** on macOS, the Python version of this package might compile **without** multi-threading capabilities. In order to enable multi-threading support, first install OpenMP:
```
brew install libomp
```
And then reinstall this package: `pip install --upgrade --no-deps --force-reinstall outliertree`.

** *
**IMPORTANT:** the setup script will try to add compilation flag `-march=native`. This instructs the compiler to tune the package for the CPU in which it is being installed (by e.g. using AVX instructions if available), but the result might not be usable in other computers. If building a binary wheel of this package or putting it into a docker image which will be used in different machines, this can be overriden either by (a) defining an environment variable `DONT_SET_MARCH=1`, or by (b) manually supplying compilation `CFLAGS` as an environment variable with something related to architecture. For maximum compatibility (but slowest speed), it's possible to do something like this:

```
export DONT_SET_MARCH=1
pip install outliertree
```

or, by specifying some compilation flag for architecture:
```
export CFLAGS="-march=x86-64"
export CXXFLAGS="-march=x86-64"
pip install outliertree
```
** *

* For C++: package doesn't have a build system, nor a `main` function that can produce an executable, but can be built as a shared object and wrapped into other languages with any C++11-compliant compiler (`std=c++11` in most compilers, `/std:c++14` in MSVC). For parallelization, needs OpenMP linkage (`-fopenmp` in most compilers, `/openmp` in MSVC). Package should *not* be built with optimization higher than `O3` (i.e. don't use `-Ofast`). Needs linkage to the `math` library, which should be enabled by default in most C++ compilers, but otherwise would require `-lm` argument. No external dependencies are required.

* For Ruby: see [external repository with wrapper](https://github.com/ankane/outliertree/).

# Sample usage

* For R:
```r
library(outliertree)

### random data frame with an obvious outlier
nrows = 100
set.seed(1)
df = data.frame(
    numeric_col1 = c(rnorm(nrows - 1), 1e6),
    numeric_col2 = rgamma(nrows, 1),
    categ_col    = sample(c('categA', 'categB', 'categC'), size = nrows, replace = TRUE)
)

### test data frame with another obvious outlier
nrows_test = 50
df_test = data.frame(
    numeric_col1 = rnorm(nrows_test),
    numeric_col2 = c(-1e6, rgamma(nrows_test - 1, 1)),
    categ_col    = sample(c('categA', 'categB', 'categC'), size = nrows_test, replace = TRUE)
)

### fit model
outliers_model = outliertree::outlier.tree(df, outliers_print = 10, save_outliers = TRUE)

### find outliers in new data
new_outliers = predict(outliers_model, df_test, outliers_print = 10, return_outliers = TRUE)

### print outliers in readable format
summary(new_outliers)
```
(see documentation for more examples)

Example [RMarkdown](http://htmlpreview.github.io/?https://github.com/david-cortes/outliertree/blob/master/vignettes/Explainable_Outlier_Detection_in_Titanic_dataset.html) using the Titanic dataset.


* For Python:
```python
import numpy as np, pandas as pd
from outliertree import OutlierTree

### random data frame with an obvious outlier
nrows = 100
np.random.seed(1)
df = pd.DataFrame({
    "numeric_col1" : np.r_[np.random.normal(size = nrows - 1), np.array([float(1e6)])],
    "numeric_col2" : np.random.gamma(1, 1, size = nrows),
    "categ_col"    : np.random.choice(['categA', 'categB', 'categC'], size = nrows)
})

### test data frame with another obvious outlier
df_test = pd.DataFrame({
    "numeric_col1" : np.random.normal(size = nrows),
    "numeric_col2" : np.r_[np.array([float(-1e6)]), np.random.gamma(1, 1, size = nrows - 1)],
    "categ_col"    : np.random.choice(['categA', 'categB', 'categC'], size = nrows)
})

### fit model
outliers_model = OutlierTree()
outliers_df = outliers_model.fit(df, outliers_print = 10, return_outliers = True)

### find outliers in new data
new_outliers = outliers_model.predict(df_test)

### print outliers in readable format
outliers_model.print_outliers(new_outliers)
```

Example [IPython notebook](http://nbviewer.ipython.org/github/david-cortes/outliertree/blob/master/example/titanic_outliertree_python.ipynb) using the Titanic dataset.

* For Ruby: see the [external repository](https://github.com/ankane/outliertree/).

* For C++: see functions `fit_outliers_models` and `find_new_outliers` in header `outlier_tree.hpp`.

# Documentation

* For R : documentation is built-in in the package (e.g. `help(outliertree::outlier.tree)`) - PDF can be downloaded in [CRAN](https://cran.r-project.org/web/packages/outliertree/index.html).

* For Python: documentation is available at [ReadTheDocs](http://outliertree.readthedocs.io/en/latest/) (and it's also built-in in the package as docstrings, e.g. `help(outliertree.OutlierTree.fit)`).

* For Ruby: see the [external repository](https://github.com/ankane/outliertree/) and the [Python documentation](http://outliertree.readthedocs.io/en/latest/).

* For C++: documentation is available in the source files (not in the header).

# References

* Cortes, David. "Explainable outlier detection through decision tree conditioning." arXiv preprint arXiv:2001.00636 (2020).
* [GritBot software](https://www.rulequest.com/gritbot-info.html) .

