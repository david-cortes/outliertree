# OutlierTree

Explainable outlier/anomaly detection based on smart decision tree grouping, similar in spirit to the GritBot software developed by RuleQuest research. Written in C++ with interfaces for R (*coming soon!!*) and Python. Supports columns of types numeric, categorical, binary/boolean, and ordinal, and can handle missing values in all of them. Ideal as a sanity checker in exploratory data analysis.

*Note: R version will be available in a couple days - at the time of writing this, has not yet been uploaded*

# How it works

Will try to fit decision trees that try to "predict" values for each column based on the values of each other column. Along the way, each time a split is evaluated, it will take the observations that fall into each branch as a homogeneous cluster in which it will search for outliers in the 1-d distribution of the column being predicted. Outliers are determined according to confidence intervals on this 1-d distribution, and need to have a large gap with respect to the next observation in sorted order to be flagged as outliers. Since outliers are searched for in a decision tree branch, it will know the conditions that make it a rare observation compared to others that meet the same conditions.

As such, it will only be able to detect outliers that can be described through a decision tree logic, and unlike other methods such as Isolation Forests, will not be able to assign an outlier score to each observation, nor to detect outliers that are just overall rare, but will always provide a human-readable justification when it flags an outlier.

ArXiv report to come in the future.

# Example outputs

Example outliers from [hypothyroid dataset](http://archive.ics.uci.edu/ml/datasets/thyroid+disease):
```
row [1137] - suspicious column: [age] - suspicious value: [75.000]
	distribution: 95.122% <= 42.000 - [mean: 31.462] - [sd: 5.281] - [norm. obs: 39]
	given:
		[pregnant] = [t]


row [2229] - suspicious column: [T3] - suspicious vale: [10.600]
	distribution: 99.951% <= 7.100 - [mean: 1.984] - [sd: 0.750] - [norm. obs: 2050]
	given:
		[query hyperthyroid] = [f]
```
(i.e. it's saying that it's abnormal to be pregnant at the age of 75, or to not be classified as hyperthyroidal when having very high thyroid hormone levels)
(this dataset is also bundled into the R package (*coming soon!!*) - e.g. `data(hypothyroid)`)


Example outlier from [Titanic dataset](https://www.kaggle.com/c/titanic):
```
row [885] - suspicious column: [Fare] - suspicious value: [29.125]
	distribution: 97.849% <= 15.500 - [mean: 7.887] - [sd: 1.173] - [norm. obs: 91]
	given:
		[Pclass] = [3]
		[SibSp] = [0]
		[Embarked] = [Q]
```
(i.e. it's saying that this person paid too much for the kind of accomodation he had)


# Installation

* For R (*not yet uploaded at the time of writing*):
```r
devtools::install_github("david-cortes/outliertree")
```
(Coming to CRAN soon)


* For Python:
```
pip install outliertree
```
(Package has only been tested in Python 3)


* For C++: package doesn't have a build system, nor a `main` function that can produce an executable, but can be built as a shared object and wrapped into other languages with any C++11-compliant compiler (`std=c++11` in most compilers, `/std:c++14` in MSVC). For parallelization, needs OpenMP linkage (`-fopenmp` in most compilers, `/openmp` in MSVC). Package should *not* be built with optimization higher than `O3` (i.e. don't use `-Ofast`). Needs linkage to the `math` library, which should be enabled by default in most C++ compilers, but otherwise would require `-lm` argument. No external dependencies are required.


# Sample usage

* For R (*coming soon!!*):
```r
library(outlier.tree)

### random data frame with an obvious outlier
nrows = 100
set.seed(1)
df = data.frame(
	numeric_col1 = c(rnorm(nrows - 1), 1e6),
	numeric_col2 = rgamma(nrows, 1),
	categ_col    = sample(c('categA', 'categB', 'categC'), size = nrows, replace = TRUE)
	)

### fit model
outliers_model = outlier.tree(df, outliers_print = 10, save_outliers = TRUE)
outliers_df = outliers_model$outliers_df

### find outliers in new data
new_outliers = predict(outliers_model, df)

### print outliers in readable format
print(new_outliers)
```
(see documentation for more examples)


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

### fit model
outliers_model = OutlierTree()
outliers_df = outliers_model.fit(df, outliers_print = 10, return_outliers = True)

### find outliers in new data
new_outliers = outliers_model.predict(df)

### print outliers in readable format
outliers_model.print_outliers(new_outliers)
```

* For C++: see functions `fit_outliers_models` and `find_new_outliers` in header `outlier_tree.hpp`.

# Documentation

* For R (*coming soon!!*): documentation is built-in in the package (e.g. `help(outlier.tree::outlier.tree)`).

* For Python: documentation is available at [ReadTheDocs](http://outliertree.readthedocs.io/en/latest/) (and it's also built-in in the package as docstrings, e.g. `help(outliertree.OutlierTree.fit)`).

* For C++: documentation is available in the source files (not in the header).
