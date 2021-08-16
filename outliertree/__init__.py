import numpy as np, pandas as pd, re, warnings, ctypes, multiprocessing, os, operator
from copy import deepcopy
from ._outlier_cpp_interface import OutlierCppObject, check_few_values, _get_has_openmp


class OutlierTree:
    """
    Outlier Tree
    
    Explainable outlier detection through decision-tree grouping. Tries to detect outliers by generating decision trees that attempt to
    "predict" the values of each column based on each other column, testing in each branch of every tried split (if it meets
    some minimum criteria) whether there are observations that seem too distant from the others in a 1-D distribution for the column that
    the split tries to "predict" (unlike other methods, this will not generate a score for each observation).

    Splits are based on gain, while outlierness is based on confidence intervals. Similar in spirit to the GritBot software
    developed by RuleQuest research.

    Supports columns of types numeric, categorical, and ordinal. Can handle missing values
    in any of them. Can also pass timestamps that will get converted to numeric but shown as timestamps in the output.
    Offers option to set columns to be used only for generating conditions without looking at outliers in them.

    Infinite values will be taken into consideration when the column is used to split another column (that is, +inf will go into
    the branch that is greater than something, -inf into the other branch), but when a column is the target of the split, they
    will be taken as missing - that is, it will not report infinite values as outliers.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the trees to grow. Can also pass zero, in which case it will only look for
        outliers with no conditions (i.e. takes each column as a 1-d distribution and looks for outliers
        in there independently of the values in other columns).
    min_gain : float
        Minimum gain that a split has to produce in order to consider it (both in terms of looking
        for outliers in each branch, and in considering whether to continue branching from them).
        Note that default value for GritBot is 1e-6, with 'gain_as_pct' = 'False', but it's recommended to pass
        higher values (e.g. 1e-1) when using 'gain_as_pct' = 'False'.
    z_norm : float
        Maximum Z-value (from standard normal distribution) that can be considered as a normal observation.
        Note that simply having values above this will not automatically flag observations as outliers,
        nor does it assume that columns follow normal distributions.
        Also used for categorical and ordinal columns for building approximate confidence intervals of
        proportions.
    z_outlier : float
        Minimum Z-value that can be considered as an outlier. There must be a large gap in the Z-value of
        the next observation in sorted order to consider it as outlier, given by (z_outlier - z_norm).
        Decreasing this parameter is likely to result in more observations being flagged as outliers.
        Ignored for categorical and ordinal columns.
    pct_outliers : float
        Approximate max percentage of outliers to expect in a given branch.
    min_size_numeric : int
        Minimum size that branches need to have when splitting a numeric column. In order to look for
        outliers in a given branch for a numeric column, it must have a minimum of twice this number
        of observations.
    min_size_categ : int
        Minimum size that branches need to have when splitting a categorical or ordinal column. In order to look for
        outliers in a given branch for a categorical, ordinal, or boolean column, it must have a minimum of twice
        this number of observations.
    categ_split : str
        How to produce categorical-by-categorical splits. Options are:

        ``"binarize"``:
            Will binarize the target variable according to whether it's equal to each present category
            within it (greater/less for ordinal), and split each binarized variable separately.
        ``"bruteforce"``:
            Will evaluate each possible binary split of the categories (that is, it evaluates 2^n potential
            splits every time). Note that trying this when there are many categories in a column will result
            in exponential computation time that might never finish.
        ``"separate"``:
            Will create one branch per category of the splitting variable (this is how GritBot handles them).
    categ_outliers : str
        How to look for outliers in categorical variables. Options are:

        ``"tail"``:
            Will try to flag outliers if there is a large gap between proportions in sorted order, and this
            gap is unexpected given the prior probabilities. Such criteria tends to sometimes flag too many
            uninteresting outliers, but is able to detect more cases and recognize outliers when there is no
            single dominant category.
        ``"majority"``:
            Will calculate an equivalent to z-value according to the number of observations that do not
            belong to the non-majority class, according to formula '(n-n_maj)/(n * p_prior) < 1/z_outlier^2'.
            Such criteria  tends to miss many interesting outliers and will only be able to flag outliers in
            large sample sizes. This is the approach used by GritBot.
    numeric_split : str
        How to determine the split point in numeric variables. Options are:

        ``"mid"``:
            Will calculate the midpoint between the largest observation that goes to the '<=' branch and the
            smallest observation that goes to the '>' branch.
        ``"raw"``:
            Will set the split point as the value of the largest observation that goes to the '<=' branch.

        This doesn't affect how outliers are determined in the training data passed to 'fit', but it does
        affect the way in which they are presented and the way in which new outliers are detected when
        using 'predict'. ``"mid"`` is recommended for continuous-valued variables, while ``"raw"`` will
        provide more readable explanations for counts data at the expense of perhaps slightly worse
        generalizability to unseen data.
    follow_all : bool
        Whether to continue branching from each split that meets the size and gain criteria. This will
        produce exponentially many more branches, and if depth is large, might take forever to finish.
        Will also produce a lot more spurious outiers. Not recommended.
    gain_as_pct : bool
        Whether the minimum gain above should be taken in absolute terms, or as a percentage of
        the standard deviation (for numerical columns) or shannon entropy (for categorical columns). Taking it in
        absolute terms will prefer making more splits on columns that have a large variance, while taking it as a
        percentage might be more restrictive on them and might create deeper trees in some columns. For GritBot
        this parameter would always be 'False'. Recommended to pass higher values for 'min_gain' when passing 'False'
        here. Not that when 'gain_as_pct' = 'False', the results will be sensitive to the scales of variables.
    nthreads : int
        Number of parallel threads to use. When fitting the model, it will only use up to one thread per
        column, while for prediction it will use up to one thread per row. The more threads that are
        used, the more memory will be required and allocated, so using more threads will not always lead
        to better speed. Passing zero or negative numbers will default to the maximum number of available CPU
        cores (but not if the object attribute is overwritten). Can be changed after the object is
        already initialized.

    Attributes
    ----------
    is_fitted_ : bool
        Indicates if the model as already been fit.
    flaggable_values : dict[ncols]
        A dictionary indicating for each column which kind of values are possible to flag as outlier in
        at least one of the explored branches. For numerical and timestamp columns, will indicate the
        lower and upper bounds of the normal range (that is, it can only flag values *outside* of that interval),
        while for categorical, ordinal, and boolean, it will indicate the categories (which it can flag as outliers).
        If the lower bound is higher than the upper bound, it means that any value can potentially be flagged as outlier.
        If no values in a column can be flagged as outlier, entry for that column will be an empty dict.
    cols_num_ : array(ncols_numeric, )
        Names of the numeric columns in the data passed to '.fit'.
    cols_cat_ : array(ncols_categ, )
        Names of the categorical/string columns in the data passed to '.fit'.
    cols_bool_ : array(ncols_bool, )
        Names of the boolean columns in the data passed to '.fit'.
    cols_ord_ : array(ncols_ordinal, )
        Names of the ordinal columns in the data passed to '.fit'.
    cols_ts_ : array(ncols_tiemstamp, )
        Names of the timestamp columns in the data passed to '.fit'.

    References
    ----------
    .. [1] GritBot software : https://www.rulequest.com/gritbot-info.html
    .. [2] Cortes, David. "Explainable outlier detection through decision tree conditioning." arXiv preprint arXiv:2001.00636 (2020).
    """
    def __init__(self, max_depth = 4, min_gain = 1e-2, z_norm = 2.67, z_outlier = 8.0, pct_outliers = 0.01,
                 min_size_numeric = 25, min_size_categ = 50, categ_split = "binarize", categ_outliers = "tail",
                 numeric_split = "raw", follow_all = False, gain_as_pct = True, nthreads = -1):

        ### validate inputs
        assert max_depth >= 0
        assert isinstance(max_depth, int)
        assert isinstance(min_gain, float)
        assert z_norm > 0
        if isinstance(z_norm, int):
            z_norm = float(z_norm)
        assert isinstance(z_norm, float)
        assert z_outlier > z_norm
        if isinstance(z_outlier, int):
            z_outlier = float(z_outlier)
        assert isinstance(z_outlier, float)
        assert pct_outliers > 0
        assert pct_outliers < 0.1
        assert min_size_numeric >= 10
        assert min_size_categ >= 10
        assert categ_split in  ["binarize", "bruteforce", "separate"]
        assert categ_outliers in ["tail", "majority"]
        assert numeric_split in ["mid", "raw"]
        assert isinstance(min_size_numeric, int)
        assert isinstance(min_size_categ, int)
        if nthreads is None:
            nthreads = 1 
        if nthreads <= 0:
            nthreads = multiprocessing.cpu_count()
        assert nthreads > 0
        assert isinstance(nthreads, int)

        if (nthreads > 1) and (not _get_has_openmp()):
            msg_omp  = "Attempting to use more than 1 thread, but "
            msg_omp += "package was built without multi-threading "
            msg_omp += "support - see the project's GitHub page for "
            msg_omp += "more information."
            warnings.warn(msg_omp)

        ### store them
        self.max_depth         =  max_depth
        self.min_gain          =  min_gain
        self.z_norm            =  z_norm
        self.z_outlier         =  z_outlier
        self.pct_outliers      =  pct_outliers
        self.min_size_numeric  =  min_size_numeric
        self.min_size_categ    =  min_size_categ
        self.categ_split       =  categ_split
        self.categ_outliers    =  categ_outliers
        self.numeric_split     =  numeric_split
        self.follow_all        =  bool(follow_all)
        self.gain_as_pct       =  bool(gain_as_pct)
        self.nthreads          =  nthreads

        ### initialize internal object
        self._reset_object()
        self._outlier_cpp_obj = OutlierCppObject()

        ### kind of a parameter
        self._min_rows = 20

    def __str__(self):
        msg = "OutlierTree model\n"
        if not self.is_fitted_:
            msg += "Object has not been fit to data."
        else:
            if self.cols_num_.shape[0]:
                msg += "\tNumeric variables: %d\n" % self.cols_num_.shape[0]
            if self.cols_ts_.shape[0]:
                msg += "\tTimestamp variables: %d\n" % self.cols_ts_.shape[0]
            if self.cols_cat_.shape[0]:
                msg += "\tCategorical variables: %d\n" % self.cols_cat_.shape[0]
            if self.cols_bool_.shape[0]:
                msg += "\tBoolean variables: %d\n" % self.cols_bool_.shape[0]
            if self.cols_ord_.shape[0]:
                msg += "\tOrdinal variables: %d\n" % self.cols_ord_.shape[0]
            nclust, ntrees = self._outlier_cpp_obj.get_n_clust_and_trees()
            msg += "\nConsists of %d clusters, spread across %d tree branches\n" % (nclust, ntrees)
        return msg

    def __repr__(self):
        return self.__str__()

    def _reset_object(self):
        self.is_fitted_   = False
        self.cols_num_    = np.empty(0, dtype = "object")
        self.cols_cat_    = np.empty(0, dtype = "object")
        self.cols_bool_   = np.empty(0, dtype = "object")
        self.cols_ord_    = np.empty(0, dtype = "object")
        self.cols_ts_     = np.empty(0, dtype = "object")
        self._cat_mapping = list()
        self._ord_mapping = list()
        self._ts_min      = np.empty(0, dtype = int)
        self.flaggable_values = dict()

    def fit(self, df, cols_ignore = None, outliers_print = 10, min_decimals = 2, return_outliers = False):
        """
        Fit Outlier Tree model to data.

        Note
        ----
        Row names will be taken as the index of the data frame.
        Column types will be taken as follows: \n
        Numeric -> Everything that is a subtype of numpy's 'number' (e.g. integers and floats). \n
        Categorical -> Python object (**even if the underlying types are numbers!!!**), boolean, and pandas Categorical. \n
        Ordinal -> pandas Categorical with ordered attribute - the order will
        be the same as they have when passed. \n
        Timestamp -> numpy's datetime64 dtype - they will only be taken with a
        precision of seconds due to numerical precision issues (code uses C 'long double'),
        and will be used as numerical but presented as timestamp in the outputs. If the required
        precision is less (e.g. just dates or years) it's recommended to pass them as numbers instead
        (e.g. 2008.1095 for some day in February 2008).

        Note
        ----
        You can look at the object's attributes to ensure that the columns are being interpreted as the type
        (numeric, categorical, ordinal, boolean, timestamp) that they should have.

        Note
        ----
        Boolean or binary columns must be passed as categorical (will not accept them as numerical, nor
        as ordinal). If they have missing values, pandas will not be able to have them with dtype "bool"
        however (they need to be passed either as "object" or "Categorical" dtypes).

        Note
        ----
        Do NOT do one-hot or dummy encoding on categorical variables.


        Parameters
        ----------
        df : DataFrame(n_rows, n_cols)
            Pandas' DataFrame with regular (i.e. non-outlier) data that might contain some outliers.
        cols_ignore : boolean array(n_cols, ) or string array(n_ignore, )
            Array containing columns which will not be split, but will be evaluated for usage in splitting other columns.
            Can pass either a boolean array with the same number of columns as 'df', or a list/array of column names
            (must match with those of 'df'). Pass 'None' to use all columns.
        outliers_print : int or None
            Maximum number of flagged outliers in the training data to print after fitting the model.
            Pass zero or None to avoid printing any.
            Outliers can be printed from resulting data frame afterwards through '.print_outliers'.
        min_decimals : int
            Minimum number of decimals to use when printing numeric values for the flagged
            outliers. The number of decimals will be dynamically increased according to the relative magnitudes of the
            values being reported. Ignored when passing ``max_outliers=0`` or ``max_outliers=None``.
        return_outliers : bool
            Whether to return a DataFrame with information about outliers flagged in the training data.
            If 'True', will return this information as a DataFrame. If 'False', will return this same
            object. See the documentation for '.predict' for more information about the format.

        Returns
        -------
        result_df or self : DataFrame(n_rows, 6) or obj
            Either a DataFrame with the information about potential outliers detected in the training data,
            or this same object if passing 'return_outliers' = 'False'. In the former, format is the
            same as when calling '.predict'. See the documentation for '.predict' for more information about
            the output format.
        """
        ## TODO: this should have 'fit' / 'fit_transform' instead of these parameters
        outliers_print = self._check_outliers_print(outliers_print, min_decimals)
        self._check_valid_data(df)
        self._reset_object()
        arr_num, arr_cat, arr_ord = self._split_types(df)
        cols_ignore = self._process_cols_ignore(df, cols_ignore)
        ncat, ncat_ord = self._get_ncat()

        has_outl, out_df = self._outlier_cpp_obj.fit_model(
                                                            arr_num, arr_cat, arr_ord,
                                                            ncat, ncat_ord, cols_ignore,
                                                            self.cols_num_.shape[0], self.cols_ts_.shape[0],
                                                            self.cols_cat_.shape[0], self.cols_bool_.shape[0],
                                                            np.r_[self.cols_num_, self.cols_ts_],
                                                            np.r_[self.cols_cat_, self.cols_bool_],
                                                            self.cols_ord_,
                                                            self._cat_mapping, self._ord_mapping,
                                                            self._ts_min.reshape(-1),
                                                            return_outliers or outliers_print,
                                                            self.nthreads,
                                                            self.categ_split == "binarize", self.categ_split == "binarize",
                                                            self.categ_split == "bruteforce", self.categ_outliers == "majority",
                                                            self.numeric_split == "mid", self.max_depth, self.pct_outliers,
                                                            self.min_size_numeric, self.min_size_categ,
                                                            self.min_gain, self.follow_all, self.gain_as_pct,
                                                            self.z_norm, self.z_outlier,
                                                            self._generate_empty_out_df(df.shape[0]) if (return_outliers or outliers_print) else None
                                                        )
        self.is_fitted_ = True

        ### obtain info on which values or categories are possible to flag as outliers
        self._determine_flaggable_values()

        ### return/print info on training data as requested
        if out_df is not None:
            out_df.index = df.index.copy()
        if outliers_print:
            if has_outl:
                self.print_outliers(out_df, outliers_print, min_decimals)
            else:
                self._print_no_outliers()

        if return_outliers:
            return out_df
        else:
            return self


    def predict(self, df, outliers_print = None, min_decimals = 2):
        """
        Detect outliers on new data

        Note
        ----
        The group statistics for outliers are calculated only on rows that are not flagged as
        outliers - that is, they do not include the row being reported in their calculations,
        and exclude any outliers that were flagged at a previous parent branch.

        Note
        ----
        This will generate conditions having these criteria: "<=", ">", "in", "=", "!=" (not equal).
        When it says "in", it means that the value is within the subset of categories provided.
        For ordinal columns, they will always be represented as "in", but the subset will follow
        the order. The ".print" method will additionally simplify conditions to numeric "between",
        and merge repeated splits that are on the same column.

        Parameters
        ----------
        df : DataFrame(n_rows, n_cols)
            Pandas DataFrame in the same format and with the same columns as the one that
            was passed to '.fit'.
        outliers_print : int or None
            Maximum number of outliers to print, if any are found.
            Pass zero or None to avoid printing any.
            Outliers can be printed from resulting data frame afterwards through '.print_outliers'.
        min_decimals : int
            Minimum number of decimals to use when printing numeric values for the flagged
            outliers. The number of decimals will be dynamically increased according to the relative magnitudes of the
            values being reported. Ignored when passing ``max_outliers=0`` or ``max_outliers=None``.

        Returns
        -------
        result_df : DataFrame(n_rows, 6)
            DataFrame indicating for each row whether it is a suspected outlier or not. When they
            are suspected outliers, the resulting DataFrame will contain columns with \n
            a) information about the column in the input data whose value is suspicious, \n
            b) aggregate statistics for the values in this column among the normal observations, \n
            c) conditions that qualify the row to be put into that group, \n
            d) depth in the decision tree branch in which they can be considered an outlier, \n
            e) Whether the conditions that make it belong to the decision tree branch contain any NA split, \n
            f) The resulting outlier score or probability (based on Chebyshyov's bound for numerical
            columns, and a simple upper conficence bound for categorical and ordinal columns - but note that it will
            always prefer to assign a row to an outlier branch that does not follow any NA path, or failing that,
            to the one with the smallest depth) - lower scores indicate lower probabilities and thus more outlierness. \n
            
            Information such as the conditions in the tree or the group statistics are returned as dictionaries,
            since they are not tabular format.
            The index of the output will be the same as that of the input.
            Outliers can be printed from resulting data frame afterwards through '.print_outliers'.
        """
        ## TODO: add a 'transform' method that would do the same thing as this
        if not self.is_fitted_:
            raise ValueError("Can only call '.predict' after the model has already been fit to some data.")
        outliers_print = self._check_outliers_print(outliers_print, min_decimals)
        if df.__class__.__name__ == "Series":
            df = df.to_frame().T
        self._check_valid_data(df)
        self._check_contains_original_cols(df)
        arr_num, arr_cat, arr_ord = self._split_types_new_df(df)
        has_outl, out_df = self._outlier_cpp_obj.predict(
                                                            arr_num, arr_cat, arr_ord,
                                                            self.cols_num_.shape[0], self.cols_ts_.shape[0],
                                                            self.cols_cat_.shape[0], self.cols_bool_.shape[0],
                                                            np.r_[self.cols_num_, self.cols_ts_],
                                                            np.r_[self.cols_cat_, self.cols_bool_],
                                                            self.cols_ord_,
                                                            self._cat_mapping, self._ord_mapping,
                                                            self._ts_min,
                                                            self.nthreads,
                                                            self._generate_empty_out_df(df.shape[0])
                                                        )

        out_df.index = df.index.copy()
        if outliers_print:
            if has_outl:
                self.print_outliers(out_df, outliers_print, min_decimals)
            else:
                self._print_no_outliers()

        return out_df

    def _check_outliers_print(self, outliers_print, min_decimals):
        if outliers_print is not None:
            assert outliers_print >= 0
            assert isinstance(outliers_print, int)
            if outliers_print == 0:
                outliers_print = None
        assert min_decimals >= 0
        return outliers_print

    def _check_valid_data(self, df):
        if df.__class__.__name__ != "DataFrame":
            raise ValueError("Input must be a pandas DataFrame with named columns.")

        if not self.is_fitted_:
            if df.shape[0] < self._min_rows:
                raise ValueError("Input data has too few rows.")
        else:
            if df.shape[0] == 0:
                raise ValueError("Input data has zero rows.")

        if len(df.shape) != 2:
            raise ValueError("Must pass a 2-dimensional DataFrame.")

        if df.shape[1] == 0:
            raise ValueError("Input data has zero columns.")

        #https://stackoverflow.com/questions/21081042/detect-whether-a-dataframe-has-a-multiindex
        if type(df.index) == pd.MultiIndex:
            raise ValueError("Cannot use multi-dimensional indices in input DataFrame.")

        if np.any(pd.isnull(df.index)):
            raise ValueError("DataFrame Index cannot contain missing values.")

        if np.unique(df.index).shape[0] < df.index.shape[0]:
            raise ValueError("DataFrame Index contains duplicate values.")

    def _split_types(self, df):
        #https://stackoverflow.com/questions/29803093/check-which-columns-in-dataframe-are-categorical
        #https://stackoverflow.com/questions/29518923/numpy-asarray-how-to-check-up-that-its-result-dtype-is-numeric
        def check_is_num_dtype(x):
            try:
                return np.issubdtype(x, np.number)
            except:
                return False

        def check_is_dt64_dtype(x):
            try:
                return np.issubdtype(x, np.datetime64)
            except:
                return False

        cols_num  = df.dtypes.map(check_is_num_dtype).to_numpy()
        cols_bool = (df.dtypes == "bool").to_numpy()
        cols_cat  = (df.dtypes == "category").to_numpy()
        cols_str  = (df.dtypes == "object").to_numpy()
        cols_ts   = df.dtypes.map(check_is_dt64_dtype).to_numpy()
        cols_unsupported = (~cols_num) & (~cols_bool) & (~cols_cat) & (~cols_str) & (~cols_ts)
        if np.any(cols_unsupported):
            err_msg = "Model support only columns of types: numeric, boolean, string, categorical, ordinal, datetime64"
            err_msg += " - got these pandas dtypes: "
            err_msg += re.sub(r"\n", ", ", a.dtypes.to_string(header = False, index = False)).strip()
            raise ValueError(err_msg)

        cols_ord = np.array([False] * df.dtypes.shape[0]).astype(cols_num.dtype)
        if np.any(cols_cat):
            cols_ord = df.columns[cols_cat][ df[df.columns[cols_cat]].apply(lambda x: x.dtype.ordered, axis = 0) ]
            cols_ord = np.in1d(df.columns.values, cols_ord)
            if np.any(cols_ord):
                cols_cat = cols_cat & (~cols_ord)
                cols_str = cols_str | cols_cat

        ### https://github.com/pandas-dev/pandas/issues/27490
        df_num  = df[df.columns[cols_num]].astype(ctypes.c_double)
        df_bool = df[df.columns[cols_bool]].astype(ctypes.c_int)
        df_cat  = df[df.columns[cols_str]].copy()
        df_ord  = df[df.columns[cols_ord]].copy()

        ### Note: numpy represents NAs in timestamps as negative integers
        if np.any(cols_ts):
            np_ts = df[df.columns[cols_ts]].to_numpy().astype('datetime64[s]').astype(int).astype(ctypes.c_double)
            np_ts[pd.isnull(df[df.columns[cols_ts]]).to_numpy()] = np.nan
            df_ts = pd.DataFrame(np_ts, columns = df.columns[cols_ts])
        else:
            df_ts = pd.DataFrame()

        ### TODO: parallelize these parts using joblib

        if df_num.shape[1] == 0:
            df_num = None
        else:
            ### check that it doesn't contain booleans disguised as numeric or all the same value
            too_few_values = check_few_values(df_num.values, self.nthreads)
            if np.any(too_few_values):
                warnings.warn("Some numeric columns have less than 3 different values - head: " + str(df_num.columns[too_few_values][:3]))
            self.cols_num_ = df_num.columns.values.copy()


        ### Categorical variables need to be encoded
        if df_cat.shape[1] == 0:
            df_cat = None
        else:
            self.cols_cat_ = df_cat.columns.values.copy()
            self._cat_mapping = list()
            for cl in range(df_cat.shape[1]):
                df_cat[df_cat.columns[cl]], encoding_this = pd.factorize(df_cat[df_cat.columns[cl]])
                df_cat[df_cat.columns[cl]] = df_cat[df_cat.columns[cl]].astype(ctypes.c_int)
                # https://github.com/pandas-dev/pandas/issues/30618
                if encoding_this.__class__.__name__ == "CategoricalIndex":
                    encoding_this = encoding_this.to_numpy()
                self._cat_mapping.append(encoding_this)

        ### Booleans are taken as categoricals
        if df_bool.shape[1] > 0:
            self.cols_bool_ = df_bool.columns.values.copy()
            self._cat_mapping += [np.array([False,True])]*df_bool.shape[1]
            if df_cat is not None:
                df_cat = pd.concat([df_cat, df_bool], axis = 1)
            else:
                df_cat = df_bool

        ### Same for ordinal ones
        if df_ord.shape[1] == 0:
            df_ord = None
        else:
            self.cols_ord_ = df_ord.columns.values.copy()
            self._ord_mapping = list()
            for cl in range(df_ord.shape[1]):
                self._ord_mapping.append(df_ord[df_ord.columns[cl]].dtype.categories)
                if self._ord_mapping[-1].shape[0] <= 2:
                    raise ValueError("Can only pass ordinal columns with at least 3 levels - column " + f_ord.columns[cl] + " has fewer.")
                df_ord[df_ord.columns[cl]] = df_ord[df_ord.columns[cl]].cat.codes.astype(ctypes.c_int)


        ### Timestamps are very large numbers that can overflow when summing their squares, so they are
        ### shortened to a precision of seconds only, and get subtracted their minimum value minus one
        ### (this -1 is due to the way that it applies logarithm transformations when advantageous)
        if df_ts.shape[1] == 0:
            df_ts = None
        else:
            if df_ts.shape[0] >= 5000:
                warn_msg = "Calculations on timestamps with many rows are not precise "
                warn_msg += "(these use sums of squares with C doubles (windows) or long doubles (others)). "
                warn_msg += "If these are dates or years, for which the required precision is less, it's recommended "
                warn_msg += "to pass them as numerical floats instead (e.g. '2008.0', '2009.0', etc.)."
                warnings.warn(warn_msg)
            self.cols_ts_ = df_ts.columns.values.copy()
            self._ts_min = df_ts.min(axis = 0).to_numpy().reshape((1, -1))
            df_ts = (df_ts - self._ts_min + 1).astype(ctypes.c_double)

            if df_num is not None:
                df_num = pd.concat([df_num, df_ts], axis = 1)
            else:
                df_num = df_ts

        return  np.asfortranarray(df_num.to_numpy()) if df_num is not None else np.empty((0, 0), dtype = ctypes.c_double), \
                np.asfortranarray(df_cat.to_numpy()) if df_cat is not None else np.empty((0, 0), dtype = ctypes.c_int), \
                np.asfortranarray(df_ord.to_numpy()) if df_ord is not None else np.empty((0, 0), dtype = ctypes.c_int)

    def _split_types_new_df(self, df):
        df_num = df[self.cols_num_].astype(ctypes.c_double) if self.cols_num_.shape[0] > 0 else None
        df_cat = df[self.cols_cat_].copy() if self.cols_cat_.shape[0] > 0 else None
        df_ord = df[self.cols_ord_].copy() if self.cols_ord_.shape[0] > 0 else None
        warn_new_cols = False
        cols_warn_new = list()

        if df_cat is not None:
            for cl in range(self.cols_cat_.shape[0]):
                new_cat = (~np.in1d(df_cat[self.cols_cat_[cl]], self._cat_mapping[cl])) & (~pd.isnull(df_cat[self.cols_cat_[cl]]).to_numpy())
                df_cat[self.cols_cat_[cl]] = pd.Categorical(df_cat[self.cols_cat_[cl]], self._cat_mapping[cl]).codes.astype(ctypes.c_int)
                if np.any(new_cat):
                    warn_new_cols = True
                    cols_warn_new.append(self.cols_cat_[cl])
                    df_cat.loc[new_cat, self.cols_cat_[cl]] = self._cat_mapping[cl].shape[0]

        if df_ord is not None:
            for cl in range(self.cols_ord_.shape[0]):
                new_cat = (~np.in1d(df_ord[self.cols_ord_[cl]], self._ord_mapping[cl])) & (~pd.isnull(df_ord[self.cols_ord_[cl]]).to_numpy())
                df_ord[self.cols_ord_[cl]] = pd.Categorical(df_ord[self.cols_ord_[cl]], self._ord_mapping[cl]).codes.astype(ctypes.c_int)
                if np.any(new_cat):
                    warn_new_cols = True
                    cols_warn_new.append(self.cols_ord_[cl])
                    df_ord.loc[new_cat, self.cols_ord_[cl]] = self._ord_mapping[cl].shape[0]

        if self.cols_bool_.shape[0] > 0:
            df_bool = df[self.cols_bool_].astype("bool").astype(ctypes.c_int)
            if df_cat is None:
                df_cat = df_bool
            else:
                df_cat = pd.concat([df_cat, df_bool], axis = 1)

        if self.cols_ts_.shape[0] > 0:
            df_ts = pd.DataFrame((df[self.cols_ts_].to_numpy().astype('datetime64[s]').astype(int) - self._ts_min + 1).reshape((df.shape[0], -1)), columns = self.cols_ts_).astype(ctypes.c_double)
            if df_num is None:
                df_num = df_ts
            else:
                df_num = pd.concat([df_num, df_ts], axis = 1)

        if warn_new_cols:
            warn_msg  = "Some categorical/ordinal columns had new values that were not "
            warn_msg += "present in the training data. These values will be ignored. "
            warn_msg += "Columns head: " + ", ".join([str(cl) for cl in cols_warn_new])[:3]
            warnings.warn(warn_msg)

        return  np.asfortranarray(df_num.to_numpy()) if df_num is not None else np.empty((0, 0), dtype = ctypes.c_double), \
                np.asfortranarray(df_cat.to_numpy()) if df_cat is not None else np.empty((0, 0), dtype = ctypes.c_int), \
                np.asfortranarray(df_ord.to_numpy()) if df_ord is not None else np.empty((0, 0), dtype = ctypes.c_int)



    def _check_contains_original_cols(self, df):
        self._check_cols_present(df, self.cols_num_)
        self._check_cols_present(df, self.cols_cat_)
        self._check_cols_present(df, self.cols_ts_)
        self._check_cols_present(df, self.cols_ord_)
        self._check_cols_present(df, self.cols_bool_)

    def _check_cols_present(self, df, cols):
        if np.any(~np.in1d(cols, df.columns)):
            missing_cols = cols[~np.in1d(cols, df.columns)]
            raise ValueError("Input DataFrame missing " + str(missing_cols.shape[0]) + " columns - head: " + str(list(missing_cols[:3])))

    def _process_cols_ignore(self, df, cols_ignore):
        if cols_ignore is None:
            return np.empty(0).astype("bool")
        cols_ignore = np.array(cols_ignore).reshape(-1)
        cols_concat = np.r_[
                            self.cols_num_,
                            self.cols_ts_,
                            self.cols_cat_,
                            self.cols_bool_,
                            self.cols_ord_
                            ]
        if cols_ignore.shape[0] == df.shape[1]:
            ### Note: speed here could be improved if NumPy ever decides to introduce something like R's 'match'
            #https://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r
            df_cols_lst = list(df.columns.values)
            order_cols = [df_cols_lst.index(cl) for cl in cols_concat]
            cols_ignore = cols_ignore[np.array(order_cols)]
            #https://github.com/numpy/numpy/issues/13973
            cols_ignore = cols_ignore.astype("bool")
            if np.any(cols_ignore):
                return cols_ignore
            else:
                return np.empty(0).astype("bool")
        else:
            if np.any(~np.in1d(cols_ignore, df.columns)):
                warnings.warn("'cols_ignore' contains column names not present in the input DataFrame.")
            cols_ignore = np.in1d(cols_concat, cols_ignore)
            if np.any(cols_ignore):
                return cols_ignore
            else:
                return np.empty(0).astype("bool")

    def _get_ncat(self):
        ncat = np.zeros(self.cols_cat_.shape[0] + self.cols_bool_.shape[0], dtype = ctypes.c_int)
        for cl in range(self.cols_cat_.shape[0]):
            if isinstance(self._cat_mapping[cl], list):
                ncat[cl] = len(self._cat_mapping[cl])
            else:
                ncat[cl] = self._cat_mapping[cl].shape[0]
        if self.cols_bool_.shape[0] > 0:
            ncat[self.cols_cat_.shape[0] :] = 2

        ncat_ord = np.zeros(self.cols_ord_.shape[0], dtype = ctypes.c_int)
        for cl in range(self.cols_ord_.shape[0]):
            if isinstance(self._ord_mapping[cl], list):
                ncat_ord[cl] = len(self._ord_mapping[cl])
            else:
                ncat_ord[cl] = self._ord_mapping[cl].shape[0]

        return ncat, ncat_ord


    def _decode_date(self, ts_int, cl_num):
        try:
            return np.datetime64(np.int(ts_int - 1 + self._ts_min[cl_num]), "s")
        except:
            return np.nan

    def _generate_empty_out_df(self, nrows):
        empty_dict_col = [dict() for row in range(nrows)]
        return pd.DataFrame({
            "suspicious_value"  :  deepcopy(empty_dict_col),
            "group_statistics" :  deepcopy(empty_dict_col),
            "conditions"       :  [list() for row in range(nrows)],
            "tree_depth"       :  np.nan,
            "uses_NA_branch"   :  np.nan,
            "outlier_score"    :  np.nan
            })

    def _determine_flaggable_values(self):
        self.flaggable_values = dict()
        min_outl, max_outl, cat_outl = self._outlier_cpp_obj.get_flaggable_bounds()

        for cl in range(min_outl.shape[0]):
            if cl < self.cols_num_.shape[0]:
                self.flaggable_values[self.cols_num_[cl]] = {"low" : min_outl[cl], "high" : max_outl[cl]}
            else:
                cl_ts = cl - self.cols_num_.shape[0]
                self.flaggable_values[self.cols_ts_[cl_ts]] = {"low" :  self._decode_date(min_outl[cl], cl_ts),
                                                               "high" : self._decode_date(max_outl[cl], cl_ts)}

        for cl in range(len(cat_outl)):
            if cl < self.cols_cat_.shape[0]:
                if cat_outl[cl].shape[0] == 0:
                    self.flaggable_values[self.cols_cat_[cl]] = dict()
                else:
                    self.flaggable_values[self.cols_cat_[cl]] = np.array(self._cat_mapping[cl])[cat_outl[cl]]

            elif cl < (self.cols_cat_.shape[0] + self.cols_bool_.shape[0]):
                if cat_outl[cl].shape[0] == 0:
                    self.flaggable_values[self.cols_bool_[cl - self.cols_cat_.shape[0]]] = dict()
                else:
                    self.flaggable_values[self.cols_bool_[cl - self.cols_cat_.shape[0]]] = np.array([False, True])[cat_outl[cl]]
            
            else:
                rel_col = cl - self.cols_cat_.shape[0] - self.cols_bool_.shape[0]
                if cat_outl[cl].shape[0] == 0:
                    self.flaggable_values[self.cols_ord_[rel_col]] = dict()
                else:
                    self.flaggable_values[self.cols_ord_[rel_col]] = np.array(self._ord_mapping[rel_col])[cat_outl[cl]]

    def _print_no_outliers(self):
        print("No outliers found in input data.\n")

    def print_outliers(self, df_outliers, max_outliers = 10, min_decimals = 2):
        """
        Print outliers in readable format

        See the documentation for 'predict' for more details. This function will additionally
        perform some simplifications on the branch conditions, such as taking the smallest
        value when it is split two times by "<=". Can also pass a smaller DataFrame with
        only selected outliers to be printed.

        Parameters
        ----------
        df_outliers : DataFrame(n_rows, 6)
            DataFrame with outliers information as output by 'fit' or 'predict'.
        max_outliers : int
            Maximum number of outliers to print.
        min_decimals : int
            Minimum number of decimals to use when printing numeric values for the flagged
            outliers. The number of decimals will be dynamically increased according to the relative magnitudes of the
            values being reported. Ignored when passing ``max_outliers=0``.

        Returns
        -------
        None : None
            No return value.
        """
        if df_outliers.__class__.__name__ == "Series":
            df_outliers = df_outliers.to_frame().T
        self._check_valid_outliers_df(df_outliers)
        max_outliers = self._check_outliers_print(max_outliers, min_decimals)
        if not max_outliers:
            raise ValueError("Invalid value passed for 'max_outliers'.")

        ### sort according to conditions
        tot_outliers = (~pd.isnull(df_outliers.uses_NA_branch)).sum()
        df_outliers = df_outliers.sort_values(['uses_NA_branch', 'tree_depth', 'outlier_score'], ascending = True).head(max_outliers)

        ### check that there are actual outliers
        if pd.isnull(df_outliers["outlier_score"].iloc[0]):
            self._print_no_outliers()
            return None

        ### do not iterate over all rows
        df_outliers = df_outliers.loc[~pd.isnull(df_outliers.uses_NA_branch).to_numpy()]
        cnt = 0

        ## TODO: should have the number of decimal places as a parameter

        ### pretty-print
        print("Reporting top %d outliers [out of %d found]\n\n" % (df_outliers.shape[0], tot_outliers))
        for row in df_outliers.itertuples():

            min_dec_this = min_decimals

            ### first suspicious value
            ln_value = "row [%s]" % row.Index
            ln_value += " - suspicious column: [%s] - " % row.suspicious_value["column"]
            if np.in1d(row.suspicious_value["column"], self.cols_num_).max():
                if "decimals" in row.suspicious_value:
                    min_dec_this = max(min_dec_this, row.suspicious_value["decimals"])
                ln_value += ("suspicious value: [%%.%df]" % min_dec_this) % row.suspicious_value["value"]
            else:
                ln_value += "suspicious value: [%s]" % str(row.suspicious_value["value"])
            print(ln_value)

            ### then group statistics
            ln_group = ""
            if "mean" in row.group_statistics:
                ## numeric
                if "upper_thr" in row.group_statistics:
                    if np.in1d(row.suspicious_value["column"], self.cols_num_).max():
                        ln_group += (("\tdistribution: %%.%df%%%% <= %%.%df" % (3, min_dec_this))
                                                                     % (row.group_statistics["pct_below"] * 100,
                                                                        row.group_statistics["upper_thr"]))
                    else:
                        ln_group += "\tdistribution: %.3f%% <= [%s]" % (row.group_statistics["pct_below"] * 100,
                                                                        row.group_statistics["upper_thr"])
                else:
                    if np.in1d(row.suspicious_value["column"], self.cols_num_).max():
                        ln_group += (("\tdistribution: %%.%df%%%% >= %%.%df" % (3, min_dec_this))
                                                                     % (row.group_statistics["pct_above"] * 100,
                                                                        row.group_statistics["lower_thr"]))
                    else:
                        ln_group += "\tdistribution: %.3f%% >= [%s]" % (row.group_statistics["pct_above"] * 100,
                                                                        row.group_statistics["lower_thr"])
                
                if np.in1d(row.suspicious_value["column"], self.cols_num_).max():
                    ln_group += ((" - [mean: %%.%df] - [sd: %%.%df] - [norm. obs: %%d]" % (min_dec_this, min_dec_this))
                                                                                 % (row.group_statistics["mean"],
                                                                                    row.group_statistics["sd"],
                                                                                    row.group_statistics["n_obs"]))
                else:
                    ln_group += " - [mean: %s] - [norm. obs: %d]" % (row.group_statistics["mean"],
                                                                     row.group_statistics["n_obs"])

            elif "categs_common" in row.group_statistics:
                ## categorical
                if len(row.group_statistics["categs_common"]) == 1:
                    ln_group += "\tdistribution: %.3f%% = [%s]" % (row.group_statistics["pct_common"] * 100,
                                                                   str(row.group_statistics["categs_common"][0]))
                else:
                    ln_group += "\tdistribution: %.3f%% in [%s]" % (row.group_statistics["pct_common"] * 100,
                                                                    ", ".join([str(cat) for cat in row.group_statistics["categs_common"]]))
                if len(row.conditions) > 0:
                    ln_group += "\n\t( [norm. obs: %d] - [prior_prob: %.3f%%] - [next smallest: %.3f%%] )" % (row.group_statistics["n_obs"],
                                                                                                              row.group_statistics["prior_prob"] * 100,
                                                                                                              row.group_statistics["pct_next_most_comm"] * 100)
                else:
                    ln_group += "\n\t( [norm. obs: %d] - [next smallest: %.3f%%] )" % (row.group_statistics["n_obs"],
                                                                                       row.group_statistics["pct_next_most_comm"] * 100)
            elif "categ_maj" in row.group_statistics:
                ln_group += "\tdistribution: %.3f%% = [%s]" % (row.group_statistics["pct_common"] * 100,
                                                               str(row.group_statistics["categ_maj"]))
                ln_group += "\n\t( [norm. obs: %d] - [prior_prob: %.3f%%] )" % (row.group_statistics["n_obs"],
                                                                                row.group_statistics["prior_prob"] * 100)
            else:
                ## boolean
                ln_group += "\tdistribution: %.3f%% different [norm. obs: %d]" % (row.group_statistics["pct_other"] * 100,
                                                                                  row.group_statistics["n_obs"])
                if len(row.conditions) > 0:
                    ln_group += " - [prior_prob: %.3f%%]" % (row.group_statistics["prior_prob"] * 100)
            print(ln_group)

            ### then conditions
            if len(row.conditions) == 0:
                ln_cond = ""
            else:
                conditions = self._simplify_condition(row.conditions)
                ln_cond = "\tgiven:"
                for cond in conditions:

                    min_dec_this = max(min_decimals, cond["decimals"] if ("decimals" in cond) else 0)

                    if cond["comparison"] == "is NA":
                        ln_cond += "\n\t\t[%s] is NA" % cond["column"]
                    elif cond["comparison"] == "<=":
                        if np.in1d(cond["column"], self.cols_num_).max():
                            ln_cond += (("\n\t\t[%%s] <= [%%.%df] (value: %%.%df)" % (min_dec_this, min_dec_this))
                                                                            % (cond["column"],
                                                                               cond["value_comp"],
                                                                               cond["value_this"]))
                        else:
                            ln_cond += "\n\t\t[%s] <= [%s] (value: %s)" % (cond["column"],
                                                                           cond["value_comp"],
                                                                           cond["value_this"])
                    elif cond["comparison"] == ">":
                        if np.in1d(cond["column"], self.cols_num_).max():
                            ln_cond += (("\n\t\t[%%s] > [%%.%df] (value: %%.%df)" % (min_dec_this, min_dec_this))
                                                                           % (cond["column"],
                                                                              cond["value_comp"],
                                                                              cond["value_this"]))
                        else:
                            ln_cond += "\n\t\t[%s] > [%s] (value: %s)" % (cond["column"],
                                                                          cond["value_comp"],
                                                                          cond["value_this"])
                    
                    elif cond["comparison"] == "between":
                        if np.in1d(cond["column"], self.cols_num_).max():
                            ln_cond += (("\n\t\t[%%s] between (%%.%df, %%.%df] (value: %%.%df)" % (min_dec_this, min_dec_this, min_dec_this))
                                                                                       % (cond["column"],
                                                                                          cond["value_comp"][0],
                                                                                          cond["value_comp"][1],
                                                                                          cond["value_this"]))
                        else:
                            ln_cond += "\n\t\t[%s] between (%s, %s] (value: %s)" % (cond["column"],
                                                                                    str(cond["value_comp"][0]),
                                                                                    str(cond["value_comp"][1]),
                                                                                    cond["value_this"])

                    elif cond["comparison"] == "=":
                        ln_cond += "\n\t\t[%s] = [%s]" % (cond["column"], str(cond["value_comp"]))
                    elif cond["comparison"] == "!=":
                        ln_cond += "\n\t\t[%s] != [%s] (value: %s)" % (cond["column"], str(cond["value_comp"]), str(cond["value_this"]))
                    elif cond["comparison"] == "in":
                        ln_cond += "\n\t\t[%s] in [%s] (value: %s)" % (cond["column"],
                                                                       ", ".join([str(cat) for cat in cond["value_comp"]]),
                                                                       str(cond["value_this"]))
                    else:
                        raise ValueError("Unexpected error. Please open an issue in GitHub indicating what you were doing.")


            print(ln_cond)
            print("\n")
            cnt += 1
            if cnt >= max_outliers:
                break

    def _check_valid_outliers_df(self, df_outliers):
        required_cols = np.array([
                                    "suspicious_value", "group_statistics",
                                    "conditions",      "tree_depth",
                                    "uses_NA_branch",  "outlier_score"
                                ])
        if np.any(~np.in1d(required_cols, df_outliers.columns.values)):
            raise ValueError("DataFrame passed is not an output from this object's '.fit' or '.predict' methods.")

    def _simplify_condition(self, condition):
        ### look if there are repeated conditions for the same column
        cols_taken = [cond["column"] for cond in condition]
        if np.unique(cols_taken).shape[0] < len(cols_taken):
            repeated_cols, cnt_cols = np.unique(cols_taken, return_counts = True)
            repeated_cols = repeated_cols[cnt_cols > 1]
            replacing_cond = list()

            ### TODO: this is extremely inefficient, need to improve speed
            for cl in repeated_cols:
                n_le  = 0
                n_gt  = 0
                n_in  = 0
                n_eq  = 0
                n_neq = 0
                lowest_le   = None
                highest_gt  = None
                val_eq      = None
                val_neq     = None
                smallest_in = None
                highest_dec = 0

                for cn in condition:
                    if cn["column"] != cl:
                        continue
                    else:
                        val_this = cn["value_this"]

                    if cn["comparison"] == "<=":
                        n_le += 1
                        if lowest_le is None:
                            lowest_le = cn["value_comp"]
                        if cn["value_comp"] < lowest_le:
                            lowest_le = cn["value_comp"]
                        if "decimals" in cn:
                            highest_dec = max(highest_dec, cn["decimals"])
                    elif cn["comparison"] == ">":
                        n_gt += 1
                        if highest_gt is None:
                            highest_gt = cn["value_comp"]
                        if cn["value_comp"] > highest_gt:
                            highest_gt = cn["value_comp"]
                        if "decimals" in cn:
                            highest_dec = max(highest_dec, cn["decimals"])
                    elif cn["comparison"] == "in":
                        n_in += 1
                        if smallest_in is None:
                            smallest_in = np.array(cn["value_comp"])
                        else:
                            smallest_in = smallest_in[np.in1d(smallest_in, cn["value_comp"])]
                    elif cn["comparison"] == "=":
                        n_eq += 1
                        val_eq = cn["value_comp"]
                    elif cn["comparison"] == "!=":
                        n_neq += 1
                        val_neq = cn["value_comp"]
                if (n_le > 0) and (n_gt == 0):
                    replacing_cond.append({"column" : cl, "comparison" : "<=", "value_comp" : lowest_le,
                                           "value_this" : val_this, "decimals" : highest_dec})
                elif (n_gt > 0) and (n_le == 0):
                    replacing_cond.append({"column" : cl, "comparison" : ">", "value_comp" : highest_gt,
                                           "value_this" : val_this, "decimals" : highest_dec})
                elif (n_le > 0) and (n_gt > 0):
                    replacing_cond.append({"column" : cl, "comparison" : "between", "value_comp" : [highest_gt, lowest_le],
                                           "value_this" : val_this, "decimals" : highest_dec})
                elif (n_in > 0) and (n_eq == 0) and (n_neq == 0):
                    replacing_cond.append({"column" : cl, "comparison" : "in", "value_comp" : smallest_in, "value_this" : val_this})
                elif (n_in > 0) and (n_eq > 0):
                    replacing_cond.append({"column" : cl, "comparison" : "=", "value_comp" : val_eq, "value_this" : val_this})
                elif (n_in > 0) and (n_neq > 0):
                    replacing_cond.append({"column" : cl, "comparison" : "!=", "value_comp" : val_neq, "value_this" : val_this})

            condition = [cond for cond in condition if not (cond["column"] in repeated_cols)] + replacing_cond


        ### simplify to equals if there are more possible
        for cn in range(len(condition)):
            if condition[cn]["comparison"] == "in":
                if isinstance(condition[cn]["value_comp"], list):
                    if len(condition[cn]["value_comp"]) == 1:
                        condition[cn]["value_comp"] = condition[cn]["value_comp"][0]
                        condition[cn]["comparison"] = "="
                else:
                    if np.array(condition[cn]["value_comp"]).shape[0] == 1:
                        condition[cn]["value_comp"] = np.array(condition[cn]["value_comp"])[0]
                        condition[cn]["comparison"] = "="

        return condition[::-1]

    def generate_gritbot_files(self, df, cols_ignore = None, save_folder = ".", file_name = "data"):
        """
        Generate data files for GritBot software

        Generates CSV (.data) and naming (.names) files as required for use by the GritBot
        software (not included in this Python package). Note that this only generates the
        data files and doesn't do anything else.

        Parameters
        ----------
        df : DataFrame(nrows, ncols)
            Pandas DataFrame in the same format as requried by '.fit'.
        cols_ignore : boolean array(n_cols, ) or string array(n_ignore, )
            Array containing columns which will not be split, but will be evaluated
            for usage in splitting other columns. Pass 'None' to use all columns.
        save_folder : str
            Path to folder where to save the generated files.
        file_name : str
            Prefix for the file names (before the dot).

        References
        ----------
        .. [1] GritBot software : https://www.rulequest.com/gritbot-info.html
        """
        save_folder = os.path.expanduser(save_folder)
        assert os.path.exists(save_folder)
        temp_obj = OutlierTree()
        temp_obj._check_valid_data(df)
        row_names = df.index.copy()
        arr_num, arr_cat, arr_ord = temp_obj._split_types(df)
        cols_ignore = temp_obj._process_cols_ignore(df, cols_ignore)
        # return cols_ignore

        ### now re-add metadata and re-code categoricals and time series
        df_out = pd.DataFrame(arr_num)
        df_out.columns = np.r_[temp_obj.cols_num_, temp_obj.cols_ts_]
        for cl in range(temp_obj.cols_ts_.shape[0]):
            df_out[temp_obj.cols_ts_[cl]] = pd.Series(df[temp_obj.cols_ts_[cl]]).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        for cl in range(temp_obj.cols_cat_.shape[0]):
            df_out[temp_obj.cols_cat_[cl]] = df[temp_obj.cols_cat_[cl]].to_numpy()
        for cl in range(temp_obj.cols_bool_.shape[0]):
            df_out[temp_obj.cols_bool_[cl]] = df[temp_obj.cols_bool_[cl]].astype("bool").to_numpy()
        for cl in range(temp_obj.cols_ord_.shape[0]):
            df_out[temp_obj.cols_ord_[cl]] = df[temp_obj.cols_ord_[cl]].to_numpy()
        df_out.index = row_names

        ### generate CSV file
        df_out.to_csv(os.path.join(save_folder, file_name + ".data"), index = True, header = False)

        ### generate names file
        with open(os.path.join(save_folder, file_name + ".names"), "w") as f_out:
            f_out.write(str(df_out.columns[0]) + ".\n\n") ## redundant line, but required by gritbot
            for num_cl in temp_obj.cols_num_:
                f_out.write(str(num_cl) + ": continuous.\n")
            for ts_cl in temp_obj.cols_ts_:
                f_out.write(str(ts_cl) + ": timestamp.\n")
            for cat_cl in temp_obj.cols_cat_:
                outstr = str(cat_cl) + ":  "
                categs = df_out[cat_cl].dropna().unique()
                for cat in categs:
                    outstr += str(cat) + ", "
                outstr = re.sub(", $", ".\n", outstr)
                f_out.write(outstr)
            for bool_cl in temp_obj.cols_bool_:
                f_out.write(str(bool_cl) + ": True, False.\n")
            for ord_cl in temp_obj.cols_ord_:
                outstr = str(ord_cl) + ": [ordered] "
                levels = pd.Categorical(df_out[ord_cl]).categories
                for lev in levels:
                    outstr += str(lev) + ", "
                outstr = re.sub(", $", ".\n", outstr)
                f_out.write(outstr)

            if cols_ignore.shape[0] > 0:
                outstr = "\n\nattributes excluded: "
                cols_concat = np.r_[
                                    temp_obj.cols_num_,
                                    temp_obj.cols_ts_,
                                    temp_obj.cols_cat_,
                                    temp_obj.cols_bool_,
                                    temp_obj.cols_ord_
                                    ]
                cols_ignore = cols_ignore.astype(int)
                for cl in range(cols_ignore.shape[0]):
                    if cols_ignore[cl] != 0:
                        outstr += str(cols_concat[cl]) + ", "
                outstr = re.sub(", $", ".\n", outstr)
                f_out.write(outstr)

    def _get_model_outputs(self):
        return self._outlier_cpp_obj.get_model_outputs()
