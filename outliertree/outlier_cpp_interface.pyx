#cython: auto_pickle=True
import  numpy as np
cimport numpy as np
from libcpp cimport bool as bool_t ###don't confuse it with Python bool
from libcpp.vector cimport vector
from cython cimport boundscheck, nonecheck, wraparound
import ctypes

cdef extern from "outlier_tree.hpp":

    ctypedef enum ColType:
        Numeric
        Categorical
        Ordinal
        NoType

    ctypedef enum SplitType:
        LessOrEqual
        Greater
        Equal
        NotEqual
        InSubset
        NotInSubset
        SingleCateg
        SubTrees
        IsNa
        Root

    ctypedef enum ColTransf:
        NoTransf
        Log
        Exp

    ctypedef struct Cluster:
        ColType      column_type
        size_t       col_num
        SplitType    split_type
        double       split_point
        vector[signed char] split_subset
        int          split_lev
        bool_t       has_NA_branch

        size_t    cluster_size
        double    lower_lim
        double    upper_lim
        double    perc_below
        double    perc_above
        double    display_lim_low
        double    display_lim_high
        double    display_mean
        double    display_sd
        vector[signed char] subset_common
        double    perc_in_subset
        double    perc_next_most_comm
        int       categ_maj

        double    cluster_mean
        double    cluster_sd
        vector[double] score_categ

    ctypedef struct ClusterTree:
        size_t parent
        SplitType parent_branch
        vector[size_t] clusters

        SplitType split_this_branch
        vector[size_t] all_branches

        ColType   column_type
        size_t    col_num
        double split_point
        vector[signed char] split_subset
        int split_lev

        size_t tree_NA
        size_t tree_left
        size_t tree_right
        vector[size_t] binary_branches

    ctypedef struct ModelOutputs:
        vector[vector[ClusterTree]] all_trees
        vector[vector[Cluster]]     all_clusters
        vector[double] outlier_scores_final
        vector[size_t] outlier_clusters_final
        vector[size_t] outlier_columns_final
        vector[size_t] outlier_trees_final
        vector[size_t] outlier_depth_final
        vector[int] outlier_decimals_distr
        vector[size_t] start_ix_cat_counts
        vector[long double] prop_categ
        vector[ColTransf] col_transf
        vector[double] transf_offset
        vector[double] sd_div
        vector[int]    min_decimals_col
        vector[int]    ncat
        vector[int]    ncat_ord
        size_t ncols_numeric
        size_t ncols_categ
        size_t ncols_ord
        vector[double] min_outlier_any_cl
        vector[double] max_outlier_any_cl
        vector[vector[bool_t]] cat_outlier_any_cl
        size_t max_depth

    bool_t get_has_openmp() nogil except +

    bool_t fit_outliers_models(ModelOutputs &model_outputs,
                               double *numeric_data,     size_t ncols_numeric,
                               int    *categorical_data, size_t ncols_categ,   int *ncat,
                               int    *ordinal_data,     size_t ncols_ord,     int *ncat_ord,
                               size_t nrows, char *cols_ignore, int nthreads,
                               bool_t categ_as_bin, bool_t ord_as_bin, bool_t cat_bruteforce_subset, bool_t categ_from_maj, bool_t take_mid,
                               size_t max_conditions, double max_perc_outliers, size_t min_size_numeric, size_t min_size_categ,
                               double min_gain, bool_t gain_as_pct, bool_t follow_all, double z_norm, double z_outlier) nogil except +

    bool_t find_new_outliers(double *numeric_data,
                             int    *categorical_data,
                             int    *ordinal_data,
                             size_t nrows, int nthreads, ModelOutputs &model_outputs) nogil except +

    void forget_row_outputs(ModelOutputs &model_outputs)

    void check_more_two_values(double *arr_num, size_t nrows, size_t ncols, int nthreads, char *too_few_values)

    void dealloc_ModelOutputs(ModelOutputs &model_outputs)

    ModelOutputs get_empty_ModelOutputs()

    bool_t cy_check_interrupt_switch()

    void cy_tick_off_interrupt_switch()

def _get_has_openmp():
    return get_has_openmp()

def check_few_values(np.ndarray[double, ndim = 2] arr, int nthreads = 1):
    cdef size_t nrows = arr.shape[0]
    cdef size_t ncols = arr.shape[1]
    cdef np.ndarray[char, ndim = 1] too_few_values = np.zeros(ncols, dtype = ctypes.c_char)
    cdef char *ptr_too_few_values = &too_few_values[0]
    if not arr.flags['F_CONTIGUOUS']:
        arr = np.asfortranarray(arr)
    check_more_two_values(&arr[0, 0], nrows, ncols, nthreads, ptr_too_few_values)
    too_few_values_bool = np.empty(ncols, dtype = "bool")
    for col in range(ncols):
        too_few_values_bool[col] = <bool_t> ptr_too_few_values[col]
    return too_few_values_bool


cdef class OutlierCppObject:
    cdef ModelOutputs model_outputs

    def __init__(self):
        dealloc_ModelOutputs(self.model_outputs)
        self.model_outputs = get_empty_ModelOutputs()

    def __dealloc__(self):
        dealloc_ModelOutputs(self.model_outputs)

    def get_model_outputs(self):
        return self.model_outputs

    def get_n_clust_and_trees(self):
        nclust = sum([self.model_outputs.all_clusters[cl].size() for cl in range(self.model_outputs.all_clusters.size())])
        ntrees = sum([self.model_outputs.all_trees[cl].size() for cl in range(self.model_outputs.all_trees.size())])
        return nclust, ntrees

    def get_flaggable_bounds(self):
        return  np.array(self.model_outputs.min_outlier_any_cl), \
                np.array(self.model_outputs.max_outlier_any_cl), \
                [np.array(cl) for cl in self.model_outputs.cat_outlier_any_cl]

    def fit_model(self,
                  np.ndarray[double, ndim = 2] arr_num, np.ndarray[int, ndim = 2] arr_cat, np.ndarray[int, ndim = 2] arr_ord,
                  np.ndarray[int, ndim = 1] ncat, np.ndarray[int, ndim = 1] ncat_ord, cols_ignore,
                  size_t ncols_true_numeric = 0, size_t ncols_true_ts = 0, size_t ncols_true_cat = 0, size_t ncols_true_bool = 0,
                  colnames_num = [], colnames_cat = [], colnames_ord = [], levs_cat = [], levs_ord = [], ts_min = None,
                  bool_t return_outliers = 1, int nthreads = 1, bool_t categ_as_bin = 1, bool_t ord_as_bin = 1,
                  bool_t cat_bruteforce_subset = 0, bool_t categ_from_maj = 0, bool_t take_mid = 1, size_t max_conditions = 4,
                  double max_perc_outliers = 0.01, size_t min_size_numeric = 25, size_t min_size_categ = 75, double min_gain = 0.001,
                  bool_t follow_all = 0, bool_t gain_as_pct = 1, double z_norm = 2.67, double z_outlier = 8.0,
                  out_df = None):

        cdef size_t nrows = np.max([arr_num.shape[0], arr_cat.shape[0], arr_ord.shape[0]])
        cdef size_t ncol_num = arr_num.shape[1]
        cdef size_t ncol_cat = arr_cat.shape[1]
        cdef size_t ncol_ord = arr_ord.shape[1]

        cdef double *ptr_arr_num      = NULL
        cdef int    *ptr_arr_cat      = NULL
        cdef int    *ptr_arr_ord      = NULL
        cdef int    *ptr_arr_ncat     = NULL
        cdef int    *ptr_arr_ncat_ord = NULL
        cdef char   *ptr_cols_ignore  = NULL
        cdef int iiii

        if ncol_num > 0:
            ptr_arr_num = &arr_num[0, 0]
        if ncol_cat > 0:
            ptr_arr_cat = &arr_cat[0, 0]
            ptr_arr_ncat = &ncat[0]
        if ncol_ord > 0:
            ptr_arr_ord = &arr_ord[0, 0]
            ptr_arr_ncat_ord = &ncat_ord[0]
        cdef np.ndarray[char, ndim = 1] cols_ignore_c = np.zeros(cols_ignore.shape[0], dtype = ctypes.c_char)
        cdef size_t cl_ix
        if cols_ignore.shape[0] > 0:
            ptr_cols_ignore = &cols_ignore_c[0]
            for cl_ix in range(cols_ignore.shape[0]):
                cols_ignore_c[cl_ix] = <bool_t> cols_ignore[cl_ix]

        cdef bool_t found_outliers
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            found_outliers = fit_outliers_models(
                                                self.model_outputs,
                                                ptr_arr_num, ncol_num,
                                                ptr_arr_cat, ncol_cat, ptr_arr_ncat,
                                                ptr_arr_ord, ncol_ord, ptr_arr_ncat_ord,
                                                nrows, ptr_cols_ignore, nthreads,
                                                categ_as_bin, ord_as_bin, cat_bruteforce_subset, categ_from_maj, take_mid,
                                                max_conditions, max_perc_outliers, min_size_numeric, min_size_categ,
                                                min_gain, gain_as_pct, follow_all, z_norm, z_outlier
                                            )

        if cy_check_interrupt_switch():
            cy_tick_off_interrupt_switch()
            raise InterruptedError("Error: procedure was interrupted.")

        if not return_outliers:
            forget_row_outputs(self.model_outputs)
            return found_outliers, None

        if found_outliers:
            ### fill output data frame with the outliers
            out_df = self.create_df(arr_num, arr_cat, arr_ord,
                                    ncols_true_numeric, ncols_true_ts, ncols_true_cat, ncols_true_bool,
                                    colnames_num, colnames_cat, colnames_ord, levs_cat, levs_ord, ts_min,
                                    nthreads, out_df)

        forget_row_outputs(self.model_outputs)
        return found_outliers, out_df


    def predict(self, 
                np.ndarray[double, ndim = 2] arr_num, np.ndarray[int, ndim = 2] arr_cat, np.ndarray[int, ndim = 2] arr_ord,
                size_t ncols_true_numeric = 0, size_t ncols_true_ts = 0, size_t ncols_true_cat = 0, size_t ncols_true_bool = 0,
                colnames_num = [], colnames_cat = [], colnames_ord = [], levs_cat = [], levs_ord = [], ts_min = None,
                int nthreads = 1, out_df = None):

        cdef size_t nrows = np.max([arr_num.shape[0], arr_cat.shape[0], arr_ord.shape[0]])
        cdef double *ptr_arr_num      = NULL
        cdef int    *ptr_arr_cat      = NULL
        cdef int    *ptr_arr_ord      = NULL

        if arr_num.shape[0] > 0:
            ptr_arr_num = &arr_num[0, 0]
        if arr_cat.shape[0] > 0:
            ptr_arr_cat = &arr_cat[0, 0]
        if arr_ord.shape[0] > 0:
            ptr_arr_ord = &arr_ord[0, 0]

        cdef bool_t found_outliers
        with nogil, boundscheck(False), nonecheck(False), wraparound(False):
            found_outliers = find_new_outliers(
                                                ptr_arr_num,
                                                ptr_arr_cat,
                                                ptr_arr_ord,
                                                nrows, nthreads,
                                                self.model_outputs
                                            )

        
        if found_outliers:
            out_df = self.create_df(arr_num, arr_cat, arr_ord,
                                    ncols_true_numeric, ncols_true_ts, ncols_true_cat, ncols_true_bool,
                                    colnames_num, colnames_cat, colnames_ord, levs_cat, levs_ord, ts_min,
                                    nthreads, out_df)
        forget_row_outputs(self.model_outputs)
        return found_outliers, out_df


    def create_df(self,
                  np.ndarray[double, ndim = 2] arr_num, np.ndarray[int, ndim = 2] arr_cat, np.ndarray[int, ndim = 2] arr_ord,
                  size_t ncols_true_numeric = 0, size_t ncols_true_ts = 0, size_t ncols_true_cat = 0, size_t ncols_true_bool = 0,
                  colnames_num = [], colnames_cat = [], colnames_ord = [], levs_cat = [], levs_ord = [], ts_min = None,
                  int nthreads = 1, out_df = None):

        cdef size_t nrows = np.max([arr_num.shape[0], arr_cat.shape[0], arr_ord.shape[0]])
        cdef size_t ncol_num = arr_num.shape[1]
        cdef size_t ncol_cat = arr_cat.shape[1]
        cdef size_t ncol_ord = arr_ord.shape[1]

        ### TODO: parallelize this using joblib (cannot use cython prange as it has non-typed python code)
        cdef size_t row
        cdef size_t outl_col, outl_clust, outl_tree, curr_tree, parent_tree, cat_col_num
        cdef int coldecim
        for row in range(self.model_outputs.outlier_scores_final.size()):
            if self.model_outputs.outlier_scores_final[row] < 1:

                outl_col   = self.model_outputs.outlier_columns_final[row]
                outl_clust = self.model_outputs.outlier_clusters_final[row]
                outl_tree  = self.model_outputs.outlier_trees_final[row]

                ### info used to rank outlierness
                out_df.iat[row, 3] = self.model_outputs.outlier_depth_final[row]
                out_df.iat[row, 4] = bool(self.model_outputs.all_clusters[outl_col][outl_clust].has_NA_branch)
                out_df.iat[row, 5] = self.model_outputs.outlier_scores_final[row]

                ### info about the suspicious value
                if outl_col < ncol_num:
                    colname = colnames_num[outl_col]
                    if outl_col < ncols_true_numeric:
                        out_df.iat[row, 0] = {
                                                "column" : colnames_num[outl_col],
                                                "value" : arr_num[row, outl_col],
                                                "decimals" : self.model_outputs.outlier_decimals_distr[row]
                                            }
                    else:
                        out_df.iat[row, 0] = {
                                                "column" : colnames_num[outl_col],
                                                "value" : np.datetime64(np.int(arr_num[row, outl_col] - 1 + ts_min[outl_col - ncols_true_numeric]), "s")
                                            }
                elif outl_col < (ncol_num + ncol_cat):
                    colname = colnames_cat[outl_col - ncol_num]
                    if outl_col < (ncol_num + ncols_true_cat):
                        out_df.iat[row, 0] = {"column" : colnames_cat[outl_col - ncol_num], "value" : levs_cat[outl_col - ncol_num][arr_cat[row, outl_col - ncol_num]]}
                    else:
                        out_df.iat[row, 0] = {"column" : colnames_cat[outl_col - ncol_num], "value" : bool(arr_cat[row, outl_col - ncol_num])}
                else:
                    colname = colnames_ord[outl_col - ncol_num - ncol_cat]
                    out_df.iat[row, 0] = {
                                            "column" : colnames_ord[outl_col - ncol_num - ncol_cat],
                                            "value" : levs_ord[outl_col - ncol_num - ncol_cat][arr_ord[row, outl_col - ncol_num - ncol_cat]]
                                        }

                ### info about the normal observations in tree branch
                if outl_col < ncol_num:

                    if arr_num[row, outl_col] >= self.model_outputs.all_clusters[outl_col][outl_clust].upper_lim:
                        compar_pct = self.model_outputs.all_clusters[outl_col][outl_clust].perc_below
                        compar_thr = self.model_outputs.all_clusters[outl_col][outl_clust].display_lim_high
                        if outl_col < ncols_true_numeric:
                            out_df.iat[row, 1] = {
                                                    "upper_thr" : compar_thr,
                                                    "pct_below" : compar_pct,
                                                    "mean"  : self.model_outputs.all_clusters[outl_col][outl_clust].display_mean,
                                                    "sd"    : self.model_outputs.all_clusters[outl_col][outl_clust].display_sd,
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }
                        else:
                            out_df.iat[row, 1] = {
                                                    "upper_thr" : np.datetime64(np.int(compar_thr - 1 + ts_min[outl_col - ncols_true_numeric]), "s"),
                                                    "pct_below" : compar_pct,
                                                    "mean"  : np.datetime64(np.int(self.model_outputs.all_clusters[outl_col][outl_clust].display_mean \
                                                                                   - 1 + ts_min[outl_col - ncols_true_numeric]), "s"),
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }

                    else:
                        compar_pct = self.model_outputs.all_clusters[outl_col][outl_clust].perc_above
                        compar_thr = self.model_outputs.all_clusters[outl_col][outl_clust].display_lim_low
                        if outl_col < ncols_true_numeric:
                            out_df.iat[row, 1] = {
                                                    "lower_thr" : compar_thr,
                                                    "pct_above" : compar_pct,
                                                    "mean"  : self.model_outputs.all_clusters[outl_col][outl_clust].display_mean,
                                                    "sd"    : self.model_outputs.all_clusters[outl_col][outl_clust].display_sd,
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }
                        else:
                            out_df.iat[row, 1] = {
                                                    "lower_thr" : np.datetime64(np.int(compar_thr - 1 + ts_min[outl_col - ncols_true_numeric]), "s"),
                                                    "pct_above" : compar_pct,
                                                    "mean"  : np.datetime64(np.int(self.model_outputs.all_clusters[outl_col][outl_clust].display_mean \
                                                                                   - 1 + ts_min[outl_col - ncols_true_numeric]), "s"),
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }

                elif outl_col < (ncol_num + ncol_cat):
                    if outl_col < (ncol_num + ncols_true_cat):
                        if self.model_outputs.all_clusters[outl_col][outl_clust].categ_maj < 0:
                            out_df.iat[row, 1] = {
                                                    "categs_common" : levs_cat[outl_col - ncol_num][
                                                                            (np.array(self.model_outputs.all_clusters[
                                                                                outl_col][outl_clust
                                                                             ].subset_common) == 0).astype("bool")
                                                                            ],
                                                    "pct_common" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                                    "pct_next_most_comm" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm,
                                                    "prior_prob" : self.model_outputs.prop_categ[
                                                                            self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                            arr_cat[row, outl_col - ncol_num]
                                                                            ],
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }
                        else:
                            out_df.iat[row, 1] = {
                                                    "categ_maj" : levs_cat[outl_col - ncol_num][
                                                                        self.model_outputs.all_clusters[outl_col][outl_clust].categ_maj
                                                                        ],
                                                    "pct_common" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                                    "prior_prob" : self.model_outputs.prop_categ[
                                                                            self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                            arr_cat[row, outl_col - ncol_num]
                                                                            ],
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size,
                                                }
                    else:
                        out_df.iat[row, 1] = {
                                                "pct_other"  : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                                "prior_prob" : self.model_outputs.prop_categ[
                                                                        self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                        arr_cat[row, outl_col - ncol_num]
                                                                        ],
                                                "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                            }

                    if (self.model_outputs.all_clusters[outl_col][outl_clust].split_type == Root):
                        del out_df.iat[row, 1]["prior_prob"]

                else:

                    if self.model_outputs.all_clusters[outl_col][outl_clust].categ_maj < 0:
                        out_df.iat[row, 1] = {
                                                "categs_common" : levs_ord[outl_col - ncol_num - ncol_cat][
                                                                        (np.array(self.model_outputs.all_clusters[
                                                                            outl_col][outl_clust
                                                                         ].subset_common) == 0).astype("bool")
                                                                        ],
                                                "pct_common" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                                "pct_next_most_comm" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm,
                                                "prior_prob" : self.model_outputs.prop_categ[
                                                                        self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                        arr_ord[row, outl_col - ncol_num - ncol_cat]
                                                                        ],
                                                "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                            }
                    else:
                        out_df.iat[row, 1] = {
                                                    "categ_maj" : levs_ord[outl_col - ncol_num - ncol_cat][
                                                                        self.model_outputs.all_clusters[outl_col][outl_clust].categ_maj
                                                                        ],
                                                    "pct_common" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                                    "prior_prob" : self.model_outputs.prop_categ[
                                                                        self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                        arr_ord[row, outl_col - ncol_num - ncol_cat]
                                                                        ],
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }

                    if (self.model_outputs.all_clusters[outl_col][outl_clust].split_type == Root):
                        del out_df.iat[row, 1]["prior_prob"]


                ### info about the conditions - first from the cluster (final branch in the tree, need to follow it back to the root afterwards)
                if self.model_outputs.all_clusters[outl_col][outl_clust].column_type != NoType:
                    if self.model_outputs.all_clusters[outl_col][outl_clust].split_type == IsNa:
                        colcond  = "is NA"
                        condval  = np.nan
                        colval   = np.nan

                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == LessOrEqual:
                        if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric:
                            colcond = "<="
                            if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_numeric:
                                condval = self.model_outputs.all_clusters[outl_col][outl_clust].split_point
                            else:
                                condval = np.datetime64(
                                                        np.int(
                                                                self.model_outputs.all_clusters[outl_col][outl_clust].split_point \
                                                                - 1 \
                                                                + ts_min[self.model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_true_numeric]
                                                            ),
                                                        "s")
                        else:
                            colcond = "in"
                            condval = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]\
                                              [:(self.model_outputs.all_clusters[outl_col][outl_clust].split_lev + 1)]

                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == Greater:
                        if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric:
                            colcond = ">"
                            if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_numeric:
                                condval = self.model_outputs.all_clusters[outl_col][outl_clust].split_point
                            else:
                                condval = np.datetime64(
                                                        np.int(
                                                                self.model_outputs.all_clusters[outl_col][outl_clust].split_point \
                                                                - 1 \
                                                                + ts_min[self.model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_true_numeric]
                                                            ),
                                                        "s")
                        else:
                            colcond = "in"
                            condval = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]\
                                              [(self.model_outputs.all_clusters[outl_col][outl_clust].split_lev + 1):]


                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == InSubset:
                        if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < (ncol_num + ncols_true_cat):
                            levs = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                            colcond = "in"
                            condval = [levs[lv] for lv in range(len(levs)) if self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[lv] > 0]
                        else:
                            ### Note: this is redundant
                            colcond = "="
                            condval = bool(self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[1])

                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == NotInSubset:
                        if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < (ncol_num + ncols_true_cat):
                            levs = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                            colcond = "in"
                            condval = [levs[lv] for lv in range(len(levs)) if not (self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[lv] > 0)]
                        else:
                            ### Note: this is redundant
                            colcond = "="
                            condval = not bool(self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[1])


                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == Equal:
                        colcond = "="
                        if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Categorical:
                            if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_cat:
                                condval = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num][self.model_outputs.all_clusters[outl_col][outl_clust].split_lev]
                            else:
                                condval = bool(self.model_outputs.all_clusters[outl_col][outl_clust].split_lev)
                        else:
                            condval = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num][self.model_outputs.all_clusters[outl_col][outl_clust].split_lev]

                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == NotEqual:
                        colcond = "!="
                        if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Categorical:
                            if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_cat:
                                condval = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num][self.model_outputs.all_clusters[outl_col][outl_clust].split_lev]
                            else:
                                condval = not bool(self.model_outputs.all_clusters[outl_col][outl_clust].split_lev)
                        else:
                            condval = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num][self.model_outputs.all_clusters[outl_col][outl_clust].split_lev]
                            

                    ### add the column name and actual value for the row
                    coldecim = 0
                    if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric:
                        cond_col = colnames_num[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                        colval   = arr_num[row, self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                        if self.model_outputs.all_clusters[outl_col][outl_clust].col_num >= ncols_true_numeric:
                            colval = np.datetime64(np.int(colval - 1 + ts_min[self.model_outputs.all_clusters[outl_col][outl_clust].col_num - ncols_true_numeric]), "s")
                        else:
                            coldecim = self.model_outputs.min_decimals_col[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                    elif self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Categorical:
                        cond_col = colnames_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                        if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_cat:
                            colval = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]\
                                             [arr_cat[row, self.model_outputs.all_clusters[outl_col][outl_clust].col_num]]
                        else:
                            colval = bool(arr_cat[row, self.model_outputs.all_clusters[outl_col][outl_clust].col_num])
                    else:
                        cond_col = colnames_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                        colval   = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]\
                                           [arr_ord[row, self.model_outputs.all_clusters[outl_col][outl_clust].col_num]]

                    out_df.iat[row, 2].append({
                                                "column"     : cond_col,
                                                "comparison" : colcond,
                                                "value_comp" : condval,
                                                "value_this" : colval,
                                                "decimals"   : coldecim
                                            })

                ### now add all the conditions from the tree branches that lead to the cluster
                curr_tree = outl_tree
                while True:

                    if curr_tree == 0 or self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == SubTrees:
                        break
                    parent_tree = self.model_outputs.all_trees[outl_col][curr_tree].parent

                    ### when using 'follow_all'
                    if self.model_outputs.all_trees[outl_col][parent_tree].all_branches.size() > 0:

                        if self.model_outputs.all_trees[outl_col][curr_tree].column_type == Numeric:
                            if self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == IsNa:
                                colcond = "is NA"
                                condval = np.nan
                                colval  = np.nan
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == LessOrEqual:
                                colcond = "<="
                                condval = self.model_outputs.all_trees[outl_col][curr_tree].split_point
                                colval  = arr_num[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                                if self.model_outputs.all_trees[outl_col][curr_tree].col_num >= ncols_true_numeric:
                                    condval = np.datetime64(np.int(condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_true_numeric]), "s")
                                    colval  = np.datetime64(np.int(colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_true_numeric]),  "s")
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Greater:
                                colcond = ">"
                                condval = self.model_outputs.all_trees[outl_col][curr_tree].split_point
                                colval  = arr_num[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                                if self.model_outputs.all_trees[outl_col][curr_tree].col_num >= ncols_true_numeric:
                                    condval = np.datetime64(np.int(condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_true_numeric]), "s")
                                    colval  = np.datetime64(np.int(colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num - ncols_true_numeric]),  "s")

                        elif self.model_outputs.all_trees[outl_col][curr_tree].column_type == Categorical:
                            if self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == IsNa:
                                colcond = "is NA"
                                condval = np.nan
                                colval  = np.nan
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == InSubset:
                                colcond = "in"
                                levs    = levs_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                                condval = [levs[i] for i in range(len(levs)) if self.model_outputs.all_trees[outl_col][curr_tree].split_subset[i] > 0]
                                colval  = levs[arr_cat[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == NotInSubset:
                                colcond = "in"
                                levs    = levs_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                                condval = [levs[i] for i in range(len(levs)) if not (self.model_outputs.all_trees[outl_col][curr_tree].split_subset[i] > 0)]
                                colval  = levs[arr_cat[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Equal:
                                colcond = "="
                                if self.model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_true_cat:
                                    condval = levs_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num][self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                    colval  = condval
                                else:
                                    condval = bool(self.model_outputs.all_trees[outl_col][curr_tree].split_lev)
                                    colval  = condval
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == NotEqual:
                                colcond = "!="
                                condval = levs_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                colval  = levs_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [arr_cat[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]
                                ### Note: booleans and binaries always get translated to "=" and swapped

                        elif self.model_outputs.all_trees[outl_col][curr_tree].column_type == Ordinal:
                            if self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == IsNa:
                                colcond = "is NA"
                                condval = np.nan
                                colval  = np.nan
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == LessOrEqual:
                                colcond = "in"
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [:(self.model_outputs.all_trees[outl_col][curr_tree].split_lev + 1)]
                                colval  = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [arr_ord[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Greater:
                                colcond = "in"
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [(self.model_outputs.all_trees[outl_col][curr_tree].split_lev + 1):]
                                colval  = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [arr_ord[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Equal:
                                colcond = "="
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                colval  = condval
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == NotEqual:
                                colcond = "!="
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                colval  = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]\
                                                  [arr_ord[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == IsNa:
                        colcond = "is NA"
                        condval = np.nan
                        colval  = np.nan

                    ### regular case (when *not* using 'follow_all')
                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == LessOrEqual:
                        if self.model_outputs.all_trees[outl_col][parent_tree].column_type == Numeric:
                            colcond = "<="
                            condval = self.model_outputs.all_trees[outl_col][parent_tree].split_point
                            colval  = arr_num[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                            if self.model_outputs.all_trees[outl_col][parent_tree].col_num >= ncols_true_numeric:
                                condval = np.datetime64(np.int(condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_true_numeric]), "s")
                                colval  = np.datetime64(np.int(colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_true_numeric]),  "s")
                        else:
                            colcond = "in"
                            condval = levs_ord[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                              [:(self.model_outputs.all_trees[outl_col][parent_tree].split_lev + 1)]
                            colval  = levs_ord[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                              [arr_ord[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]]

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == Greater:
                        if self.model_outputs.all_trees[outl_col][parent_tree].column_type == Numeric:
                            colcond = ">"
                            condval = self.model_outputs.all_trees[outl_col][parent_tree].split_point
                            colval  = arr_num[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                            if self.model_outputs.all_trees[outl_col][parent_tree].col_num >= ncols_true_numeric:
                                condval = np.datetime64(np.int(condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_true_numeric]), "s")
                                colval  = np.datetime64(np.int(colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num - ncols_true_numeric]),  "s")
                        else:
                            colcond = "in"
                            condval = levs_ord[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                              [(self.model_outputs.all_trees[outl_col][parent_tree].split_lev + 1):]
                            colval  = levs_ord[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                              [arr_ord[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]]

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == InSubset:
                        levs = levs_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                        colcond = "in"
                        if self.model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_true_cat:
                            condval = [levs[lv] for lv in range(len(levs)) if self.model_outputs.all_trees[outl_col][parent_tree].split_subset[lv] > 0]
                            colval  = levs[arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]]
                        else:
                            condval = bool(self.model_outputs.all_trees[outl_col][parent_tree].split_subset[1])
                            colval  = bool(arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num])

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == NotInSubset:
                        levs = levs_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                        colcond = "in"
                        if self.model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_true_cat:
                            condval = [levs[lv] for lv in range(len(levs)) if not (self.model_outputs.all_trees[outl_col][parent_tree].split_subset[lv] > 0)]
                            colval  = levs[arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]]
                        else:
                            condval = bool(self.model_outputs.all_trees[outl_col][parent_tree].split_subset[0])
                            colval  = bool(arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num])

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == Equal:
                        colcond = "="
                        if self.model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_true_cat:
                            condval = levs_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                              [self.model_outputs.all_trees[outl_col][parent_tree].split_lev]
                        else:
                            condval = bool(self.model_outputs.all_trees[outl_col][parent_tree].split_lev)
                        colval  = condval

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == NotEqual:
                        colcond = "!="
                        condval = levs_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                          [self.model_outputs.all_trees[outl_col][parent_tree].split_lev]
                        colval  = levs_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                          [arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]]

                    elif self.model_outputs.all_trees[outl_col][curr_tree].parent_branch == SingleCateg:
                        colcond = "="
                        if self.model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_true_cat:
                            condval = levs_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]\
                                              [arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num]]
                        else:
                            condval = bool(arr_cat[row, self.model_outputs.all_trees[outl_col][parent_tree].col_num])
                        colval  = condval


                    ### add column name
                    coldecim = 0
                    if self.model_outputs.all_trees[outl_col][parent_tree].all_branches.size() == 0:
                        if self.model_outputs.all_trees[outl_col][parent_tree].column_type == Numeric:
                            cond_col = colnames_num[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                            if self.model_outputs.all_trees[outl_col][parent_tree].col_num < ncols_true_numeric:
                                coldecim = self.model_outputs.min_decimals_col[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                        elif self.model_outputs.all_trees[outl_col][parent_tree].column_type == Categorical:
                            cond_col = colnames_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                        else:
                            cond_col = colnames_ord[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                    else:
                        if self.model_outputs.all_trees[outl_col][curr_tree].column_type == Numeric:
                            cond_col = colnames_num[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                            if self.model_outputs.all_trees[outl_col][curr_tree].col_num < ncols_true_numeric:
                                coldecim = self.model_outputs.min_decimals_col[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                        elif self.model_outputs.all_trees[outl_col][curr_tree].column_type == Categorical:
                            cond_col = colnames_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                        else:
                            cond_col = colnames_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]

                    out_df.iat[row, 2].append({
                                                "column"     : cond_col,
                                                "comparison" : colcond,
                                                "value_comp" : condval,
                                                "value_this" : colval,
                                                "decimals"   : coldecim
                                            })
                    curr_tree = parent_tree

        return out_df
