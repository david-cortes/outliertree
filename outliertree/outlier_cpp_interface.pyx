import  numpy as np
cimport numpy as np
from libcpp cimport bool as bool_t ###don't confuse it with Python bool
from libcpp.vector cimport vector

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
        vector[char] split_subset
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
        vector[char] subset_common
        double    perc_in_subset
        double    perc_next_most_comm

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
        vector[char] split_subset
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
        vector[size_t] start_ix_cat_counts
        vector[long double] prop_categ
        vector[ColTransf] col_transf
        vector[double] transf_offset
        vector[double] sd_div
        size_t ncols_numeric
        size_t ncols_categ
        size_t ncols_ord
        vector[double] min_outlier_any_cl
        vector[double] max_outlier_any_cl
        vector[vector[bool_t]] cat_outlier_any_cl
        size_t max_depth

    bool_t fit_outliers_models(ModelOutputs &model_outputs,
                               double *numeric_data,     size_t ncols_numeric,
                               int    *categorical_data, size_t ncols_categ,   int *ncat,
                               int    *ordinal_data,     size_t ncols_ord,     int *ncat_ord,
                               size_t nrows, char *cols_ignore, int nthreads,
                               bool_t categ_as_bin, bool_t ord_as_bin, bool_t cat_bruteforce_subset,
                               size_t max_conditions, double max_perc_outliers, size_t min_size_numeric, size_t min_size_categ,
                               double min_gain, bool_t follow_all, double z_norm, double z_outlier)

    bool_t find_new_outliers(double *numeric_data,
                             int    *categorical_data,
                             int    *ordinal_data,
                             size_t nrows, int nthreads, ModelOutputs &model_outputs)

    void forget_row_outputs(ModelOutputs &model_outputs)


cdef class OutlierCppObject:
    cdef ModelOutputs model_outputs

    def __init__(self):
        self.model_outputs = ModelOutputs()

    def fit_model(self,
                  np.ndarray[double, ndim = 2] arr_num, np.ndarray[int, ndim = 2] arr_cat, np.ndarray[int, ndim = 2] arr_ord,
                  np.ndarray[int, ndim = 1] ncat, np.ndarray[int, ndim = 1] ncat_ord, np.ndarray[char, ndim = 1] cols_ignore,
                  size_t ncols_true_numeric = 0, size_t ncols_true_ts = 0, size_t ncols_true_cat = 0, size_t ncols_true_bool = 0,
                  colnames_num = [], colnames_cat = [], colnames_ord = [], levs_cat = [], levs_ord = [], ts_min = None,
                  bool_t return_outliers = 1, int nthreads = 1, bool_t categ_as_bin = 1, bool_t ord_as_bin = 1, bool_t cat_bruteforce_subset = 0,
                  size_t max_conditions = 4, double max_perc_outliers = 0.01, size_t min_size_numeric = 35, size_t min_size_categ = 75,
                  double min_gain = 0.01, bool_t follow_all = 0, double z_norm = 2.67, double z_outlier = 8.0,
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

        if ncol_num > 0:
            ptr_arr_num = &arr_num[0, 0]
        if ncol_cat > 0:
            ptr_arr_cat = &arr_cat[0, 0]
            ptr_arr_ncat = &ncat[0]
        if ncol_ord > 0:
            ptr_arr_ord = &arr_ord[0, 0]
            ptr_arr_ncat_ord = &ncat_ord[0]
        if cols_ignore.shape[0] > 0:
            ptr_cols_ignore = &cols_ignore[0]

        cdef bool_t found_outliers
        found_outliers = fit_outliers_models(
                                                self.model_outputs,
                                                ptr_arr_num, ncol_num,
                                                ptr_arr_cat, ncol_cat, ptr_arr_ncat,
                                                ptr_arr_ord, ncol_ord, ptr_arr_ncat_ord,
                                                nrows, ptr_cols_ignore, nthreads,
                                                categ_as_bin, ord_as_bin, cat_bruteforce_subset,
                                                max_conditions, max_perc_outliers, min_size_numeric, min_size_categ,
                                                min_gain, follow_all, z_norm, z_outlier
                                            )

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
                        out_df.iat[row, 0] = {"column" : colnames_num[outl_col], "value" : arr_num[row, outl_col]}
                    else:
                        out_df.iat[row, 0] = {
                                                "column" : colnames_num[outl_col],
                                                "value" : np.datetime64((arr_num[row, outl_col] - 1 + ts_min[outl_col - ncols_true_numeric]) * 10**6, "ns")
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
                                            "value" : levs_ord[outl_col - ncol_num - ncol_cat][arr_cat[row, outl_col - ncol_num - ncol_cat]]
                                        }

                ### info about the normal observations in tree branch
                if outl_col < ncol_num:

                    if arr_num[row, outl_col] > self.model_outputs.all_clusters[outl_col][outl_clust].upper_lim:
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
                                                    "upper_thr" : compar_thr,
                                                    "pct_below" : compar_pct,
                                                    "mean"  : np.datetime64((self.model_outputs.all_clusters[outl_col][outl_clust].display_mean \
                                                                            - 1 + ts_min[outl_col - ncols_true_numeric]) * 10**6, "ns"),
                                                    "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                                }

                    else:
                        compar_pct = self.model_outputs.all_clusters[outl_col][outl_clust].perc_above
                        compar_thr = self.model_outputs.all_clusters[outl_col][outl_clust].display_lim_low
                        out_df.iat[row, 1] = {
                                                "lower_thr" : compar_thr,
                                                "pct_above" : compar_pct,
                                                "mean"  : self.model_outputs.all_clusters[outl_col][outl_clust].display_mean,
                                                "sd"    : self.model_outputs.all_clusters[outl_col][outl_clust].display_sd,
                                                "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                            }

                elif outl_col < (ncol_num + ncol_cat):
                    if outl_col < (ncol_num + ncols_true_cat):
                        out_df.iat[row, 1] = {
                                                "categs_common" : levs_cat[outl_col - ncol_num][
                                                                        (np.array(self.model_outputs.all_clusters[outl_col][outl_clust].subset_common) == 0).astype("bool")
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
                                                "pct_other"  : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                                "prior_prob" : self.model_outputs.prop_categ[
                                                                        self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                        arr_cat[row, outl_col - ncol_num]
                                                                        ],
                                                "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                            }

                else:

                    out_df.iat[row, 1] = {
                                            "categs_common" : levs_ord[outl_col - ncol_num - ncol_cat][
                                                                    (np.array(self.model_outputs.all_clusters[outl_col][outl_clust].subset_common) == 0).astype("bool")
                                                                    ],
                                            "pct_common" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_in_subset,
                                            "pct_next_most_comm" : self.model_outputs.all_clusters[outl_col][outl_clust].perc_next_most_comm,
                                            "prior_prob" : self.model_outputs.prop_categ[
                                                                    self.model_outputs.start_ix_cat_counts[outl_col - ncol_num] + \
                                                                    arr_ord[row, outl_col - ncol_num - ncol_cat]
                                                                    ],
                                            "n_obs" : self.model_outputs.all_clusters[outl_col][outl_clust].cluster_size
                                        }


                ### info about the conditions - first from the cluster (final branch in the tree, need to follow it back to the root afterwards)
                if self.model_outputs.all_clusters[outl_col][outl_clust].column_type != NoType:
                    if self.model_outputs.all_clusters[outl_col][outl_clust].split_type == IsNa:
                        colcond = "is NA"
                        condval = np.nan
                        colval  = np.nan

                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == LessOrEqual:
                        if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric:
                            colcond = "<="
                            if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_numeric:
                                condval = self.model_outputs.all_clusters[outl_col][outl_clust].split_point
                            else:
                                condval = np.datetime64(
                                                        (
                                                            self.model_outputs.all_clusters[outl_col][outl_clust].split_point \
                                                            - 1 \
                                                            + ts_min[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                        ) * 10**6,
                                                        "ns")
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
                                                        (
                                                            self.model_outputs.all_clusters[outl_col][outl_clust].split_point \
                                                            - 1 \
                                                            + ts_min[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                                                        ) * 10**6,
                                                        "ns")
                        else:
                            colcond = "in"
                            condval = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]\
                                              [(self.model_outputs.all_clusters[outl_col][outl_clust].split_lev + 1):]


                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == InSubset:
                        if outl_col < (ncol_num + ncols_true_cat):
                            levs = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                            colcond = "in"
                            condval = [levs[lv] for lv in range(len(levs)) if self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[lv] > 0]
                        else:
                            ### TODO: this is redundant
                            colcond = "="
                            condval = bool(self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[1])

                    elif self.model_outputs.all_clusters[outl_col][outl_clust].split_type == NotInSubset:
                        if outl_col < (ncol_num + ncols_true_cat):
                            levs = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                            colcond = "in"
                            condval = [levs[lv] for lv in range(len(levs)) if not (self.model_outputs.all_clusters[outl_col][outl_clust].split_subset[lv] > 0)]
                        else:
                            ### TODO: this is redundant
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
                        colcond = "!= "
                        if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Categorical:
                            if self.model_outputs.all_clusters[outl_col][outl_clust].col_num < ncols_true_cat:
                                condval = levs_cat[self.model_outputs.all_clusters[outl_col][outl_clust].col_num][self.model_outputs.all_clusters[outl_col][outl_clust].split_lev]
                            else:
                                condval = not bool(self.model_outputs.all_clusters[outl_col][outl_clust].split_lev)
                        else:
                            condval = levs_ord[self.model_outputs.all_clusters[outl_col][outl_clust].col_num][self.model_outputs.all_clusters[outl_col][outl_clust].split_lev]
                            

                    ### add the column name and actual value for the row
                    if self.model_outputs.all_clusters[outl_col][outl_clust].column_type == Numeric:
                        cond_col = colnames_num[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                        colval   = arr_num[row, self.model_outputs.all_clusters[outl_col][outl_clust].col_num]
                        if self.model_outputs.all_clusters[outl_col][outl_clust].col_num >= ncols_true_numeric:
                            colval = np.datetime64((colval - 1 + ts_min[self.model_outputs.all_clusters[outl_col][outl_clust].col_num]) * 10**6, "ns")
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

                    out_df.iat[row, 2].append({"column" : cond_col, "comparison" : colcond, "value_comp" : condval, "value_this" : colval})

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
                                    condval = np.datetime64((condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num]) * 10**6, "ns")
                                    colval  = np.datetime64((colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num]) * 10**6, "ns")
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Greater:
                                colcond = ">"
                                condval = self.model_outputs.all_trees[outl_col][curr_tree].split_point
                                colval  = arr_num[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                                if self.model_outputs.all_trees[outl_col][curr_tree].col_num >= ncols_true_numeric:
                                    condval = np.datetime64((condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num]) * 10**6, "ns")
                                    colval  = np.datetime64((colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][curr_tree].col_num]) * 10**6, "ns")

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
                                condval = levs_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num][self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                colval  = levs_cat[arr_cat[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]
                                ### Note: booleans and binaries always get translated to "=" and swapped

                        elif self.model_outputs.all_trees[outl_col][curr_tree].column_type == Ordinal:
                            if self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == IsNa:
                                colcond = "is NA"
                                condval = np.nan
                                colval  = np.nan
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == LessOrEqual:
                                colcond = "in"
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num][:(self.model_outputs.all_trees[outl_col][curr_tree].split_lev + 1)]
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Greater:
                                colcond = "in"
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num][(self.model_outputs.all_trees[outl_col][curr_tree].split_lev + 1):]
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == Equal:
                                colcond = "="
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num][self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                colval  = condval
                            elif self.model_outputs.all_trees[outl_col][curr_tree].split_this_branch == NotEqual:
                                colcond = "!="
                                condval = levs_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num][self.model_outputs.all_trees[outl_col][curr_tree].split_lev]
                                colval  = levs_ord[arr_ord[row, self.model_outputs.all_trees[outl_col][curr_tree].col_num]]

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
                                condval = np.datetime64((condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num]) * 10**6, "ns")
                                colval  = np.datetime64((colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num]) * 10**6, "ns")
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
                                condval = np.datetime64((condval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num]) * 10**6, "ns")
                                colval  = np.datetime64((colval - 1 + ts_min[self.model_outputs.all_trees[outl_col][parent_tree].col_num]) * 10**6, "ns")
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



                    ### add column name
                    if self.model_outputs.all_trees[outl_col][parent_tree].all_branches.size() == 0:
                        if self.model_outputs.all_trees[outl_col][parent_tree].column_type == Numeric:
                            cond_col = colnames_num[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                        elif self.model_outputs.all_trees[outl_col][parent_tree].column_type == Categorical:
                            cond_col = colnames_cat[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                        else:
                            cond_col = colnames_ord[self.model_outputs.all_trees[outl_col][parent_tree].col_num]
                    else:
                        if self.model_outputs.all_trees[outl_col][curr_tree].column_type == Numeric:
                            cond_col = colnames_num[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                        elif self.model_outputs.all_trees[outl_col][curr_tree].column_type == Categorical:
                            cond_col = colnames_cat[self.model_outputs.all_trees[outl_col][curr_tree].col_num]
                        else:
                            cond_col = colnames_ord[self.model_outputs.all_trees[outl_col][curr_tree].col_num]

                    out_df.iat[row, 2].append({"column" : cond_col, "comparison" : colcond, "value_comp" : condval, "value_this" : colval})
                    curr_tree = parent_tree

        return out_df
