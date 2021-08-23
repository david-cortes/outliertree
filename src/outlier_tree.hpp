/********************************************************************************************************************
*    Explainable outlier detection
*    
*    Tries to detect outliers by generating decision trees that attempt to predict the values of each column based on
*    each other column, testing in each branch of every tried split (if it meets some minimum criteria) whether there
*    are observations that seem too distant from the others in a 1-D distribution for the column that the split tries
*    to "predict" (will not generate a score for each observation).
*    Splits are based on gain, while outlierness is based on confidence intervals.
*    Similar in spirit to the GritBot software developed by RuleQuest research. Reference article is:
*      Cortes, David. "Explainable outlier detection through decision tree conditioning."
*      arXiv preprint arXiv:2001.00636 (2020).
*    
*    
*    Copyright 2020 David Cortes.
*    
*    Written for C++11 standard and OpenMP 2.0 or later. Code is meant to be wrapped into scripting languages
*    such as R or Python.
*
*    This file is part of OutlierTree.
*
*    OutlierTree is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OutlierTree is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OutlierTree.  If not, see <https://www.gnu.org/licenses/>.
********************************************************************************************************************/

/***********************
    Standard headers
************************/
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <math.h>
#include <cmath>
#include <stddef.h>
#include <limits.h>
#include <limits>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#ifdef _FOR_R
    #include <Rcpp.h>
#endif
#include <signal.h>
typedef void (*sig_t_)(int);


/************************
    Short Functions
*************************/
#define extract_bit(number, bit) (((number) >> (bit)) & 1) /* https://stackoverflow.com/questions/2249731/how-do-i-get-bit-by-bit-data-from-an-integer-value-in-c */
#define pow2(n) ( ((size_t) 1) << (n) ) /* https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int */
#define avg_between(a, b) ((a) + 0.5*((b) - (a)))
#define square(x) ((x) * (x))
#ifndef isinf
    #define isinf std::isinf
#endif
#ifndef isnan
    #define isnan std::isnan
#endif
#define is_na_or_inf(x) (isnan(x) || isinf(x))

/* Aliasing for compiler optimizations */
#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(SUPPORTS_RESTRICT)
    #define restrict __restrict
#else
    #define restrict 
#endif

/* MSVC is stuck with an OpenMP version that's 19 years old at the time of writing and does not support unsigned iterators */
#ifdef _OPENMP
    #if (_OPENMP < 200801) || defined(_WIN32) || defined(_WIN64) /* OpenMP < 3.0 */
        #define size_t_for long long
    #else
        #define size_t_for size_t
    #endif
#else
    #define size_t_for size_t
#endif

#ifndef _OPENMP
    #define omp_get_thread_num() 0
#endif

#define unexpected_error() throw std::runtime_error("Unexpected error. Please open an issue in GitHub.\n")


/****************************************************************
    Data types and structs that are returned from this module
*****************************************************************/
typedef enum ColType {Numeric, Categorical, Ordinal, NoType} ColType;
typedef enum SplitType {
    LessOrEqual, Greater,  /* for numerical and ordinal */
    Equal, NotEqual,       /* will try to simplify to these post-hoc if possible */
    InSubset, NotInSubset, /* for categoricals */
    SingleCateg, SubTrees, /* one branch per category of a categorical column */
    IsNa, Root
} SplitType;
typedef enum ColTransf {NoTransf, Log, Exp} ColTransf; /* transformation to apply to numeric column */

/*    
*    1-d clusters that define homogeneous groups in which observations can be outliers.
*    Note that these are associated to a tree and define one extra condition from what
*    the tree already specifies. The branch they follow is stored in the cluster, unlike
*    for trees in which it's always left and right branch, as these get discarded more often.
*/
typedef struct Cluster {
    ColType   column_type = NoType;
    size_t col_num = 0; /* numer of the column by which its being split, the target column is given by index of the cluster vector */
    SplitType split_type = Root;
    double split_point = HUGE_VAL; /* numerical */
    std::vector<signed char> split_subset = std::vector<signed char>(); /* categorical */
    int split_lev = INT_MAX;    /* ordinal */
    bool has_NA_branch = false; /* this is in order to determine the best outlier cluster when it fits under more than 1 */

    size_t    cluster_size = 0;
    double    lower_lim = HUGE_VAL;                        /* numerical target column */
    double    upper_lim = -HUGE_VAL;                       /* numerical target column */
    double    perc_below = HUGE_VAL;                       /* numerical target column */
    double    perc_above = HUGE_VAL;                       /* numerical target column */
    double    display_lim_low = HUGE_VAL;                  /* numerical target column */
    double    display_lim_high = -HUGE_VAL;                /* numerical target column */
    double    display_mean = -HUGE_VAL;                    /* numerical target column */
    double    display_sd = -HUGE_VAL;                      /* numerical target column */
    std::vector<signed char> subset_common = std::vector<signed char>(); /* categorical or ordinal target column (=0 is common) */
    double    perc_in_subset = HUGE_VAL;                   /* categorical or ordinal target column */
    double    perc_next_most_comm = -HUGE_VAL;             /* categorical or ordinal target column */ /* TODO */
    int       categ_maj = -1;                              /* when using majority-criterion for categorical outliers */

    double cluster_mean;              /* used to calculate outlier scores at prediction time */
    double cluster_sd;                /* used to calculate outlier scores at prediction time */
    std::vector<double> score_categ;  /* used to calculate outlier scores at prediction time */

    /* constructors in order to use C++'s vector emplace */

    /* full data (no conditions) */
    Cluster(ColType column_type, SplitType split_type)
    {
        this->column_type = column_type;
        this->split_type = split_type;
    }

    /* numerical split */
    Cluster(ColType column_type, size_t col_num, SplitType split_type, double split_point, bool has_NA_branch = false)
    {
        this->column_type = column_type;
        this->col_num = col_num;
        this->split_type = split_type;
        this->split_point = split_point;
        this->has_NA_branch = has_NA_branch;
    }

    /* categorical split */
    Cluster(ColType column_type, size_t col_num, SplitType split_type, signed char *split_subset, int ncat, bool has_NA_branch = false)
    {
        this->column_type = column_type;
        this->col_num = col_num;
        this->split_type = split_type;
        if (split_type != IsNa) this->split_subset.assign(split_subset, split_subset + ncat);
        this->split_subset.shrink_to_fit();
        this->has_NA_branch = has_NA_branch;
    }

    /* categorical split with only one level */
    Cluster(size_t col_num, int cat, int ncat, bool has_NA_branch = false)
    {
        this->column_type = Categorical;
        this->col_num = col_num;
        this->has_NA_branch = has_NA_branch;
        this->split_type = Equal;
        this->split_lev = cat;
    }

    /* ordinal split */
    Cluster(ColType column_type, size_t col_num, SplitType split_type, int split_lev, bool has_NA_branch = false)
    {
        this->column_type = column_type;
        this->col_num = col_num;
        this->split_type = split_type;
        this->split_lev = split_lev;
        this->has_NA_branch = has_NA_branch;
    }

    /* this is for serialization with cereal */
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
                this->column_type,
                this->col_num,
                this->split_type,
                this->split_point,
                this->split_subset,
                this->split_lev,
                this->has_NA_branch,
                this->cluster_size,
                this->lower_lim,
                this->upper_lim,
                this->perc_below,
                this->perc_above,
                this->display_lim_low,
                this->display_lim_high,
                this->display_mean,
                this->display_sd,
                this->subset_common,
                this->perc_in_subset,
                this->perc_next_most_comm,
                this->cluster_mean,
                this->cluster_sd,
                this->score_categ
                );
    }

    /* this is for serialization with both cereal and cython auto-pickle */
    Cluster() = default;
    
} Cluster;

/*    
*    Trees that host the aforementioned clusters. These work as follows:
*    - Each tree contains a split column and condition for splitting.
*    - The trees that follow them are specified in tree_left/right/NA.
*    - If the tree is dropped or not used, that branch gets an index of zero.
*    - The child tree will however remember which branch it took.
*    - At prediction time, the output will tell into which cluster and which tree
*      is each row an outlier (if they fall into any).
*    - The exact conditions are reconstructed by following the trees backwards
*      (i.e. first the cluster, then deepest tree, then follow parent tree until root).
*      This way, all the necessary information can be obtained without storing redundant
*      info, and without needing to reconstruct the conditions as the 'predict'
*      function is being called (which makes it easier to wrap into other languages).
*    - At prediction time, as the observation is passed down trees, all the clusters
*      in all those trees have to be tested for (so if a cluster is discarded, it can
*      keep only one branch of its split in the struct).
*    - As a side effect, in ordinal columns, the trees cannot be simplified to 'Equal'.
*    - All of this is ignored when using 'follow_all', in which case the trees work just
*      like the clusters, with an array 'all_branches' which contains all trees that have
*      to be follow from one particular tree.
*/
typedef struct ClusterTree {
    size_t parent = 0;              /* index in a vector */
    SplitType parent_branch = Root; /* this tree follows this branch in the split given by its parent */
    std::vector<size_t> clusters = std::vector<size_t>(); /* these clusters define additional splits */

    SplitType split_this_branch = Root;                        /* when using 'follow_all' */
    std::vector<size_t> all_branches = std::vector<size_t>();  /* when using 'follow_all' */

    ColType   column_type = NoType;
    size_t    col_num = 0;
    double    split_point = HUGE_VAL;
    std::vector<signed char> split_subset = std::vector<signed char>();
    int split_lev = INT_MAX;

    size_t tree_NA = 0;    /* binary splits */
    size_t tree_left = 0;  /* binary splits */
    size_t tree_right = 0; /* binary splits */
    std::vector<size_t> binary_branches = std::vector<size_t>(); /* multiple splits (single category or binarized categories) */

    ClusterTree(size_t parent, SplitType parent_branch)
    {
        this->parent = parent;
        this->parent_branch = parent_branch;
    }

    /* when using 'follow_all' */
    ClusterTree(size_t parent, size_t col_num, double split_point, SplitType split_this_branch)
    {
        this->parent = parent;
        this->col_num = col_num;
        this->column_type = Numeric;
        this->split_this_branch = split_this_branch;
        this->split_point = split_point;
    }

    ClusterTree(size_t parent, size_t col_num, int split_lev, SplitType split_this_branch)
    {
        this->parent = parent;
        this->col_num = col_num;
        this->column_type = Ordinal;
        this->split_this_branch = split_this_branch;
        this->split_lev = split_lev;
    }

    ClusterTree(size_t parent, size_t col_num, SplitType split_this_branch, signed char *split_subset, int ncat)
    {
        this->parent = parent;
        this->col_num = col_num;
        this->column_type = Categorical;
        if (split_this_branch != IsNa) {
            this->split_this_branch = split_this_branch;
            this->split_subset.assign(split_subset, split_subset + ncat);
            this->split_subset.shrink_to_fit();
        } else {
            this->split_this_branch = IsNa;
        }
    }

    ClusterTree(size_t parent, size_t col_num, int cat_chosen)
    {
        this->parent = parent;
        this->col_num = col_num;
        this->column_type = Categorical;
        this->split_this_branch = Equal;
        this->split_lev = cat_chosen;
    }

    /* this is for serialization with cereal */
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
                this->parent,
                this->parent_branch,
                this->clusters,
                this->split_this_branch,
                this->all_branches,
                this->column_type,
                this->col_num,
                this->split_point,
                this->split_subset,
                this->split_lev,
                this->tree_NA,
                this->tree_left,
                this->tree_right,
                this->binary_branches
                );
    }

    /* this is for serialization with both cereal and cython auto-pickle */
    ClusterTree() = default;

} ClusterTree;

/* these are needed for prediction time, and are thus returned from the function that fits the model */
typedef struct ModelOutputs {
    std::vector< std::vector<ClusterTree> > all_trees;  /* clusters in which observations can be outliers, required for prediction time */
    std::vector< std::vector<Cluster> > all_clusters;   /* decision trees that host the clusters, required for prediction time */
    std::vector<double> outlier_scores_final;   /* if an outlier is flagged, this indicates its score (lower is more outlier) as an upper probability bound */
    std::vector<size_t> outlier_clusters_final; /* if an outlier is flagged, this indicates the most suitable cluster in which to flag it as outlier */
    std::vector<size_t> outlier_columns_final;  /* if an outlier is flagged, this indicates the column that makes it an outlier */
    std::vector<size_t> outlier_trees_final;    /* if an outlier is flagged, this indicates the tree under which the cluster is found */
    std::vector<size_t> outlier_depth_final;    /* if an outlier is flagged, this indicates the split depth under which the cluster is found */
    std::vector<int> outlier_decimals_distr;    /* if an outlier is flagged, and it's a numeric column, this will indicate how many decimals to print for it */
    std::vector<size_t> start_ix_cat_counts; /* this is to determine where to index the proportions */
    std::vector<long double> prop_categ;     /* this is just for statistics to show, it's not used for anything */
    std::vector<ColTransf> col_transf;       /* tells whether each numerical columns underwent log/exp transformations */
    std::vector<double> transf_offset;       /* value subtracted for log transform, mean subtracted for exp transform */
    std::vector<double> sd_div;              /* standard deviation with which exp-transformed columns were standardized */
    std::vector<int> min_decimals_col;       /* number of decimals to show for split conditions in numeric columns */
    std::vector<int> ncat;      /* copied from the inputs, used to determine at prediction time if a category is out-of-range and skip */
    std::vector<int> ncat_ord;  /* copied from the inputs, used to determine at prediction time if a category is out-of-range and skip */
    size_t ncols_numeric;       /* copied from the inputs, used to determine at prediction time if a category is out-of-range and skip */
    size_t ncols_categ;         /* copied from the inputs, used to determine at prediction time if a category is out-of-range and skip */
    size_t ncols_ord;           /* copied from the inputs, used to determine at prediction time if a category is out-of-range and skip */
    std::vector<double> min_outlier_any_cl;             /* redundant info which speeds up prediction */
    std::vector<double> max_outlier_any_cl;             /* redundant info which speeds up prediction */
    std::vector<std::vector<bool>> cat_outlier_any_cl;  /* redundant info which speeds up prediction */
    size_t max_depth;                                   /* redundant info which speeds up prediction */


    /* this is for serialization with cereal */
    template<class Archive>
    void serialize(Archive &archive)
    {
        archive(
                this->all_trees,
                this->all_clusters,
                this->outlier_scores_final,
                this->outlier_clusters_final,
                this->outlier_columns_final,
                this->outlier_trees_final,
                this->outlier_depth_final,
                this->start_ix_cat_counts,
                this->prop_categ,
                this->col_transf,
                this->transf_offset,
                this->sd_div,
                this->ncat,
                this->ncat_ord,
                this->ncols_numeric,
                this->ncols_categ,
                this->ncols_ord,
                this->min_outlier_any_cl,
                this->max_outlier_any_cl,
                this->cat_outlier_any_cl,
                this->max_depth
                );
    }

    /* this is for serialization with both cereal and cython auto-pickle */
    ModelOutputs() = default;

} ModelOutputs;

/*    
*    Note: the vectors with proportions in these structs are supposed to be all small numbers so 'long double' is an overkill for them
*    and does not make them translate into SIMD instructions in regular x86-64 CPUs, but if setting them as 'double' and then doing casts
*    from/between 'double' and the 'size_t' and 'long double's of other arrays (such as in function 'find_outlier_categories'), comparisons
*    such as '<=' will oftentimes fail even with small counts - this is an example that will fail when mixing the 3 types together:
*    >>>    (2 / (88+1)) * 0.5 <= (1 / 89) --> produces FALSE (right answer is TRUE)
*    All due to decimals (in that example) right of the 10th digit, and ends up creating categorical clusters that it should not create.
*    So don't change them back to regular 'double', or if necessary, change every 'long double' to 'double' too.
*/

/******************************************
    Prototypes from fit_model.cpp
    (This is the main module from which
     the model is generated)
*******************************************/
bool fit_outliers_models(ModelOutputs &model_outputs,
                         double *restrict numeric_data,     size_t ncols_numeric,
                         int    *restrict categorical_data, size_t ncols_categ,   int *restrict ncat,
                         int    *restrict ordinal_data,     size_t ncols_ord,     int *restrict ncat_ord,
                         size_t nrows, char *restrict cols_ignore = NULL, int nthreads = 1,
                         bool categ_as_bin = true, bool ord_as_bin = true, bool cat_bruteforce_subset = false, bool categ_from_maj = false, bool take_mid = true,
                         size_t max_depth = 3, double max_perc_outliers = 0.01, size_t min_size_numeric = 25, size_t min_size_categ = 50,
                         double min_gain = 1e-2, bool gain_as_pct = false, bool follow_all = false, double z_norm = 2.67, double z_outlier = 8.0);

typedef struct {
    
    std::vector<size_t> ix_arr;           /* indices from the target column */
    size_t st;                            /* chunk of the indices to take for current function calls */
    size_t end;                           /* chunk of the indices to take for current function calls */
    std::vector<double> outlier_scores;   /* these hold the model outputs for 1 column before combining them */
    std::vector<size_t> outlier_clusters; /* these hold the model outputs for 1 column before combining them */
    std::vector<size_t> outlier_trees;    /* these hold the model outputs for 1 column before combining them */
    std::vector<size_t> outlier_depth;    /* these hold the model outputs for 1 column before combining them */
    size_t target_col_num;                /* if categorical or ordinal, gets subtracted the number of numeric columns (used to index other arrays) */
    long double sd_y;                     /* numerical only (standard deviation before splitting) */
    double mean_y;                        /* numerical only (used to standardize numbers for extra FP precision) */
    long double base_info;                /* categorical and ordinal (information before splitting and before binarizing) */
    long double base_info_orig;           /* categorical and ordinal (information before splitting and after binarizing if needed) */
    bool log_transf;                      /* numerical - whether the target variable underwent a logarithmic transformation */
    bool exp_transf;                      /* numerical - whether the target variable underwent exponentiation on its Z values */
    double *target_numeric_col;           /* dynamic pointer */
    int    *target_categ_col;             /* dynamic pointer */
    std::vector<double> buffer_transf_y;  /* if applying logarithm or exponentiation, transformed values are stored here */
    std::vector<int> buffer_bin_y;        /* if binarizing, transformed values are stored here */
    std::vector<Cluster>     *clusters;   /* dynamic pointer, don't change to reference as it otherwise cannot be reassigned */
    std::vector<ClusterTree> *tree;       /* dynamic pointer, don't change to reference as it otherwise cannot be reassigned */
    bool   has_outliers;                  /* temporary variable from which the other two are updated */
    bool   lev_has_outliers;              /* whether the particular depth level has outliers (if so, wil remove them at the end before new split) */
    bool   col_has_outliers;              /* whether there's any outliers in the column (will later merge them into the outputs) */
    double left_tail;                     /* approximate value where a long left tail ends */
    double right_tail;                    /* approximate value where a long right tail ends */

    bool col_is_bin;                      /* whether the target categorical/ordinal column has 2 categories or has been forcibly binarized */
    long double *prop_small_this;         /* dynamic pointer */
    long double *prior_prob;              /* dynamic pointer */

    double orig_mean;                     /* value to reconstruct originals from exponentiated */
    double orig_sd;                       /* value to reconstruct originals from exponentiated */
    double log_minval;                    /* value to reconstruct originals from logarithms */
    double *orig_target_col;              /* column as it was before applying log/exp (dynamic pointer) */
    int *untransf_target_col;             /* column as it was before forcibly binarizing (dynamic pointer) */
    int *temp_ptr_x;                      /* dynamic pointer */

    std::vector<signed char> buffer_subset_categ_best;  /* categorical split that gave the best gain */
    long double this_gain;                       /* buffer where to store gain */
    double this_split_point;                     /* numeric split threshold */
    int this_split_lev;                          /* ordinal split threshold */
    size_t this_split_ix;                        /* index at which the data is partitioned */
    size_t this_split_NA;                        /* index at which the non-NA values start */
    long double best_gain;                       /* buffer where to store the info of the splitting column that produced the highest gain */
    ColType column_type_best;                    /* buffer where to store the info of the splitting column that produced the highest gain */
    double split_point_best;                     /* buffer where to store the info of the splitting column that produced the highest gain */
    int split_lev_best;                          /* buffer where to store the info of the splitting column that produced the highest gain */
    size_t col_best;                             /* buffer where to store the info of the splitting column that produced the highest gain */

    std::vector<long double> buffer_cat_sum;         /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<long double> buffer_cat_sum_sq;      /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<size_t>      buffer_crosstab;        /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<size_t>      buffer_cat_cnt;         /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<size_t>      buffer_cat_sorted;      /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<signed char>        buffer_subset_categ;    /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<signed char>        buffer_subset_outlier;  /* buffer arrays where to allocate values required by functions and not used outside them */
    std::vector<long double> buffer_sd;              /* used for a more numerically-stable two-pass gain calculation */
    
    bool drop_cluster;          /* for categorical and ordinal variables, not all clusters can flag observations as outliers, so those are not kept */
    bool already_split_main;    /* when binarizing categoricals/ordinals, avoid attempting the same split with numerical and ordinals that take the non-binarized data */
    bool target_col_is_ord;     /* whether the target column is ordinal (rest is the same as in categoricals) */
    int  ncat_this;             /* number of categories in the target column */

} Workspace;

/* info holders to shorten function call arguments */
typedef struct {
    bool    categ_as_bin;
    bool    ord_as_bin;
    bool    cat_bruteforce_subset;
    bool    categ_from_maj;
    bool    take_mid;
    size_t  max_depth;
    double  max_perc_outliers;
    size_t  min_size_numeric;
    size_t  min_size_categ;
    double  min_gain;
    bool    gain_as_pct;
    bool    follow_all;
    double  z_norm;
    double  z_outlier;
    double  z_tail;
    std::vector<long double> prop_small; /* this is not a parameter, but a shared array determined from the parameters and data */
} ModelParams;

/* Note: the vectors here are filled within the function that fits the model, while the pointers are passed from outside */
typedef struct {
    double  *restrict numeric_data;     size_t ncols_numeric;
    int     *restrict categorical_data; size_t ncols_categ;   int *restrict ncat;
    int     *restrict ordinal_data;     size_t ncols_ord;     int *restrict ncat_ord;
    size_t  nrows; size_t tot_cols; std::vector<char> has_NA; std::vector<char> skip_col; int max_categ;
    std::vector<size_t> cat_counts;
} InputData;


void process_numeric_col(std::vector<Cluster> &cluster_root,
                         std::vector<ClusterTree> &tree_root,
                         size_t target_col_num,
                         Workspace &workspace,
                         InputData &input_data,
                         ModelParams &model_params,
                         ModelOutputs &model_outputs);
void recursive_split_numeric(Workspace &workspace,
                             InputData &input_data,
                             ModelParams &model_params,
                             size_t curr_depth, bool is_NA_branch);
void process_categ_col(std::vector<Cluster> &cluster_root,
                       std::vector<ClusterTree> &tree_root,
                       size_t target_col_num, bool is_ord,
                       Workspace &workspace,
                       InputData &input_data,
                       ModelParams &model_params,
                       ModelOutputs &model_outputs);
void recursive_split_categ(Workspace &workspace,
                           InputData &input_data,
                           ModelParams &model_params,
                           size_t curr_depth, bool is_NA_branch);


/*******************************************
    Prototypes from predict.cpp
    (This is the module from which
     new data can be flagged as outliers)
********************************************/
typedef struct {
    double  *restrict numeric_data;
    int     *restrict categorical_data;
    int     *restrict ordinal_data;
    size_t nrows;
} PredictionData;

bool find_new_outliers(double *restrict numeric_data,
                       int    *restrict categorical_data,
                       int    *restrict ordinal_data,
                       size_t nrows, int nthreads, ModelOutputs &model_outputs);
bool follow_tree(ModelOutputs &model_outputs, PredictionData &prediction_data, size_t curr_tree, size_t curr_depth,
                 size_t_for row, size_t_for col, bool col_is_num, double num_val_this, int cat_val_this);
bool check_is_outlier_in_tree(std::vector<size_t> &clusters_in_tree, size_t curr_depth, size_t curr_tree,
                              ModelOutputs &model_outputs, PredictionData &prediction_data, size_t_for row, size_t_for col,
                              bool col_is_num, double num_val_this, int cat_val_this);


/********************************
    Prototypes from split.cpp
*********************************/
#define SD_REG 1e-5 /* Regularization for standard deviation estimation */

typedef struct {
    size_t      cnt;
    long double sum;
    long double sum_sq;
} NumericBranch;

typedef struct {
    NumericBranch NA_branch    = {0, 0, 0};
    NumericBranch left_branch  = {0, 0, 0};
    NumericBranch right_branch = {0, 0, 0};
} NumericSplit;

typedef struct {
    size_t *restrict NA_branch;     /* array of counts of the target variable's categories */
    size_t *restrict left_branch;   /* array of counts of the target variable's categories */
    size_t *restrict right_branch;  /* array of counts of the target variable's categories */
    size_t ncat;                    /* number of categories/entries in the arrays above */
    size_t tot;                        /* size_NA +  size_left + size_right */
    size_t size_NA    = 0;
    size_t size_left  = 0;
    size_t size_right = 0;
} CategSplit;

void subset_to_onehot(size_t ix_arr[], size_t n_true, size_t n_tot, signed char onehot[]);
size_t move_zero_count_to_front(size_t *restrict cat_sorted, size_t *restrict cat_cnt, size_t ncat_x);
void flag_zero_counts(signed char split_subset[], size_t buffer_cat_cnt[], size_t ncat_x);
long double calc_sd(size_t cnt, long double sum, long double sum_sq);
long double calc_sd(NumericBranch &branch);
long double calc_sd(size_t ix_arr[], double *restrict x, size_t st, size_t end, double *restrict mean);
long double numeric_gain(NumericSplit &split_info, long double tot_sd);
long double numeric_gain(long double tot_sd, long double info_left, long double info_right, long double info_NA, long double cnt);
long double total_info(size_t categ_counts[], size_t ncat);
long double total_info(size_t categ_counts[], size_t ncat, size_t tot);
long double total_info(size_t *restrict ix_arr, int *restrict x, size_t st, size_t end, size_t ncat, size_t *restrict buffer_cat_cnt);
long double categ_gain(CategSplit split_info, long double base_info);
long double categ_gain(size_t *restrict categ_counts, size_t ncat, size_t *restrict ncat_col, size_t maxcat, long double base_info, size_t tot);
long double categ_gain_from_split(size_t *restrict ix_arr, int *restrict x, size_t st, size_t st_non_na, size_t split_ix, size_t end,
                                  size_t ncat, size_t *restrict buffer_cat_cnt, long double base_info);
void split_numericx_numericy(size_t *restrict ix_arr, size_t st, size_t end, double *restrict x, double *restrict y,
                             long double sd_y, bool has_na, size_t min_size, bool take_mid, long double *restrict buffer_sd,
                             long double *restrict gain, double *restrict split_point, size_t *restrict split_left, size_t *restrict split_NA);
void split_categx_numericy(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, double *restrict y, long double sd_y, double ymean,
                           bool x_is_ordinal, size_t ncat_x, size_t *restrict buffer_cat_cnt, long double *restrict buffer_cat_sum,
                           long double *restrict buffer_cat_sum_sq, size_t *restrict buffer_cat_sorted,
                           bool has_na, size_t min_size, long double *gain, signed char *restrict split_subset, int *restrict split_point);
void split_numericx_categy(size_t *restrict ix_arr, size_t st, size_t end, double *restrict x, int *restrict y,
                           size_t ncat_y, long double base_info, size_t *restrict buffer_cat_cnt,
                           bool has_na, size_t min_size, bool take_mid, long double *restrict gain, double *restrict split_point,
                           size_t *restrict split_left, size_t *restrict split_NA);
void split_ordx_categy(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                       size_t ncat_y, size_t ncat_x, long double base_info,
                       size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab, size_t *restrict buffer_ord_cnt,
                       bool has_na, size_t min_size, long double *gain, int *split_point);
void split_categx_biny(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                       size_t ncat_x, long double base_info,
                       size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab, size_t *restrict buffer_cat_sorted,
                       bool has_na, size_t min_size, long double *gain, signed char *restrict split_subset);
void split_categx_categy_separate(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                                  size_t ncat_x, size_t ncat_y, long double base_info,
                                  size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab,
                                  bool has_na, size_t min_size, long double *gain);
void split_categx_categy_subset(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                                size_t ncat_x, size_t ncat_y, long double base_info,
                                size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab, size_t *restrict buffer_split,
                                bool has_na, size_t min_size, long double *gain, signed char *restrict split_subset);



/***********************************
    Prototypes from clusters.cpp
************************************/
#define calculate_max_outliers(n, perc) (  (n) * (perc) + (long double)2 * sqrtl( (n) * (perc) * ((long double)1 - perc) ) + (long double)1  )
#define z_score(x, mu, sd) (  ((x) - (mu)) / std::max((sd), 1e-12)  )
#define chebyshyov_bound(zval) (1.0 / std::max(square(zval), 1.))

bool define_numerical_cluster(double *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                              double *restrict outlier_scores, size_t *restrict outlier_clusters, size_t *restrict outlier_trees,
                              size_t *restrict outlier_depth, Cluster &cluster, std::vector<Cluster> &clusters, size_t cluster_num, size_t tree_num, size_t tree_depth,
                              bool is_log_transf, double log_minval, bool is_exp_transf, double orig_mean, double orig_sd,
                              double left_tail, double right_tail, double *restrict orig_x,
                              double max_perc_outliers, double z_norm, double z_outlier);
void define_categ_cluster_no_cond(int *restrict x, size_t *restrict ix_arr, size_t st, size_t end, size_t ncateg,
                                  double *restrict outlier_scores, size_t *restrict outlier_clusters, size_t *restrict outlier_trees,
                                  size_t *restrict outlier_depth, Cluster &cluster,
                                  size_t *restrict categ_counts, signed char *restrict is_outlier, double perc_next_most_comm);
bool define_categ_cluster(int *restrict x, size_t *restrict ix_arr, size_t st, size_t end, size_t ncateg, bool by_maj,
                          double *restrict outlier_scores, size_t *restrict outlier_clusters, size_t *restrict outlier_trees,
                          size_t *restrict outlier_depth, Cluster &cluster, std::vector<Cluster> &clusters,
                          size_t cluster_num, size_t tree_num, size_t tree_depth,
                          double max_perc_outliers, double z_norm, double z_outlier,
                          long double *restrict perc_threshold, long double *restrict prop_prior,
                          size_t *restrict buffer_categ_counts, long double *restrict buffer_categ_pct,
                          size_t *restrict buffer_categ_ix, signed char *restrict buffer_outliers,
                          bool *restrict drop_cluster);
void simplify_when_equal_cond(std::vector<Cluster> &clusters, int ncat_ord[]);
void simplify_when_equal_cond(std::vector<ClusterTree> &trees, int ncat_ord[]);
#ifdef TEST_MODE_DEFINE
void prune_unused_trees(std::vector<ClusterTree> &trees);
#endif
bool check_tree_is_not_needed(ClusterTree &tree);
void calculate_cluster_minimums(ModelOutputs &model_outputs, size_t col);
void calculate_cluster_poss_categs(ModelOutputs &model_outputs, size_t col, size_t col_rel);


/**************************************
    Prototypes from cat_outlier.cpp
***************************************/
#define calculate_max_cat_outliers(n, perc, z_norm) ((long double)1 + ((n) * (perc) / z_norm)) /* Note: this is not anyhow probabilistic, nor based on provable bounds */
void find_outlier_categories(size_t categ_counts[], size_t ncateg, size_t tot, double max_perc_outliers,
                             long double perc_threshold[], size_t buffer_ix[], long double buffer_perc[],
                             double z_norm, signed char is_outlier[], bool *found_outliers, bool *new_is_outlier, double *next_most_comm);
void find_outlier_categories_by_maj(size_t categ_counts[], size_t ncateg, size_t tot, double max_perc_outliers,
                                    long double prior_prob[], double z_outlier, signed char is_outlier[],
                                    bool *found_outliers, bool *new_is_outlier, int *categ_maj);
bool find_outlier_categories_no_cond(size_t categ_counts[], size_t ncateg, size_t tot,
                                     signed char is_outlier[], double *next_most_comm);



/*************************************************
    Prototypes from misc.cpp and other structs
**************************************************/

/* an inefficient workaround for coding up option 'follow_all' */
typedef struct {
    double gain_restore;
    double gain_best_restore;
    double split_point_restore;
    int    split_lev_restore;
    std::vector<signed char> split_subset_restore;
    size_t ix1_restore;
    size_t ix2_restore;
    size_t ix3_restore;
    size_t ix4_restore;
    int *  temp_ptr_x;
    size_t col_best_restore;
    ColType col_type_best_rememer;
    double split_point_best_restore;
    int    split_lev_best_restore;
    std::vector<signed char> split_subset_best_restore;
    long double base_info_restore;
    long double base_info_orig_restore;
    double sd_y_restore;
    bool has_outliers_restore;
    bool lev_has_outliers_restore;
} RecursionState;


int calculate_category_indices(size_t start_ix_cat_counts[], int ncat[], size_t ncols, bool skip_col[], int max_categ = 0);
void calculate_all_cat_counts(size_t start_ix_cat_counts[], size_t cat_counts[], int ncat[],
                              int categorical_data[], size_t ncols, size_t nrows,
                              bool has_NA[], bool skip_col[], int nthreads);
void check_cat_col_unsplittable(size_t start_ix_cat_counts[], size_t cat_counts[], int ncat[],
                                size_t ncols, size_t min_conditioned_size, size_t nrows, bool skip_col[], int nthreads);
void calculate_lowerlim_proportion(long double *restrict prop_small, long double *restrict prop,
                                   size_t start_ix_cat_counts[], size_t cat_counts[],
                                   size_t ncols, size_t nrows, double z_norm, double z_tail);
void check_missing_no_variance(double numeric_data[], size_t ncols, size_t nrows, bool has_NA[],
                               bool skip_col[], int min_decimals[], int nthreads);
void calc_central_mean_and_sd(size_t ix_arr[], size_t st, size_t end, double x[], size_t size_quarter, double *mean_central, double *sd_central);
void check_for_tails(size_t ix_arr[], size_t st, size_t end, double *restrict x,
                     double z_norm, double max_perc_outliers,
                     double *restrict buffer_x, double mean, double sd,
                     double *restrict left_tail, double *restrict right_tail,
                     bool *exp_transf, bool *log_transf);
size_t move_outliers_to_front(size_t ix_arr[], double outlier_scores[], size_t st, size_t end);
size_t move_NAs_to_front(size_t ix_arr[], double x[], size_t st, size_t end, bool inf_as_NA);
size_t move_NAs_to_front(size_t ix_arr[], int x[], size_t st, size_t end);
void divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point, bool has_NA, size_t *split_NA, size_t *st_right);
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, signed char subset_categ[], int ncat, bool has_NA, size_t *split_NA, size_t *st_right);
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, int split_lev, bool has_NA, size_t *split_NA, size_t *st_right);
bool check_workspace_is_allocated(Workspace &workspace);
void allocate_thread_workspace(Workspace &workspace, size_t nrows, int max_categ);
void backup_recursion_state(Workspace &workspace, RecursionState &state_backup);
void restore_recursion_state(Workspace &workspace, RecursionState &state_backup);
void set_tree_as_numeric(ClusterTree &tree, double split_point, size_t col);
void set_tree_as_categorical(ClusterTree &tree, int ncat, signed char *split_subset, size_t col);
void set_tree_as_categorical(ClusterTree &tree, size_t col);
void set_tree_as_categorical(ClusterTree &tree, size_t col, int ncat);
void set_tree_as_ordinal(ClusterTree &tree, int split_lev, size_t col);
void forget_row_outputs(ModelOutputs &model_outputs);
void allocate_row_outputs(ModelOutputs &model_outputs, size_t nrows, size_t max_depth);
void check_more_two_values(double arr_num[], size_t nrows, size_t ncols, int nthreads, char too_few_values[]);
void calc_min_decimals_to_print(ModelOutputs &model_outputs, double *restrict numeric_data, int nthreads);
int decimals_diff(double val1, double val2);
void dealloc_ModelOutputs(ModelOutputs &model_outputs);
ModelOutputs get_empty_ModelOutputs();
bool get_has_openmp();

extern bool interrupt_switch;
extern bool handle_is_locked;
void set_interrup_global_variable(int s);
class SignalSwitcher
{
public:
    sig_t_ old_sig;
    bool is_active;
    SignalSwitcher();
    ~SignalSwitcher();
    void restore_handle();
};
void check_interrupt_switch(SignalSwitcher &ss);
#ifdef _FOR_PYTHON
bool cy_check_interrupt_switch();
void cy_tick_off_interrupt_switch();
#endif
