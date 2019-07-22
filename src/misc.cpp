/********************************************************************************************************************
*    Explainable outlier detection
*    
*    Tries to detect outliers by generating decision trees that attempt to predict the values of each column based on
*    each other column, testing in each branch of every tried split (if it meets some minimum criteria) whether there
*    are observations that seem too distant from the others in a 1-D distribution for the column that the split tries
*    to "predict" (will not generate a score for each observation).
*    Splits are based on gain, while outlierness is based on confidence intervals.
*    Similar in spirit to the GritBot software developed by RuleQuest research.
*    
*    
*    Copyright 2019 David Cortes.
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
#include "outlier_tree.hpp"


/*    
*    Calculate, in a continuous array that would hold start indices for each category for each column in sequence,
*    at which position will the counts for a given column start. Note that NAs are stored as the last index in each
*    column, so each one needs one extra category
*/
int calculate_category_indices(size_t start_ix_cat_counts[], int ncat[], size_t ncols, bool skip_col[], int max_categ)
{
    for (size_t col = 0; col < ncols; col++) {
        max_categ = std::max(ncat[col], max_categ);
        start_ix_cat_counts[col + 1] = start_ix_cat_counts[col] + ncat[col] + 1;
        if (ncat[col] < 2) skip_col[col] = true;
    }

    return max_categ;
}

/* Save the counts of each category for each column in the array determined above */
void calculate_all_cat_counts(size_t start_ix_cat_counts[], size_t cat_counts[], int ncat[],
                              int categorical_data[], size_t ncols, size_t nrows,
                              bool has_NA[], bool skip_col[], int nthreads)
{
    size_t col_st_offset;
    size_t col_stop;

    #pragma omp parallel for schedule(static, 1) num_threads(nthreads) private(col_st_offset, col_stop)
    for (size_t col = 0; col < ncols; col++) {

        if (skip_col[col]) continue;

        col_st_offset = start_ix_cat_counts[col];
        col_stop = (col + 1) * nrows;
        for (size_t row = (col * nrows); row < col_stop; row++) {

            if (categorical_data[row] >= 0) {
                cat_counts[ categorical_data[row] + col_st_offset ]++;
            } else {
                cat_counts[ ncat[col] + col_st_offset ]++;
                has_NA[col] = true;
            }

        }
    }
}

/* Check if some column has a large majority that would make any split fail to meet minimum sizes */
void check_cat_col_unsplittable(size_t start_ix_cat_counts[], size_t cat_counts[], int ncat[],
                                size_t ncols, size_t min_conditioned_size, size_t nrows, bool skip_col[], int nthreads)
{
    size_t largest_cnt;
    #pragma omp parallel for num_threads(nthreads) private(largest_cnt) shared(ncols, nrows, ncat, cat_counts, start_ix_cat_counts, min_conditioned_size, skip_col)
    for (size_t col = 0; col < ncols; col++) {

        largest_cnt = 0;
        for (int cat = 0; cat <= ncat[col]; cat++) {
            largest_cnt = std::max(largest_cnt, cat_counts[ cat + start_ix_cat_counts[col] ]);
        }
        if (largest_cnt > (nrows - min_conditioned_size)) skip_col[col] = true;
        if (largest_cnt <= 1) skip_col[col] = true;

    }
}

/* Calculate the maxmimum proportions in a subset below which a category *can* be considered as outlier (must meet other conditions too) */
void calculate_lowerlim_proportion(long double *restrict prop_small, long double *restrict prop,
                                   size_t start_ix_cat_counts[], size_t cat_counts[],
                                   size_t ncols, size_t nrows, double z_norm, double z_tail)
{
    /* TODO: come up with some way of flagging unseen categories as outliers */
    long double mean;
    long double sd;
    long double nrows_dbl = (long double) nrows;
    for (size_t col = 0; col < ncols; col++) {

        for (size_t cat = start_ix_cat_counts[col]; cat < (start_ix_cat_counts[col + 1] - 1); cat++) {
            
            if (cat_counts[cat] > 0) {
                mean = (long double) cat_counts[cat] / nrows_dbl;
                sd = sqrtl( mean * (1.0 - mean)  / nrows_dbl );
                prop_small[cat] = fminl(mean - z_norm * sd, mean * 0.5);
                prop[cat] = mean;
            } else {
                prop_small[cat] = 0;
                prop[cat] = 0;
            }

        }

    }
}

/* Check if a numerical column has no variance (i.e. will not be splittable) */
void check_missing_no_variance(double numeric_data[], size_t ncols, size_t nrows, bool has_NA[], bool skip_col[], int nthreads)
{
    long double sum;
    long double sum_sq;
    size_t cnt;
    size_t col_stop;

    #pragma omp parallel for schedule(static) num_threads(nthreads) private(sum, sum_sq, cnt, col_stop)
    for (size_t col = 0; col < ncols; col++) {
        sum = 0;
        sum_sq = 0;
        cnt = 0;
        col_stop = (col + 1) * nrows;
        for (size_t row = col * nrows; row < col_stop; row++) {
            if (!is_na_or_inf(numeric_data[row])) {
                sum    += numeric_data[row];
                sum_sq += square(numeric_data[row]);
                cnt++;
            } else {
                has_NA[col] = true;
            }
        }
        if (fabs(sum_sq - square(sum) / (long double)cnt) < 1e-1) skip_col[col] = true;
    }
}

/* Calculate mean and standard deviation from the central half of the data, and adjust SD heuristically by x2.5 */
void calc_central_mean_and_sd(size_t ix_arr[], size_t st, size_t end, double x[], size_t size_quarter, double *mean_central, double *sd_central)
{
    long double sum = 0;
    long double sum_sq = 0;
    if (ix_arr != NULL) {
        for (size_t row = (st + size_quarter); row <= (end - size_quarter); row++) {
            sum    += x[ix_arr[row]];
            sum_sq += square(x[ix_arr[row]]);
        }
    } else {
        for (size_t row = (st + size_quarter); row <= (end - size_quarter); row++) {
            sum    += x[row];
            sum_sq += square(x[row]);
        }
    }
    *mean_central = sum / (long double)(2 * size_quarter);
    *sd_central = 2.5 * calc_sd(2 * size_quarter, sum, sum_sq);
}

/*    Check whether a numerical column has long tails, and whether a transformation is appropriate
*    
*    Will check if there are too many observations with large Z values at either side. If found, will
*    see if applying a trasnformation (exponentiation for left tail, logarithm for right tail) would
*    solve the problem. If not, will report approximate values for where the tails end/start. If it
*    has tails at both sides, will not process the column.
*    
*    Parameters:
*    - ix_arr[n] (in)
*        Indices by which the 'x' variable would be sorted in ascending order. (Must be already sorted!!!)
*    - st (in)
*        Position at which ix_arr starts (inclusive).
*    - end (in)
*        Position at which ix_arr ends (inclusive).
*    - x[n] (in)
*        Column with the original values to check for tails.
*    - z_norm (in)
*        Model parameter. Default is 2.67.
*    - max_perc_outliers (in)
*        Model parameter. Default is 0.01.
*    - buffer_x[n] (temp)
*        Array where to store the transformed values of 'x' (will ignore ix_arr).
*    - mean (in)
*        Mean to use for transforming to Z scores before exponentiating.
*    - sd (in)
*        Standard deviation to use for transforming to Z scores before exponentiating.
*    - left_tail (out)
*        Approximate value at which the tail is considered to end (if found and not solvable by transforming).
*    - right_tail (out)
*        Approximate value at which the tail is considered to start (if found and not solvable by transforming).
*    - exp_transf (out)
*        Whether to apply an exponential transformation (on the Z values!!!) to solve the problem of having a long left tail.
*    - log_transf (out)
*        Whether to apply a log transform to solve the problem of having a long right tail.
*/
void check_for_tails(size_t ix_arr[], size_t st, size_t end, double *restrict x,
                     double z_norm, double max_perc_outliers,
                     double *restrict buffer_x, double mean, double sd,
                     double *restrict left_tail, double *restrict right_tail,
                     bool *exp_transf, bool *log_transf)
{
    size_t size_quarter = (end - st + 1) / 4;
    size_t tail_ix;
    size_t median = 2 * size_quarter;
    double z_tail = 2 * z_norm;
    double const_add_log;
    *left_tail  = -HUGE_VAL;
    *right_tail =  HUGE_VAL;
    size_t max_norm_tail = (size_t) calculate_max_outliers((long double)(end - st + 1), max_perc_outliers);
    double mean_central, sd_central;
    calc_central_mean_and_sd(ix_arr, st, end, x, size_quarter, &mean_central, &sd_central);
    *exp_transf = false;
    *log_transf = false;
    if ( z_score(x[ix_arr[st + max_norm_tail]], mean_central, sd_central) < (-z_tail) ) *left_tail = 1;
    if ( z_score(x[ix_arr[end - max_norm_tail]], mean_central, sd_central) > z_tail ) *right_tail = 1;

    /* check for left tail (too many low values) */
    if (*left_tail == 1) {

        /* check if exponentiation would help */
        for (size_t row = (st + size_quarter); row <= (end - size_quarter); row++)
            buffer_x[row] = exp(z_score(x[ix_arr[row]], mean, sd));
        calc_central_mean_and_sd(NULL, st, end, buffer_x, size_quarter, &mean_central, &sd_central);
        buffer_x[st + max_norm_tail] = exp(z_score(x[ix_arr[st + max_norm_tail]], mean, sd));
        if (z_score(buffer_x[st + max_norm_tail], mean_central, sd_central) >= -z_tail)
        {
            *left_tail =  HUGE_VAL;
            *exp_transf = true;
        }

        /* if exponentiation doesn't help, determine where does the tail lie on the untransformed data */
        else {

            *exp_transf = false;
            for (tail_ix = st; tail_ix <= median; tail_ix++) {
                if (z_score(x[ix_arr[tail_ix]], mean_central, sd_central) > (-z_tail)) break;
            }
            *left_tail = x[ix_arr[tail_ix]];

        }

    }

    /* check for right tail (too many high values) */
    if (*right_tail == 1 ) {

        if (x[ix_arr[st]] == 0) {
            const_add_log = +1;
        } else {
            const_add_log = - x[ix_arr[st]] + 1e-3;
        }

        /* check if a log transform would help */
        for (size_t row = (st + size_quarter); row <= (end - size_quarter); row++)
            buffer_x[row] = log(x[ix_arr[row]] + const_add_log);
        calc_central_mean_and_sd(NULL, st, end, buffer_x, size_quarter, &mean_central, &sd_central);
        buffer_x[end - max_norm_tail] = log(x[ix_arr[end - max_norm_tail]] + const_add_log);
        if (z_score(buffer_x[end - max_norm_tail], mean_central, sd_central) <= z_tail)
        {
            *right_tail =  HUGE_VAL;
            *log_transf = true;
        }

        /* if log transform doesn't help, determine where does the tail lie on the untransformed data */
        else {
            for (tail_ix = end; tail_ix >= median; tail_ix--) {
                if (z_score(x[ix_arr[tail_ix]], mean_central, sd_central) < z_tail) break;
            }
            *right_tail = x[ix_arr[tail_ix]];
        }

    }

}

/*    Move identified outliers for a given column to the beginning of the indices array,
    and return the position at which the non-outliers start */
size_t move_outliers_to_front(size_t ix_arr[], double outlier_scores[], size_t st, size_t end)
{
    size_t st_non_na = st;
    size_t temp;

    for (size_t i = st; i <= end; i++) {
        if (outlier_scores[ix_arr[i]] < 1.0) {
            temp = ix_arr[st_non_na];
            ix_arr[st_non_na] = ix_arr[i];
            ix_arr[i] = temp;
            st_non_na++;
        }
    }
    return st_non_na;
}

/* Move missing values of a numeric variable to the front of the indices array and return the position at which non-missing ones start */
size_t move_NAs_to_front(size_t ix_arr[], double x[], size_t st, size_t end, bool inf_as_NA)
{
    size_t st_non_na = st;
    size_t temp;

    if (inf_as_NA) {
        for (size_t i = st; i <= end; i++) {
            if (is_na_or_inf(x[ix_arr[i]])) {
                temp = ix_arr[st_non_na];
                ix_arr[st_non_na] = ix_arr[i];
                ix_arr[i] = temp;
                st_non_na++;
            }
        }
    } else {
        for (size_t i = st; i <= end; i++) {
            if (isnan(x[ix_arr[i]])) {
                temp = ix_arr[st_non_na];
                ix_arr[st_non_na] = ix_arr[i];
                ix_arr[i] = temp;
                st_non_na++;
            }
        }
    }

    return st_non_na;
}

/* Move missing values of a categorical variable to the front of the indices array and return the position at which non-missing ones start */
size_t move_NAs_to_front(size_t ix_arr[], int x[], size_t st, size_t end)
{
    size_t st_non_na = st;
    size_t temp;

    for (size_t i = st; i <= end; i++) {
        if (x[ix_arr[i]] < 0) { /* categorical NAs are represented as negative integers */
            temp = ix_arr[st_non_na];
            ix_arr[st_non_na] = ix_arr[i];
            ix_arr[i] = temp;
            st_non_na++;
        }
    }
    return st_non_na;
}

/* for numerical */
void divide_subset_split(size_t ix_arr[], double x[], size_t st, size_t end, double split_point, bool has_NA, size_t *split_NA, size_t *st_right)
{
    size_t temp;

    if (has_NA) {
        *split_NA = move_NAs_to_front(ix_arr, x, st, end, false);
        st = *split_NA;
    } else { *split_NA = st; }
    for (size_t row = st; row <= end; row++) {

        /* move to the left if the category is there */
        if (x[ix_arr[row]] <= split_point) {
            temp = ix_arr[st];
            ix_arr[st] = ix_arr[row];
            ix_arr[row] = temp;
            st++;
        }
    }

    *st_right = st;
}

/* for categorical */
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, char subset_categ[], int ncat, bool has_NA, size_t *split_NA, size_t *st_right)
{
    size_t temp;

    if (has_NA) {
        *split_NA = move_NAs_to_front(ix_arr, x, st, end);
        st = *split_NA;
    } else { *split_NA = st; }
    for (size_t row = st; row <= end; row++) {

        /* move to the left if the category is there */
        if (subset_categ[ x[ix_arr[row]] ] != 0) {
            temp = ix_arr[st];
            ix_arr[st] = ix_arr[row];
            ix_arr[row] = temp;
            st++;
        }
    }

    *st_right = st;
}

/* for ordinal */
void divide_subset_split(size_t ix_arr[], int x[], size_t st, size_t end, int split_lev, bool has_NA, size_t *split_NA, size_t *st_right)
{
    size_t temp;
    
    if (has_NA) {
        *split_NA = move_NAs_to_front(ix_arr, x, st, end);
        st = *split_NA;
    } else { *split_NA = st; }
    for (size_t row = st; row <= end; row++) {

        /* move to the left if the category is there */
        if (x[ix_arr[row]] <= split_lev) {
            temp = ix_arr[st];
            ix_arr[st] = ix_arr[row];
            ix_arr[row] = temp;
            st++;
        }
    }

    *st_right = st;
}

/* thread-local memory where intermediate outputs and buffers are stored */
bool check_workspace_is_allocated(Workspace &workspace)
{
    return workspace.ix_arr.size() > 0;
}

void allocate_thread_workspace(Workspace &workspace, size_t nrows, int max_categ)
{
    workspace.buffer_transf_y.resize(nrows);
    workspace.buffer_bin_y.resize(nrows);
    workspace.ix_arr.resize(nrows);
    for (size_t i = 0; i < nrows; i++) workspace.ix_arr[i] = i;

    workspace.outlier_scores.resize(nrows);
    workspace.outlier_clusters.resize(nrows);
    workspace.outlier_trees.resize(nrows);
    workspace.outlier_depth.resize(nrows);

    workspace.buffer_cat_sum.resize(max_categ + 1);
    workspace.buffer_cat_sum_sq.resize(max_categ + 1);
    workspace.buffer_cat_cnt.resize( (max_categ + 1) * 3);
    workspace.buffer_cat_sorted.resize(max_categ);
    workspace.buffer_subset_categ.resize(max_categ);

    workspace.buffer_subset_categ_best.resize(max_categ);
    workspace.buffer_crosstab.resize(square(max_categ + 1));
    workspace.buffer_subset_outlier.resize(max_categ);
}

/*    
*    This was a quick way of coding up the option 'follow_all' - it basically backs up the modifyable data that
*    is looked at during a recursion, in a rather un-optimal manner. It can be optimized further by not copying
*    everything, as it doesn't really need to always copy all variables (same for the restore function below).
*    For example, at a given point, only one of buffer_subset_categ/this_split_point is used.
*/
void backup_recursion_state(Workspace &workspace, RecursionState &state_backup)
{
    state_backup.gain_restore = workspace.this_gain;
    state_backup.gain_best_restore = workspace.best_gain;
    state_backup.split_point_restore = workspace.this_split_point;
    state_backup.split_lev_restore = workspace.this_split_lev;
    state_backup.split_subset_restore = workspace.buffer_subset_categ;
    state_backup.ix1_restore = workspace.st;
    state_backup.ix2_restore = workspace.this_split_NA;
    state_backup.ix3_restore = workspace.this_split_ix;
    state_backup.ix4_restore = workspace.end;
    state_backup.col_best_restore = workspace.col_best;
    state_backup.col_type_best_rememer = workspace.column_type_best;
    state_backup.split_point_best_restore = workspace.split_point_best;
    state_backup.split_lev_best_restore = workspace.split_lev_best;
    state_backup.split_subset_best_restore = workspace.buffer_subset_categ_best;
    state_backup.base_info_restore = workspace.base_info;
    state_backup.base_info_orig_restore = workspace.base_info_orig;
    state_backup.sd_y_restore = workspace.sd_y;
    state_backup.has_outliers_restore = workspace.has_outliers;
    state_backup.lev_has_outliers_restore = workspace.lev_has_outliers;
    state_backup.temp_ptr_x = workspace.temp_ptr_x;
}

void restore_recursion_state(Workspace &workspace, RecursionState &state_backup)
{
    workspace.this_gain = state_backup.gain_restore;
    workspace.best_gain = state_backup.gain_best_restore;
    workspace.this_split_point = state_backup.split_point_restore;
    workspace.this_split_lev = state_backup.split_lev_restore;
    workspace.buffer_subset_categ = state_backup.split_subset_restore;
    workspace.st = state_backup.ix1_restore;
    workspace.this_split_NA = state_backup.ix2_restore;
    workspace.this_split_ix = state_backup.ix3_restore;
    workspace.end = state_backup.ix4_restore;
    workspace.col_best = state_backup.col_best_restore;
    workspace.column_type_best = state_backup.col_type_best_rememer;
    workspace.split_point_best = state_backup.split_point_best_restore;
    workspace.split_lev_best = state_backup.split_lev_best_restore;
    workspace.buffer_subset_categ_best = state_backup.split_subset_best_restore;
    workspace.base_info = state_backup.base_info_restore;
    workspace.base_info_orig = state_backup.base_info_orig_restore;
    workspace.sd_y = state_backup.sd_y_restore;
    workspace.has_outliers = state_backup.has_outliers_restore;
    workspace.lev_has_outliers = state_backup.lev_has_outliers_restore;
    workspace.temp_ptr_x = state_backup.temp_ptr_x;
}

/* Next split on the trees is only decided after they are already initialized */
void set_tree_as_numeric(ClusterTree &tree, double split_point, size_t col)
{
    tree.column_type = Numeric;
    tree.split_point = split_point;
    tree.col_num = col;
}

void set_tree_as_categorical(ClusterTree &tree, int ncat, char *split_subset, size_t col)
{
    tree.column_type = Categorical;
    tree.col_num = col;
    tree.split_subset.assign(split_subset, split_subset + ncat);
    tree.split_subset.shrink_to_fit();
}

void set_tree_as_categorical(ClusterTree &tree, size_t col)
{
    tree.column_type = Categorical;
    tree.col_num = col;
    tree.split_subset.resize(2);
    tree.split_subset[0] = 1;
    tree.split_subset[1] = 0;
    tree.split_subset.shrink_to_fit();
}

void set_tree_as_categorical(ClusterTree &tree, size_t col, int ncat)
{
    tree.column_type = Categorical;
    tree.col_num = col;
    tree.binary_branches.resize(ncat, 0);
    tree.binary_branches.shrink_to_fit();
    tree.split_subset.shrink_to_fit();
}

void set_tree_as_ordinal(ClusterTree &tree, int split_lev, size_t col)
{
    tree.column_type = Ordinal;
    tree.split_lev = split_lev;
    tree.col_num = col;
}



/* After presenting outliers, it's not necessary to retain their details about column/cluster/tree/etc. */
void forget_row_outputs(ModelOutputs &model_outputs)
{
    model_outputs.outlier_scores_final.clear();
    model_outputs.outlier_clusters_final.clear();
    model_outputs.outlier_columns_final.clear();
    model_outputs.outlier_trees_final.clear();
    model_outputs.outlier_depth_final.clear();

    model_outputs.outlier_scores_final.shrink_to_fit();
    model_outputs.outlier_clusters_final.shrink_to_fit();
    model_outputs.outlier_columns_final.shrink_to_fit();
    model_outputs.outlier_trees_final.shrink_to_fit();
    model_outputs.outlier_depth_final.shrink_to_fit();
}

void allocate_row_outputs(ModelOutputs &model_outputs, size_t nrows, size_t max_depth)
{
    forget_row_outputs(model_outputs);
    model_outputs.outlier_scores_final.resize(nrows, 1.0);
    model_outputs.outlier_clusters_final.resize(nrows, 0);
    model_outputs.outlier_columns_final.resize(nrows);
    model_outputs.outlier_trees_final.resize(nrows);
    model_outputs.outlier_depth_final.resize(nrows, max_depth + 2);

    model_outputs.outlier_scores_final.shrink_to_fit();
    model_outputs.outlier_clusters_final.shrink_to_fit();
    model_outputs.outlier_columns_final.shrink_to_fit();
    model_outputs.outlier_trees_final.shrink_to_fit();
    model_outputs.outlier_depth_final.shrink_to_fit();
}
