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
#include "outlier_tree.hpp"

/*    Fit outliers model based on conditional distributions obtaines thorugh decision-tree splitting
*    
*    Note1: the function here will not perform any data validation - it must be done from outside already.
*    Note2: the data types (double/int) were chosen due to R's internal representations of data structures,
*           which only supports those types.
*    
*    Parameters:
*    - model_outputs (out)
*        Struct with the model outputs required for prediction time (trees and clusters) and information about identified outliers
*        required to display their statistics. If there was any previous information from fitting the model to other data, it will
*        be overwritten.
*    - numeric_data[n * m1] (in)
*        Array with numerical columns in the data. Must be ordered by columns like Fortran arrays.
*        Missing values should be encoded as NaN. Infinite values in most sections are treated as NaN too.
*        Binary or boolean columns must be passed as categorical.
*        If there are no numerical columns, pass NULL.
*    - ncols_numeric (in)
*        Number of numeric columns in the array 'numeric_data'.
*    - categorical_data[n * m2] (in)
*        Array with categorical columns in the data. Must be ordered by columns like Fortran arrays.
*        Negative numbers will be interpreted as missing values. Numeration must start at zero and be
*        contiguous (i.e. if there's category 2, must also have category 1).
*        If there are no categorical columns, pass NULL.
*    - ncols_categ (in)
*        Numer of categorical columns in the array 'categorical_data'.
*    - ncat[m2] (in)
*        Numer of categories in each categorical column. If there are no categorical columns, pass NULL.
*    - ordinal_data[n * m3] (in)
*        Array with ordinal categorical columns in the data. Must be ordered by columns like Fortran arrays.
*        Same rules as for categorical data. Note that the order will only be taken into consideration when
*        producing splits by these columns, but outliers are still detected in the same way as for categoricals.
*        Binary or boolean columns must be passed as categorical (i.e. minimum categories in a column is 3).
*        If there are no ordinal columns, pass NULL.
*    - ncols_ord (in)
*        Numer of ordinal columns in the array 'ordinal_data'.
*    - ncat_ord[m3] (in)
*        Numer of categories in each ordinal column. If there are no categorical columns, pass NULL.
*    - nrows (in)
*        Numer of rows in the arrays passed above.
*    - cols_ignore[m1 + m2 + m3] (in)
*        Boolean array indicating which columns should only be used as splitting criterion for other columns,
*        while being ignored at the moment of finding outlier values in them. Pass NULL if outliers are to be
*        searched for in all columns (this is the default).
*    - nthreads (in)
*        Numer of parallel threads to use. Should not be higher than the number of columns.
*        Note that the more threads used, the more memory will need to be allocated.
*    - categ_as_bin (in)
*        Whether to binarize categorical columns at each category to split them by another categorical column.
*        If this is false and 'cat_bruteforce_subset' is also false, then when splitting a categorical or ordinal
*        variable by another categorical, it will have one branch per category of the splitting column. Ignored
*        when splitting by numerical and ordinal. Overrides 'cat_bruteforce_subset' when passing true.
*    - ord_as_bin (in)
*        Same as above, but binarization is by less/greater than a level in the order.
*    - cat_bruteforce_subset (in)
*        Whether to do a brute-force search over all possible binary splits of grouped subsets of categories when
*        splitting a categorical or ordinal column by another categorical column. If this is false and 'categ_as_bin'
*        is also false, then when splitting a categorical or ordinal variable by another categorical, it will have
*        one branch per category of the splitting column. Ignored when splitting by numerical and ordinal.
*        Will be ignored when passing 'categ_as_bin' = true.
*    - categ_from_maj (in)
*        Whether to flag outliers in categorical variables according to the number of observations not belonging to
*        the majority class (formula will be (n-n_maj)/(n * p_prior) < 1/(z_outlier^2) for each category). If passing
*        'false', will instead look for outliers in categorical variables based on being a minority and having a gap
*        with respect to other categories, even if there is no dominant majority.
*    - max_depth (in)
*        Max depth of decision trees that generate conditional distributions (subsets of the data) in which to look
*        for outliers.
*    - max_perc_outliers (in)
*        Model parameter. Approximate maximum percentage of outlier observations in each cluster. Default value is 0.01.
*    - min_size_numeric (in)
*        Minimum size that numeric clusters and splits on numeric variables can have. Default value is 35.
*    - min_size_categ (in)
*        Same but for categoricals. Default value is 75.
*    - min_gain (in)
*        Minimum gain that a split must produce in order not to discard it. Default value is 0.01 (in GritBot it's 0.000001).
*    - gain_as_pct (in)
*        Whether the gain above should be taken in absolute terms (sd_full - (n1*sd1 + n2*sd2)/n), or as a percentage
*        ( (sd_full - (n1*sd1 + n2*sd2)/n) / sd_full ) (Replace 'sd' with shannon entropy for categorical variables).
*        Taking it in absolute terms will prefer making more splits on columns that have a large variance, while taking it
*        as a percentage might be more restrictive on them and might create deeper trees in some columns.
*    - follow_all (in)
*        Whether to create new tree branches (and continue creating new splits from all of them) from every split that meets them
*        minimum gain or not. Doing so (which GritBot doesn't) will make the procedure much slower, but can flag more observations
*        as outliers (with a much larger false-positive rate). Default is 'false'.
*    - z_norm (in)
*        Maximum Z value that is considered as normal in a distribution. Default value is 2.67 (percentile 99)
*    - z_outlier (in)
*        Minimum Z value that can be considered as outlier in numerical columns. Not used for categorical or ordinal columns.
*    
*    Returns:
*        Whether any outliers were identified in the data to which the model was fit.
*/
bool fit_outliers_models(ModelOutputs &model_outputs,
                         double *restrict numeric_data,     size_t ncols_numeric,
                         int    *restrict categorical_data, size_t ncols_categ,   int *restrict ncat,
                         int    *restrict ordinal_data,     size_t ncols_ord,     int *restrict ncat_ord,
                         size_t nrows, char *restrict cols_ignore, int nthreads,
                         bool   categ_as_bin, bool ord_as_bin, bool cat_bruteforce_subset, bool categ_from_maj, bool take_mid,
                         size_t max_depth, double max_perc_outliers, size_t min_size_numeric, size_t min_size_categ,
                         double min_gain, bool gain_as_pct, bool follow_all, double z_norm, double z_outlier)
{
    SignalSwitcher ss = SignalSwitcher();

    /* put parameters and data into structs to avoid passing too many function arguments each time */
    double z_tail = z_outlier - z_norm;
    ModelParams model_params = {
                                categ_as_bin, ord_as_bin, cat_bruteforce_subset, categ_from_maj, take_mid,
                                max_depth, max_perc_outliers, min_size_numeric, min_size_categ,
                                min_gain, gain_as_pct, follow_all, z_norm, z_outlier, z_tail,
                                std::vector<long double>()
                            };

    size_t tot_cols = ncols_numeric + ncols_categ + ncols_ord;
    InputData input_data = {
                            numeric_data, ncols_numeric, categorical_data, ncols_categ, ncat,
                            ordinal_data, ncols_ord, ncat_ord, nrows, tot_cols, std::vector<char>(),
                            std::vector<char>(), -1, std::vector<size_t>(),
                        };

    model_outputs.ncat.assign(ncat, ncat + ncols_categ);
    model_outputs.ncat_ord.assign(ncat_ord, ncat_ord + ncols_ord);
    model_outputs.ncols_numeric = ncols_numeric;
    model_outputs.ncols_categ = ncols_categ;
    model_outputs.ncols_ord = ncols_ord;
    model_outputs.max_depth = max_depth;
    model_outputs.min_outlier_any_cl.resize(model_outputs.ncols_numeric, -HUGE_VAL);
    model_outputs.max_outlier_any_cl.resize(model_outputs.ncols_numeric,  HUGE_VAL);
    model_outputs.cat_outlier_any_cl.resize(model_outputs.ncols_categ + model_outputs.ncols_ord);

    if (tot_cols < (size_t)nthreads)
        nthreads = (int) tot_cols;
    #ifndef _OPENMP
        std::vector<Workspace> workspace(1);
    #else
        std::vector<Workspace> workspace(nthreads);
    #endif
    workspace.shrink_to_fit();

    /* in case the model was already fit from before */
    model_outputs.all_clusters.clear();
    model_outputs.all_trees.clear();
    allocate_row_outputs(model_outputs, nrows, max_depth);

    /* initialize info holders as needed */
    bool found_outliers = false;
    input_data.has_NA.resize(tot_cols, false);
    input_data.skip_col.resize(tot_cols, false);
    model_outputs.start_ix_cat_counts.resize(ncols_categ + ncols_ord + 1);
    model_outputs.col_transf.resize(ncols_numeric, NoTransf);
    model_outputs.transf_offset.resize(ncols_numeric);
    model_outputs.sd_div.resize(ncols_numeric);
    model_outputs.min_decimals_col.resize(ncols_numeric);

    /* determine maximum number of categories in a column, allocate arrays for category counts and proportions */
    model_outputs.start_ix_cat_counts[0] = 0;
    if (tot_cols > ncols_numeric) {
        input_data.max_categ = calculate_category_indices(&model_outputs.start_ix_cat_counts[0], input_data.ncat, input_data.ncols_categ,
                                                          (bool*) input_data.skip_col.data() + ncols_numeric);
        input_data.max_categ = calculate_category_indices(model_outputs.start_ix_cat_counts.data() + input_data.ncols_categ, input_data.ncat_ord, input_data.ncols_ord,
                                                          (bool*) input_data.skip_col.data() + input_data.ncols_numeric + input_data.ncols_categ, input_data.max_categ);
    } else {
        input_data.max_categ = 0;
    }

    /* now allocate arrays for proportions */
    input_data.cat_counts.resize(model_outputs.start_ix_cat_counts[ncols_categ + ncols_ord], 0);
    model_params.prop_small.resize(model_outputs.start_ix_cat_counts[ncols_categ + ncols_ord]);
    model_outputs.prop_categ.resize(model_outputs.start_ix_cat_counts[ncols_categ + ncols_ord]);

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return false;
    #endif

    /* calculate prior probabilities for categorical variables (in parallel), see if any is unsplittable */
    if (tot_cols > ncols_numeric) {
        #pragma omp parallel
        {
            #pragma omp sections
            {

                #pragma omp section
                {
                    if (ncols_categ > 0) {
                        calculate_all_cat_counts(&model_outputs.start_ix_cat_counts[0], &input_data.cat_counts[0], input_data.ncat,
                                                 input_data.categorical_data, input_data.ncols_categ, input_data.nrows,
                                                 (bool*) &input_data.has_NA[ncols_numeric], (bool*) &input_data.skip_col[input_data.ncols_numeric],
                                                 std::min(input_data.ncols_categ, (size_t)std::max(1, nthreads - 1)) );

                        check_cat_col_unsplittable(&model_outputs.start_ix_cat_counts[0], &input_data.cat_counts[0], input_data.ncat,
                                                   input_data.ncols_categ, std::min(model_params.min_size_numeric, model_params.min_size_categ), input_data.nrows,
                                                   (bool*) &input_data.skip_col[input_data.ncols_numeric],
                                                   std::min(input_data.ncols_categ, (size_t)std::max(1, nthreads - 1)));
                    }


                }

                #pragma omp section
                {
                    if (ncols_ord > 0) {
                        calculate_all_cat_counts(&model_outputs.start_ix_cat_counts[input_data.ncols_categ], &input_data.cat_counts[0], input_data.ncat_ord,
                                                 input_data.ordinal_data, input_data.ncols_ord, input_data.nrows,
                                                 (bool*) &input_data.has_NA[input_data.ncols_numeric + input_data.ncols_categ],
                                                 (bool*) &input_data.skip_col[input_data.ncols_numeric + input_data.ncols_categ],
                                                 std::max((int)1, nthreads - (int)input_data.ncols_categ) );

                        check_cat_col_unsplittable(&model_outputs.start_ix_cat_counts[input_data.ncols_categ], &input_data.cat_counts[0], input_data.ncat_ord,
                                                   ncols_ord, std::min(model_params.min_size_numeric, model_params.min_size_categ), input_data.nrows,
                                                   (bool*) &input_data.skip_col[input_data.ncols_numeric + input_data.ncols_categ],
                                                   std::max((int)1, nthreads - (int)input_data.ncols_categ));
                    }
                }
            }

        }
    

        /* calculate proprotion limit and CI for each category of each column */
        calculate_lowerlim_proportion(&model_params.prop_small[0], &model_outputs.prop_categ[0], &model_outputs.start_ix_cat_counts[0],
                                      &input_data.cat_counts[0], input_data.ncols_categ, input_data.nrows, model_params.z_norm, model_params.z_tail);
        calculate_lowerlim_proportion(&model_params.prop_small[0], &model_outputs.prop_categ[0], &model_outputs.start_ix_cat_counts[input_data.ncols_categ],
                                      &input_data.cat_counts[0], input_data.ncols_ord,  input_data.nrows, model_params.z_norm, model_params.z_tail);
    }

    /* for numerical columns, check if they have NAs or if total variance is  too small */
    check_missing_no_variance(input_data.numeric_data, input_data.ncols_numeric, input_data.nrows,
                              (bool*) &input_data.has_NA[0], (bool*) &input_data.skip_col[0],
                              model_outputs.min_decimals_col.data(), nthreads);

    /* determine an approximate size for the output clusters, and reserve memory right away */
    model_outputs.all_clusters.resize(tot_cols);
    model_outputs.all_trees.resize(tot_cols);
    #pragma omp parallel for shared(model_outputs, input_data, model_params, tot_cols)
    for (size_t_for col = 0; col < tot_cols; col++) {
        if (input_data.skip_col[col]) continue;
        if (cols_ignore != NULL && cols_ignore[col]) continue;
        model_outputs.all_clusters[col].reserve(tot_cols * std::min(2 * input_data.nrows, pow2(model_params.max_depth + 1)));
        model_outputs.all_trees[col].reserve( square(model_params.max_depth) );
        /* this is not exact as categoricals and ordinals can also be split multiple times */
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return false;
    #endif

    /* now run the procedure on each column separately */
    int tid;
    bool threw_exception = false;
    std::exception_ptr ex = NULL;
    nthreads = std::min(nthreads, (int)(ncols_numeric + ncols_categ + ncols_ord));
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1) private(tid) shared(workspace, model_outputs, input_data, model_params, tot_cols)
    for (size_t_for col = 0; col < tot_cols; col++) {

        if (interrupt_switch || threw_exception) continue;

        if (cols_ignore != NULL && cols_ignore[col]) continue;
        if (input_data.skip_col[col] && col < input_data.ncols_numeric) continue;
        tid = omp_get_thread_num();

        try {
            /* re-use thread-private memory if possible */
            if (!check_workspace_is_allocated(workspace[tid]))
                allocate_thread_workspace(workspace[tid], input_data.nrows, input_data.max_categ);
                
            /* numerical column */
            if (col < input_data.ncols_numeric) {
                process_numeric_col(model_outputs.all_clusters[col],
                                    model_outputs.all_trees[col],
                                    col,
                                    workspace[tid],
                                    input_data,
                                    model_params, model_outputs);
                calculate_cluster_minimums(model_outputs, col);
            }

            /* categorical column */
            else if (col < (input_data.ncols_numeric + input_data.ncols_categ)) {
                process_categ_col(model_outputs.all_clusters[col],
                                  model_outputs.all_trees[col],
                                  col, false,
                                  workspace[tid],
                                  input_data,
                                  model_params, model_outputs);
                calculate_cluster_poss_categs(model_outputs, col, col - input_data.ncols_numeric);
            }

            /* ordinal column */
            else {
                process_categ_col(model_outputs.all_clusters[col],
                                  model_outputs.all_trees[col],
                                  col, true,
                                  workspace[tid],
                                  input_data,
                                  model_params, model_outputs);
                calculate_cluster_poss_categs(model_outputs, col, col - input_data.ncols_numeric);
            }

            /* shrink the dynamic vectors to what ended up used only */
            #ifdef TEST_MODE_DEFINE
            prune_unused_trees(model_outputs.all_trees[col]);
            #endif
            if (
                model_outputs.all_clusters[col].size() == 0 ||
                model_outputs.all_trees[col].size() == 0 ||
                check_tree_is_not_needed(model_outputs.all_trees[col][0])
            )
            {
                model_outputs.all_trees[col].clear();
                model_outputs.all_clusters[col].clear();
            }
            model_outputs.all_trees[col].shrink_to_fit();
            model_outputs.all_clusters[col].shrink_to_fit();
            
            /* simplify single-elements in subset to 'equals' or 'not equals' */
            simplify_when_equal_cond(model_outputs.all_clusters[col], ncat_ord);
            simplify_when_equal_cond(model_outputs.all_trees[col],    ncat_ord);

            /* remember only the best (rarest) value for each row */
            #pragma omp critical
            if (workspace[tid].col_has_outliers) {

                found_outliers = true;
                for (size_t row = 0; row < input_data.nrows; row++) {

                    if (workspace[tid].outlier_scores[row] < 1.0) {

                        if (
                            model_outputs.outlier_scores_final[row] >= 1.0 ||
                            (
                                workspace[tid].outlier_depth[row] < model_outputs.outlier_depth_final[row] &&
                                (
                                    !model_outputs.all_clusters[col][workspace[tid].outlier_clusters[row]].has_NA_branch ||
                                    model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
                                )
                            ) ||
                                (
                                model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch &&
                                !model_outputs.all_clusters[col][workspace[tid].outlier_clusters[row]].has_NA_branch
                                ) ||
                                (
                                workspace[tid].outlier_depth[row] == model_outputs.outlier_depth_final[row] &&
                                    model_outputs.all_clusters[col][workspace[tid].outlier_clusters[row]].has_NA_branch
                                        ==
                                    model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
                                    &&
                                    model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].cluster_size
                                        <
                                    model_outputs.all_clusters[col][workspace[tid].outlier_clusters[row]].cluster_size
                                ) ||
                                (
                                    workspace[tid].outlier_depth[row] == model_outputs.outlier_depth_final[row] &&
                                    model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].cluster_size
                                        ==
                                    model_outputs.all_clusters[col][workspace[tid].outlier_clusters[row]].cluster_size
                                    &&
                                    model_outputs.all_clusters[col][workspace[tid].outlier_clusters[row]].has_NA_branch
                                        ==
                                    model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
                                    &&
                                    workspace[tid].outlier_scores[row] < model_outputs.outlier_scores_final[row]
                                )
                        )
                        {
                            model_outputs.outlier_scores_final[row] = workspace[tid].outlier_scores[row];
                            model_outputs.outlier_clusters_final[row] = workspace[tid].outlier_clusters[row];
                            model_outputs.outlier_trees_final[row] = workspace[tid].outlier_trees[row];
                            model_outputs.outlier_depth_final[row] = workspace[tid].outlier_depth[row];
                            model_outputs.outlier_columns_final[row] = col;
                        }
                    }

                }
            }
        }

        catch(...) {
            #pragma omp critical
            {
                if (!threw_exception) {
                    threw_exception = true;
                    ex = std::current_exception();
                }
            }
        }
    }

    check_interrupt_switch(ss);
    #if defined(DONT_THROW_ON_INTERRUPT)
    if (interrupt_switch) return false;
    #endif

    if (threw_exception)
        std::rethrow_exception(ex);

    /* once finished, determine how many decimals to report for numerical outliers */
    if (found_outliers)
      calc_min_decimals_to_print(model_outputs, input_data.numeric_data, nthreads);

    #ifdef TEST_MODE_DEFINE
    for (size_t col = 0; col < tot_cols; col++) {
        std::cout << "col " << col << " has " << model_outputs.all_clusters[col].size() << " clusters [" << model_outputs.all_trees[col].size() << " trees]" << std::endl;
    }

    find_new_outliers(numeric_data,
                      categorical_data,
                      ordinal_data,
                      nrows, nthreads, model_outputs);


    // /* extract data for only one row */
    // std::vector<double> num_data_row(ncols_numeric);
    // std::vector<int> cat_data_row(ncols_categ);
    // std::vector<int> ord_data_row(ncols_ord);
    // size_t chosen_row = 38;
    // for (size_t rowcol = 0; rowcol < ncols_numeric; rowcol++)
    //     num_data_row.at(rowcol) = numeric_data[chosen_row + rowcol * nrows];
    // for (size_t rowcol = 0; rowcol < ncols_categ; rowcol++)
    //     cat_data_row.at(rowcol) = categorical_data[chosen_row + rowcol * nrows];
    // for (size_t rowcol = 0; rowcol < ncols_ord; rowcol++)
    //     ord_data_row.at(rowcol) = ordinal_data[chosen_row + rowcol * nrows];


    // find_new_outliers(&num_data_row[0],
    //                    &cat_data_row[0],
    //                    &ord_data_row[0],
    //                    1, 1, model_outputs);
    // calc_min_printable_digits(model_outputs);
    #endif

    return found_outliers;
}

void process_numeric_col(std::vector<Cluster> &cluster_root,
                         std::vector<ClusterTree> &tree_root,
                         size_t target_col_num,
                         Workspace &workspace,
                         InputData &input_data,
                         ModelParams &model_params,
                         ModelOutputs &model_outputs)
{
    if (interrupt_switch) return;

    /* discard NAs and infinites */
    workspace.target_col_num = target_col_num;
    workspace.target_numeric_col = input_data.numeric_data + target_col_num * input_data.nrows;
    workspace.orig_target_col = workspace.target_numeric_col;
    workspace.end = input_data.nrows - 1;
    workspace.st = move_NAs_to_front(&workspace.ix_arr[0], workspace.target_numeric_col, 0, workspace.end, true);
    workspace.col_has_outliers = false;

    /* check for problematic distributions - need to sort data first */
    std::sort(&workspace.ix_arr[0] + workspace.st, &workspace.ix_arr[0] + workspace.end + 1,
              [&workspace](const size_t a, const size_t b){return workspace.target_numeric_col[a] < workspace.target_numeric_col[b];});

    long double running_mean = 0;
    long double running_ssq  = 0;
    long double mean_prev    = workspace.target_numeric_col[workspace.ix_arr[workspace.st]];
    double xval;
    for (size_t row = workspace.st; row <= workspace.end; row++) {
        xval = workspace.target_numeric_col[workspace.ix_arr[row]];
        running_mean += (xval - running_mean) / (long double)(row - workspace.st + 1);
        running_ssq  += (xval - running_mean) * (xval - mean_prev);
        mean_prev     = running_mean;
    }
    
    check_for_tails(&workspace.ix_arr[0], workspace.st, workspace.end, workspace.target_numeric_col,
                    model_params.z_norm, model_params.max_perc_outliers,
                    &workspace.buffer_transf_y[0], (double)running_mean,
                    (double)sqrtl(running_ssq / (long double)(workspace.end - workspace.st)),
                    &workspace.left_tail, &workspace.right_tail,
                    &workspace.exp_transf, &workspace.log_transf);

    /* if it's double-tailed, skip it as this model doesn't work properly with this */
    if ( (workspace.exp_transf || !isinf(workspace.left_tail)) && (workspace.log_transf || !isinf(workspace.right_tail)) ) return;

    /* apply log or exp transformation if necessary */
    if (workspace.exp_transf) {

        workspace.orig_mean = (double) running_mean;
        workspace.orig_sd   = (double) sqrtl(running_ssq / (long double)(workspace.end - workspace.st));
        for (size_t row = workspace.st; row <= workspace.end; row++) {
            workspace.buffer_transf_y[workspace.ix_arr[row]] = exp(z_score(workspace.target_numeric_col[workspace.ix_arr[row]], workspace.orig_mean, workspace.orig_sd));
        }
        workspace.target_numeric_col = &workspace.buffer_transf_y[0];
        model_outputs.col_transf[workspace.target_col_num] = Exp;
        model_outputs.transf_offset[workspace.target_col_num] = workspace.orig_mean;
        model_outputs.sd_div[workspace.target_col_num] = workspace.orig_sd;


    } else if (workspace.log_transf) {

        if (workspace.target_numeric_col[workspace.ix_arr[workspace.st]] == 0) {
            workspace.log_minval = -1;
        } else {
            workspace.log_minval = workspace.target_numeric_col[workspace.ix_arr[workspace.st]] - 1e-3;
        }

        for (size_t row = workspace.st; row <= workspace.end; row++) {
            workspace.buffer_transf_y[workspace.ix_arr[row]] = log(workspace.target_numeric_col[workspace.ix_arr[row]] - workspace.log_minval);
        }
        workspace.target_numeric_col = &workspace.buffer_transf_y[0];
        model_outputs.col_transf[workspace.target_col_num] = Log;
        model_outputs.transf_offset[workspace.target_col_num] = workspace.log_minval;

    }

    /* create a cluster with no conditions */
    workspace.clusters = &cluster_root;
    workspace.tree = &tree_root;
    std::fill(workspace.outlier_scores.begin(), workspace.outlier_scores.end(), (double)1.0);
    workspace.tree->emplace_back(0, Root);

    workspace.clusters->emplace_back(NoType, Root);
    workspace.col_has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.st,
                                                          workspace.end, &workspace.outlier_scores[0],
                                                          &workspace.outlier_clusters[0], &workspace.outlier_trees[0], &workspace.outlier_depth[0],
                                                          workspace.clusters->back(), *(workspace.clusters), 0, 0, 0,
                                                          workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                          workspace.orig_mean, workspace.orig_sd,
                                                          workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                          model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
    workspace.tree->back().clusters.push_back(0);

    /* remove outliers if any were found */
    if (workspace.has_outliers)
        workspace.st = move_outliers_to_front(&workspace.ix_arr[0], &workspace.outlier_scores[0], workspace.st, workspace.end);

    /* update statistics if they've changed */
    if (workspace.has_outliers || workspace.exp_transf || workspace.log_transf)
        workspace.sd_y = calc_sd(&workspace.ix_arr[0], workspace.target_numeric_col,
                                 workspace.st, workspace.end, &workspace.mean_y);
    else
        workspace.sd_y = sqrtl(running_ssq / (long double)(workspace.end - workspace.st));

    if (model_params.max_depth > 0) recursive_split_numeric(workspace, input_data, model_params, 0, false);
}

void recursive_split_numeric(Workspace &workspace,
                             InputData &input_data,
                             ModelParams &model_params,
                             size_t curr_depth, bool is_NA_branch)
{
    if (interrupt_switch) return;

    workspace.best_gain = -HUGE_VAL;
    workspace.column_type_best = NoType;
    workspace.lev_has_outliers = false;
    if (curr_depth > 0) workspace.sd_y = calc_sd(&workspace.ix_arr[0], workspace.target_numeric_col,
                                                 workspace.st, workspace.end, &workspace.mean_y);

    /* these are used to keep track of where to continue after calling a further recursion */
    size_t ix1, ix2, ix3;
    SplitType spl1, spl2;
    size_t tree_from = workspace.tree->size() - 1;

    /* when using 'follow_all' need to keep track of a lot more things */
    std::unique_ptr<RecursionState> state_backup;
    if (model_params.follow_all) state_backup = std::unique_ptr<RecursionState>(new RecursionState);

    
    /* procedure: split with each other column */

    /* first numeric */
    for (size_t col = 0; col < input_data.ncols_numeric; col++) {

        if (col == workspace.target_col_num) continue;
        if (input_data.skip_col[col]) continue;
        split_numericx_numericy(&workspace.ix_arr[0], workspace.st, workspace.end, input_data.numeric_data + col * input_data.nrows,
                                workspace.target_numeric_col, workspace.sd_y, (bool)(input_data.has_NA[col]), model_params.min_size_numeric,
                                model_params.take_mid, &workspace.buffer_sd[0], &(workspace.this_gain), &(workspace.this_split_point),
                                &(workspace.this_split_ix), &(workspace.this_split_NA));
        if (model_params.gain_as_pct) workspace.this_gain /= workspace.sd_y;

        /* if the gain is not insignificant, check clusters created by this split */
        if (workspace.this_gain >= model_params.min_gain) {

            /* NA branch */
            if (workspace.this_split_NA > workspace.st &&
                (workspace.this_split_NA - workspace.st) > model_params.min_size_numeric) {

                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Numeric, col, IsNa, -HUGE_VAL, true);
                workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.st,
                                                                  workspace.this_split_NA - 1, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                                  &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                                  workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                                  workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                                  workspace.orig_mean, workspace.orig_sd,
                                                                  workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                                  model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, HUGE_VAL, IsNa);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.end = workspace.this_split_NA - 1;
                    recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, true);
                    restore_recursion_state(workspace, *state_backup);
                }
                
            }

            /* left branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Numeric, col, LessOrEqual, workspace.this_split_point, is_NA_branch);
            workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.this_split_NA,
                                                              workspace.this_split_ix, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                              &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                              workspace.orig_mean, workspace.orig_sd,
                                                              workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_point, LessOrEqual);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_NA;
                workspace.end = workspace.this_split_ix;
                recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }


            /* right branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Numeric, col, Greater, workspace.this_split_point, is_NA_branch);
            workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.this_split_ix + 1,
                                                              workspace.end, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                              &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                              workspace.orig_mean, workspace.orig_sd,
                                                              workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_point, Greater);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_ix + 1;
                recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }


            /* if this is the best split, remember it for later */
            if (workspace.this_gain > workspace.best_gain && !model_params.follow_all) {
                workspace.best_gain = workspace.this_gain;
                workspace.column_type_best = Numeric;
                workspace.col_best = col;
                workspace.split_point_best = workspace.this_split_point;
            }

        }

    }

    /* then categorical */
    for (size_t col = 0; col < input_data.ncols_categ; col++) {

        if (input_data.skip_col[col + input_data.ncols_numeric]) continue;

        split_categx_numericy(&workspace.ix_arr[0], workspace.st, workspace.end, input_data.categorical_data + col * input_data.nrows,
                              workspace.target_numeric_col, workspace.sd_y, workspace.mean_y, false, input_data.ncat[col], &workspace.buffer_cat_cnt[0],
                              &workspace.buffer_cat_sum[0], &workspace.buffer_cat_sum_sq[0], &workspace.buffer_cat_sorted[0],
                              (bool)(input_data.has_NA[col + input_data.ncols_numeric]), model_params.min_size_numeric,
                              &(workspace.this_gain), &workspace.buffer_subset_categ[0], NULL);
        if (model_params.gain_as_pct) workspace.this_gain /= workspace.sd_y;

        if (workspace.this_gain >= model_params.min_gain) {

            /* data is not arranged inside the splitting function, need to now assign to the branches as determined */
            divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + col * input_data.nrows, workspace.st, workspace.end,
                                &workspace.buffer_subset_categ[0], input_data.ncat[col], (bool)(workspace.buffer_cat_cnt[input_data.ncat[col]] > 0),
                                &(workspace.this_split_NA), &(workspace.this_split_ix));

            /* NA branch */
            if ((workspace.this_split_NA - workspace.st) > model_params.min_size_numeric) {

                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Categorical, col, IsNa, (signed char*)NULL, (int)0, true);
                workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.st,
                                                                  workspace.this_split_NA - 1, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                                  &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                                  workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                                  workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                                  workspace.orig_mean, workspace.orig_sd,
                                                                  workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                                  model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, IsNa, (signed char*)NULL, 0);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.end = workspace.this_split_NA - 1;
                    recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, true);
                    restore_recursion_state(workspace, *state_backup);
                }

            }

            /* left branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Categorical, col, InSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col], is_NA_branch);
            workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.this_split_NA,
                                                              workspace.this_split_ix - 1, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                              &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                              workspace.orig_mean, workspace.orig_sd,
                                                              workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, InSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col]);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_NA;
                workspace.end = workspace.this_split_ix - 1;
                recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }

            /* right branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Categorical, col, NotInSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col], is_NA_branch);
            workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.this_split_ix,
                                                              workspace.end, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                              &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                              workspace.orig_mean, workspace.orig_sd,
                                                              workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, NotInSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col]);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_ix;
                recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }

            if (workspace.this_gain > workspace.best_gain && !model_params.follow_all) {
                workspace.best_gain = workspace.this_gain;
                workspace.column_type_best = Categorical;
                workspace.col_best = col;
                memcpy(&workspace.buffer_subset_categ_best[0], &workspace.buffer_subset_categ[0], input_data.ncat[col] * sizeof(signed char));
            }

        }

    }

    /* then ordinal */
    for (size_t col = 0; col < input_data.ncols_ord; col++) {

        if (input_data.skip_col[col + input_data.ncols_numeric + input_data.ncols_categ]) continue;

        /* same code as for categorical, but this time with split level as int instead of boolean array as subset */
        split_categx_numericy(&workspace.ix_arr[0], workspace.st, workspace.end, input_data.ordinal_data + col * input_data.nrows,
                              workspace.target_numeric_col, workspace.sd_y, workspace.mean_y, true, input_data.ncat_ord[col], &workspace.buffer_cat_cnt[0],
                              &workspace.buffer_cat_sum[0], &workspace.buffer_cat_sum_sq[0], &workspace.buffer_cat_sorted[0],
                              (bool)(input_data.has_NA[col + input_data.ncols_numeric + input_data.ncols_categ]), model_params.min_size_numeric,
                              &(workspace.this_gain), &workspace.buffer_subset_categ[0], &(workspace.this_split_lev));
        if (model_params.gain_as_pct) workspace.this_gain /= workspace.sd_y;

        if (workspace.this_gain >= model_params.min_gain) {

            divide_subset_split(&workspace.ix_arr[0], input_data.ordinal_data + col * input_data.nrows, workspace.st, workspace.end,
                                workspace.this_split_lev, (bool)(workspace.buffer_cat_cnt[ input_data.ncat_ord[col] ] > 0),
                                &(workspace.this_split_NA), &(workspace.this_split_ix) );

            if ((workspace.this_split_NA - workspace.st) > model_params.min_size_numeric) {

                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Ordinal, col, IsNa, (int)0, true);
                workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.st,
                                                                  workspace.this_split_NA - 1, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                                  &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                                  workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                                  workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                                  workspace.orig_mean, workspace.orig_sd,
                                                                  workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                                  model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, (int)-1, IsNa);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.end = workspace.this_split_NA - 1;
                    recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, true);
                    restore_recursion_state(workspace, *state_backup);
                }

            }

            /* left branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Ordinal, col, LessOrEqual, workspace.this_split_lev, is_NA_branch);
            workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.this_split_NA,
                                                              workspace.this_split_ix - 1, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                              &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                              workspace.orig_mean, workspace.orig_sd,
                                                              workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_lev, LessOrEqual);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_NA;
                workspace.end = workspace.this_split_ix - 1;
                recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }



            /* right branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Ordinal, col, Greater, workspace.this_split_lev, is_NA_branch);
            workspace.has_outliers = define_numerical_cluster(workspace.target_numeric_col, &workspace.ix_arr[0], workspace.this_split_ix,
                                                              workspace.end, &workspace.outlier_scores[0], &workspace.outlier_clusters[0],
                                                              &workspace.outlier_trees[0], &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              workspace.log_transf, workspace.log_minval, workspace.exp_transf,
                                                              workspace.orig_mean, workspace.orig_sd,
                                                              workspace.left_tail, workspace.right_tail, workspace.orig_target_col,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier);
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_lev, Greater);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_ix;
                recursive_split_numeric(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }

            if (workspace.this_gain > workspace.best_gain && !model_params.follow_all) {
                workspace.best_gain = workspace.this_gain;
                workspace.column_type_best = Ordinal;
                workspace.col_best = col;
                workspace.split_lev_best = workspace.this_split_lev;
            }

        }

    }

    /* avoid unnecessary memory usage */
    workspace.col_has_outliers = workspace.lev_has_outliers? true : workspace.col_has_outliers;
    (*workspace.tree)[tree_from].clusters.shrink_to_fit();
    if ((*workspace.tree)[tree_from].all_branches.size() > 0) (*workspace.tree)[tree_from].all_branches.shrink_to_fit();


    /* continue splitting further if meeting threshold criteria */
    if (workspace.best_gain >= model_params.min_gain && !model_params.follow_all) {

        /* check if depth limit is reached */
        curr_depth++;
        if (curr_depth >= model_params.max_depth) return;

        /* discard outliers if any */
        if (workspace.lev_has_outliers)
            workspace.st = move_outliers_to_front(&workspace.ix_arr[0], &workspace.outlier_scores[0], workspace.st, workspace.end);

        /* assign rows to their corresponding branch */
        switch(workspace.column_type_best) {
            case Numeric:
            {
                divide_subset_split(&workspace.ix_arr[0], input_data.numeric_data + workspace.col_best * input_data.nrows,
                                    workspace.st, workspace.end, workspace.split_point_best,
                                    (bool)(input_data.has_NA[workspace.col_best]),
                                    &(workspace.this_split_NA), &(workspace.this_split_ix) );
                spl1 = LessOrEqual; spl2 = Greater;
                set_tree_as_numeric(workspace.tree->back(), workspace.split_point_best, workspace.col_best);
                break;
            }

            case Categorical:
            {
                divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + workspace.col_best * input_data.nrows,
                                    workspace.st, workspace.end, &workspace.buffer_subset_categ_best[0], input_data.ncat[workspace.col_best],
                                    (bool)(input_data.has_NA[workspace.col_best + input_data.ncols_numeric]),
                                    &(workspace.this_split_NA), &(workspace.this_split_ix) );
                spl1 = InSubset; spl2 = NotInSubset;
                set_tree_as_categorical(workspace.tree->back(), input_data.ncat[workspace.col_best],
                                        &workspace.buffer_subset_categ_best[0], workspace.col_best);
                break;
            }

            case Ordinal:
            {
                divide_subset_split(&workspace.ix_arr[0], input_data.ordinal_data + workspace.col_best * input_data.nrows,
                                    workspace.st, workspace.end, workspace.split_lev_best,
                                    (bool)(input_data.has_NA[workspace.col_best + input_data.ncols_numeric + input_data.ncols_categ]),
                                    &(workspace.this_split_NA), &(workspace.this_split_ix) );
                spl1 = LessOrEqual; spl2 = Greater;
                set_tree_as_ordinal(workspace.tree->back(), workspace.split_lev_best, workspace.col_best);
                break;
            }

            default:
            {
                unexpected_error();
            }
        }

        /* continue splitting recursively - need to remember from where */
        ix1 = workspace.this_split_NA;
        ix2 = workspace.this_split_ix;
        ix3 = workspace.end;

        /* NA branch */
        if (workspace.st > workspace.this_split_NA &&
            (workspace.st - workspace.this_split_NA) >= 2 * model_params.min_size_numeric) {

            workspace.end = ix1 - 1;
            (*workspace.tree)[tree_from].tree_NA = workspace.tree->size();
            workspace.tree->emplace_back(tree_from, IsNa);
            recursive_split_numeric(workspace, input_data, model_params, curr_depth, true);
        }

        /* left branch */
        if ((ix2 - ix1) >= 2 * model_params.min_size_numeric) {
            workspace.st = ix1;
            workspace.end = ix2 - 1;
            (*workspace.tree)[tree_from].tree_left = workspace.tree->size();
            workspace.tree->emplace_back(tree_from, spl1);
            recursive_split_numeric(workspace, input_data, model_params, curr_depth, is_NA_branch);
        }

        /* right branch */
        if ((ix3 - ix2 + 1) >= 2 * model_params.min_size_numeric) {
            workspace.st = ix2;
            workspace.end = ix3;
            (*workspace.tree)[tree_from].tree_right = workspace.tree->size();
            workspace.tree->emplace_back(tree_from, spl2);
            recursive_split_numeric(workspace, input_data, model_params, curr_depth, is_NA_branch);
        }

    }

    /* if tree has no clusters and no subtrees, disconnect it from parent and then drop */
    if (check_tree_is_not_needed((*workspace.tree)[tree_from])) {

        if (tree_from == 0) {
            workspace.tree->clear();
        } else if ((*workspace.tree)[(*workspace.tree)[tree_from].parent].all_branches.size() > 0) {
            (*workspace.tree)[(*workspace.tree)[tree_from].parent].all_branches.pop_back();
            workspace.tree->pop_back();
        } else {
            switch((*workspace.tree)[tree_from].parent_branch) {

                case IsNa:
                {
                    (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_NA = 0;
                    break;
                }

                case LessOrEqual:
                {
                    (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_left = 0;
                    break;
                }

                case Greater:
                {
                    (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_right = 0;
                    break;
                }

                case InSubset:
                {
                    (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_left = 0;
                    break;
                }

                case NotInSubset:
                {
                    (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_right = 0;
                    break;
                }

                default:
                {
                    unexpected_error();
                }
            }
            workspace.tree->pop_back();
        }
    }

}

void process_categ_col(std::vector<Cluster> &cluster_root,
                       std::vector<ClusterTree> &tree_root,
                       size_t target_col_num, bool is_ord,
                       Workspace &workspace,
                       InputData &input_data,
                       ModelParams &model_params,
                       ModelOutputs &model_outputs)
{
    if (interrupt_switch) return;

    if (model_params.max_depth <= 0) return;

    /* extract necesary info from column and discard NAs */
    workspace.target_col_is_ord = is_ord;
    workspace.target_col_num = target_col_num - input_data.ncols_numeric;
    if (!workspace.target_col_is_ord) {
        workspace.target_categ_col = input_data.categorical_data + workspace.target_col_num * input_data.nrows;
        workspace.ncat_this = input_data.ncat[workspace.target_col_num];
    } else {
        workspace.target_categ_col = input_data.ordinal_data + (workspace.target_col_num - input_data.ncols_categ) * input_data.nrows;
        workspace.ncat_this = input_data.ncat_ord[workspace.target_col_num - input_data.ncols_categ];
    }
    workspace.untransf_target_col = workspace.target_categ_col;
    workspace.end = input_data.nrows - 1;
    workspace.st = move_NAs_to_front(&workspace.ix_arr[0], workspace.target_categ_col, 0, workspace.end);
    workspace.col_has_outliers = false;
    workspace.col_is_bin = workspace.ncat_this <= 2;
    workspace.prop_small_this = &model_params.prop_small[ model_outputs.start_ix_cat_counts[workspace.target_col_num] ];
    workspace.prior_prob = &model_outputs.prop_categ[ model_outputs.start_ix_cat_counts[workspace.target_col_num] ];

    /* create cluster root and reset outlier scores for this column */
    workspace.clusters = &cluster_root;
    workspace.tree = &tree_root;
    std::fill(workspace.outlier_scores.begin(), workspace.outlier_scores.end(), (double)1.0);
    workspace.tree->emplace_back(0, Root);


    /* at first, see if there's a category with 1-2 observations among only categories with large counts */
    workspace.col_has_outliers = find_outlier_categories_no_cond(&input_data.cat_counts[ model_outputs.start_ix_cat_counts[workspace.target_col_num] ],
                                                                 workspace.ncat_this, workspace.end - workspace.st + 1,
                                                                 &workspace.buffer_subset_categ[0], &(workspace.orig_mean));

    /* if there is any such case, create a cluster for them */
    if (workspace.col_has_outliers) {
        workspace.tree->back().clusters.push_back(0);
        workspace.clusters->emplace_back(NoType, Root);
        define_categ_cluster_no_cond(workspace.untransf_target_col, &workspace.ix_arr[0], workspace.st, workspace.end, workspace.ncat_this,
                                     &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                     &workspace.outlier_depth[0], workspace.clusters->back(),
                                     &input_data.cat_counts[ model_outputs.start_ix_cat_counts[workspace.target_col_num] ],
                                     &workspace.buffer_subset_categ[0], workspace.orig_mean);
        workspace.st = move_outliers_to_front(&workspace.ix_arr[0], &workspace.outlier_scores[0], workspace.st, workspace.end);
    }

    /* if no conditional outliers are required, stop there */
    if (model_params.max_depth == 0) return;

    /* if the rest of the data is all one category, do not process it any further */
    if (workspace.ncat_this == 2 && workspace.col_has_outliers) return;

    /* if there isn't a single catchable outlier category, skip */
    bool should_skip = true;
    for (int cat = 0; cat < workspace.ncat_this; cat++) {

        if (workspace.prop_small_this[cat] > (long double)1 / (long double)(workspace.end - workspace.st + 1 - model_params.min_size_categ))
            should_skip = false;
    }
    if (should_skip) return;


    /* if the column is already binary, or if using multiple categories, or if there are no more categorical columns, split the data as-is */
    if (
        (!model_params.categ_as_bin && !workspace.target_col_is_ord) ||
        (!model_params.ord_as_bin && workspace.target_col_is_ord) ||
        workspace.col_is_bin ||
        input_data.ncols_categ == (1 - ((workspace.target_col_is_ord)? 1 : 0))
    )
    {

        /* calculate base information */
        workspace.base_info = total_info(&input_data.cat_counts[ model_outputs.start_ix_cat_counts[workspace.target_col_num] ],
                                         workspace.ncat_this, workspace.end - workspace.st + 1);
        workspace.base_info_orig = workspace.base_info;

        /* then split */
        recursive_split_categ(workspace, input_data, model_params, 0, false);
    }


    else {
        /* otherwise, process the column 1 category at a time */
        size_t st_orig = workspace.st;
        size_t end_orig = workspace.end;
        size_t cat_counts_bin[2];
        workspace.col_is_bin = true;
        workspace.already_split_main = false;
        workspace.base_info_orig = total_info(&input_data.cat_counts[ model_outputs.start_ix_cat_counts[workspace.target_col_num] ],
                                              workspace.ncat_this, workspace.end - workspace.st + 1);
        workspace.tree->back().column_type = NoType;


        for (int cat = 0; cat < workspace.ncat_this - ((workspace.target_col_is_ord)? 1 : 0); cat++) {

            workspace.st = st_orig;
            workspace.end = end_orig;

            /* convert to binary */
            if (!workspace.target_col_is_ord) {

                for (size_t row = workspace.st; row <= workspace.end; row++) {
                    workspace.buffer_bin_y[workspace.ix_arr[row]] = (workspace.untransf_target_col[workspace.ix_arr[row]] == cat)? 1 : 0;
                }
                cat_counts_bin[0] = workspace.end - workspace.st + 1 - input_data.cat_counts[ cat + model_outputs.start_ix_cat_counts[workspace.target_col_num] ];
                cat_counts_bin[1] = input_data.cat_counts[ cat + model_outputs.start_ix_cat_counts[workspace.target_col_num] ];

            } else {

                for (size_t row = workspace.st; row <= workspace.end; row++) {
                    workspace.buffer_bin_y[workspace.ix_arr[row]] = (workspace.untransf_target_col[workspace.ix_arr[row]] <= cat)? 1 : 0;
                }
                cat_counts_bin[0] = 0;
                cat_counts_bin[1] = workspace.end - workspace.st + 1;
                for (int catcat = 0; catcat <= cat; catcat++) {
                    cat_counts_bin[0] += input_data.cat_counts[ catcat + model_outputs.start_ix_cat_counts[workspace.target_col_num] ];
                    cat_counts_bin[1] -= input_data.cat_counts[ catcat + model_outputs.start_ix_cat_counts[workspace.target_col_num] ];
                }

            }

            if (cat_counts_bin[0] > 0 && cat_counts_bin[1] > 0) {
                workspace.target_categ_col = &workspace.buffer_bin_y[0];
                workspace.base_info = total_info(cat_counts_bin, 2, workspace.end - workspace.st + 1);
                (*workspace.tree)[0].binary_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(0, SubTrees);
                recursive_split_categ(workspace, input_data, model_params, 0, false);
            }

        }
        (*workspace.tree)[0].binary_branches.shrink_to_fit();

    }

}


void recursive_split_categ(Workspace &workspace,
                           InputData &input_data,
                           ModelParams &model_params,
                           size_t curr_depth, bool is_NA_branch)
{
    if (interrupt_switch) return;
    
    /*    idea is the same as its numeric counterpart, only splitting by another categorical
        is less clear how to do and offers different options */
    workspace.best_gain = -HUGE_VAL;
    workspace.column_type_best = NoType;
    workspace.lev_has_outliers = false;
    size_t ix1, ix2, ix3;
    SplitType spl1, spl2;
    size_t tree_from = workspace.tree->size() - 1;

    /* when using 'follow_all' need to keep track of a lot more things */
    std::unique_ptr<RecursionState> state_backup;
    if (model_params.follow_all) state_backup = std::unique_ptr<RecursionState>(new RecursionState);

    if (curr_depth > 0) {
        workspace.base_info_orig = total_info(&workspace.ix_arr[0], workspace.untransf_target_col, workspace.st, workspace.end,
                                              workspace.ncat_this, &workspace.buffer_cat_cnt[0]);

        /* check that there's still more than 1 category */
        size_t ncat_present = 0;
        for (int cat = 0; cat < workspace.ncat_this; cat++) {
            ncat_present += (workspace.buffer_cat_cnt[cat])? 1 : 0;
            if (ncat_present >= 2) break;
        }
        if (ncat_present < 2) goto drop_if_not_needed;
        if (workspace.col_is_bin && workspace.ncat_this > 2) {
            workspace.base_info = total_info(&workspace.ix_arr[0], workspace.target_categ_col, workspace.st, workspace.end,
                                             2, &workspace.buffer_cat_cnt[0]);
            if (workspace.buffer_cat_cnt[0] < model_params.min_size_categ || workspace.buffer_cat_cnt[1] == model_params.min_size_categ) goto drop_if_not_needed;
        } else {
            workspace.base_info = workspace.base_info_orig;
        }
    }

    /* split with each other column */


    /* first numeric */
    for (size_t col = 0; col < input_data.ncols_numeric; col++) {

        if (curr_depth == 0 && workspace.col_is_bin && workspace.ncat_this > 2 && workspace.already_split_main) break;
        if (input_data.skip_col[col]) continue;
        split_numericx_categy(&workspace.ix_arr[0], workspace.st, workspace.end, input_data.numeric_data + col * input_data.nrows,
                              workspace.untransf_target_col, workspace.ncat_this, workspace.base_info_orig,
                              &workspace.buffer_cat_cnt[0], (bool)(input_data.has_NA[col]), model_params.min_size_categ,
                              model_params.take_mid, &(workspace.this_gain), &(workspace.this_split_point),
                              &(workspace.this_split_ix), &(workspace.this_split_NA));
        if (model_params.gain_as_pct) workspace.this_gain /= workspace.base_info_orig;

        if (workspace.this_gain >= model_params.min_gain) {
            
            /* NA branch */
            if (workspace.this_split_NA > workspace.st &&
                (workspace.this_split_NA - workspace.st) > model_params.min_size_categ)  {

                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Numeric, col, IsNa, -HUGE_VAL, true);
                workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                              &workspace.ix_arr[0], workspace.st, workspace.this_split_NA - 1,
                                                              workspace.ncat_this, model_params.categ_from_maj,
                                                              &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                              &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                              workspace.prop_small_this, workspace.prior_prob,
                                                              &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                              &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                if (workspace.drop_cluster) {
                    workspace.clusters->pop_back();
                    (*workspace.tree)[tree_from].clusters.pop_back();
                }

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, HUGE_VAL, IsNa);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.end = workspace.this_split_NA - 1;
                    recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, true);
                    restore_recursion_state(workspace, *state_backup);
                }

            }

            /* left branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Numeric, col, LessOrEqual, workspace.this_split_point, is_NA_branch);
            workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                          &workspace.ix_arr[0], workspace.this_split_NA, workspace.this_split_ix,
                                                          workspace.ncat_this, model_params.categ_from_maj,
                                                          &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                          &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                          workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                          model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                          workspace.prop_small_this, workspace.prior_prob,
                                                          &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                          &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
            if (workspace.drop_cluster) {
                workspace.clusters->pop_back();
                (*workspace.tree)[tree_from].clusters.pop_back();
            }

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_point, LessOrEqual);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_NA;
                workspace.end = workspace.this_split_ix;
                recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }


            /* right branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Numeric, col, Greater, workspace.this_split_point, is_NA_branch);
            workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                          &workspace.ix_arr[0], workspace.this_split_ix + 1, workspace.end,
                                                          workspace.ncat_this, model_params.categ_from_maj,
                                                          &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                          &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                          workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                          model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                          workspace.prop_small_this, workspace.prior_prob,
                                                          &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                          &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
            if (workspace.drop_cluster) {
                workspace.clusters->pop_back();
                (*workspace.tree)[tree_from].clusters.pop_back();
            }

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_point, Greater);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_ix + 1;
                recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }


            /* if this is the best split, remember it for later */
            if (workspace.this_gain > workspace.best_gain) {
                workspace.best_gain = workspace.this_gain;
                workspace.column_type_best = Numeric;
                workspace.col_best = col;
                workspace.split_point_best = workspace.this_split_point;
            }

        }

    }


    /* then categorical */
    for (size_t col = 0; col < input_data.ncols_categ; col++) {

        /* TODO: could make a pre-check that the splitting column up to this recursion still has
           more than 1 category, and skip for this and further recursions otherwise */

        if (col == workspace.target_col_num && !workspace.target_col_is_ord) continue;
        if (input_data.skip_col[col + input_data.ncols_numeric]) continue;

        if (workspace.col_is_bin) {
            
            split_categx_biny(&workspace.ix_arr[0], workspace.st, workspace.end,
                              input_data.categorical_data + col * input_data.nrows, workspace.target_categ_col,
                              input_data.ncat[col], workspace.base_info, &workspace.buffer_cat_cnt[0],
                              &workspace.buffer_crosstab[0], &workspace.buffer_cat_sorted[0],
                              (bool)(input_data.has_NA[col + input_data.ncols_numeric]), model_params.min_size_categ,
                              &(workspace.this_gain), &workspace.buffer_subset_categ[0]);

            /* If it was forcibly binarized, need to calculate the gain on the original categories to make it comparable */
            if (
                    !isinf(workspace.this_gain) &&
                    (
                        (!workspace.target_col_is_ord && input_data.ncat[workspace.target_col_num] > 2) ||
                        (workspace.target_col_is_ord && input_data.ncat_ord[workspace.target_col_num - input_data.ncols_categ] > 2)
                    )
                )
            {
                divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + col * input_data.nrows,
                                    workspace.st, workspace.end, &workspace.buffer_subset_categ[0], input_data.ncat[col],
                                    (bool)input_data.has_NA[col + input_data.ncols_numeric],
                                    &(workspace.this_split_NA), &(workspace.this_split_ix) );
                workspace.this_gain = categ_gain_from_split(&workspace.ix_arr[0], workspace.untransf_target_col, workspace.st,
                                                            workspace.this_split_NA, workspace.this_split_ix, workspace.end,
                                                            workspace.ncat_this, &workspace.buffer_cat_cnt[0], workspace.base_info_orig);
            }

        } else {

            if (model_params.cat_bruteforce_subset && input_data.ncat[col] > 2) {
                split_categx_categy_subset(&workspace.ix_arr[0], workspace.st, workspace.end,
                                           input_data.categorical_data + col * input_data.nrows, workspace.target_categ_col,
                                           input_data.ncat[col], workspace.ncat_this, workspace.base_info_orig,
                                           &workspace.buffer_cat_sorted[0], &workspace.buffer_crosstab[0], &workspace.buffer_cat_cnt[0],
                                           (bool)(input_data.has_NA[col + input_data.ncols_numeric]), model_params.min_size_categ,
                                           &(workspace.this_gain), &workspace.buffer_subset_categ[0]);
            } else {
                split_categx_categy_separate(&workspace.ix_arr[0], workspace.st, workspace.end,
                                             input_data.categorical_data + col * input_data.nrows, workspace.target_categ_col,
                                             input_data.ncat[col], workspace.ncat_this, workspace.base_info_orig,
                                             &workspace.buffer_cat_cnt[0], &workspace.buffer_crosstab[0],
                                             (bool)(input_data.has_NA[col + input_data.ncols_numeric]),
                                             model_params.min_size_categ, &(workspace.this_gain));
            }

        }

        if (model_params.gain_as_pct) workspace.this_gain /= workspace.base_info_orig;
        if (workspace.this_gain >= model_params.min_gain) {
            
            /* NA branch */
            workspace.this_split_NA = move_NAs_to_front(&workspace.ix_arr[0], input_data.categorical_data + col * input_data.nrows, workspace.st, workspace.end);
            if ((workspace.this_split_NA - workspace.st) > model_params.min_size_categ) {

                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Categorical, col, IsNa, (signed char*)NULL, (int)0, true);
                workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                              &workspace.ix_arr[0], workspace.st, workspace.this_split_NA - 1,
                                                              workspace.ncat_this, model_params.categ_from_maj,
                                                              &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                              &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                              workspace.prop_small_this, workspace.prior_prob,
                                                              &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                              &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                if (workspace.drop_cluster) {
                    workspace.clusters->pop_back();
                    (*workspace.tree)[tree_from].clusters.pop_back();
                }

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, IsNa, (signed char*)NULL, 0);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.end = workspace.this_split_NA - 1;
                    recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, true);
                    restore_recursion_state(workspace, *state_backup);
                }

            }

            if (!model_params.cat_bruteforce_subset && !workspace.col_is_bin && input_data.ncat[col] > 2) {

                /* sort by the splitting variable and iterate over to determine the split points */
                workspace.temp_ptr_x = input_data.categorical_data + col * input_data.nrows;
                std::sort(&workspace.ix_arr[0] + workspace.this_split_NA, &workspace.ix_arr[0] + workspace.end + 1,
                          [&workspace](const size_t a, const size_t b){return workspace.temp_ptr_x[a] < workspace.temp_ptr_x[b];});
                workspace.this_split_ix = workspace.this_split_NA;

                /* TODO: should instead use std::lower_bound to calculate the start and end indices of each category */
                for (size_t row = workspace.this_split_NA + 1; row <= workspace.end; row++) {

                    /* if the next observation is in a different category, then the split ends here */
                    if (workspace.temp_ptr_x[workspace.ix_arr[row]] != workspace.temp_ptr_x[workspace.ix_arr[row-1]]) {

                        if ((row - workspace.this_split_ix) >= model_params.min_size_categ) {

                            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                            workspace.clusters->emplace_back(col, workspace.temp_ptr_x[workspace.ix_arr[row-1]], input_data.ncat[col], is_NA_branch);
                            workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                                          &workspace.ix_arr[0], workspace.this_split_ix, row - 1,
                                                                          workspace.ncat_this, model_params.categ_from_maj,
                                                                          &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                                          &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                                          workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                                          model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                                          workspace.prop_small_this, workspace.prior_prob,
                                                                          &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                                          &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                            if (workspace.drop_cluster) {
                                workspace.clusters->pop_back();
                                (*workspace.tree)[tree_from].clusters.pop_back();
                            }
                            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                                workspace.tree->emplace_back(tree_from, col, workspace.temp_ptr_x[workspace.ix_arr[workspace.this_split_ix]]);
                                backup_recursion_state(workspace, *state_backup);
                                workspace.st = workspace.this_split_ix;
                                workspace.end = row - 1;
                                recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                                restore_recursion_state(workspace, *state_backup);
                            }
                        }
                        workspace.this_split_ix = row;
                    }
                }
                /* last category is given by the end indices */
                if ((workspace.end - workspace.this_split_ix) > model_params.min_size_categ) {
                    (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                    workspace.clusters->emplace_back(col, workspace.temp_ptr_x[workspace.ix_arr[workspace.end]], input_data.ncat[col], is_NA_branch);
                    workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                                  &workspace.ix_arr[0], workspace.this_split_ix, workspace.end,
                                                                  workspace.ncat_this, model_params.categ_from_maj,
                                                                  &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                                  &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                                  workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                                  model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                                  workspace.prop_small_this, workspace.prior_prob,
                                                                  &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                                  &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                    workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                    if (workspace.drop_cluster) {
                        workspace.clusters->pop_back();
                        (*workspace.tree)[tree_from].clusters.pop_back();
                    }
                    if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                        (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                        workspace.tree->emplace_back(tree_from, col, workspace.temp_ptr_x[workspace.ix_arr[workspace.end]]);
                        backup_recursion_state(workspace, *state_backup);
                        workspace.st = workspace.this_split_ix;
                        recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                        restore_recursion_state(workspace, *state_backup);
                    }

                }

                if (workspace.this_gain > workspace.best_gain) {
                    workspace.best_gain = workspace.this_gain;
                    workspace.column_type_best = Categorical;
                    workspace.col_best = col;
                }


            } else {

                /* split by subsets of categories */

                if (input_data.ncat[col] == 2) {

                    workspace.buffer_subset_categ[0] = 1;
                    workspace.buffer_subset_categ[1] = 0;
                    divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + col * input_data.nrows, workspace.this_split_NA, workspace.end,
                                        (int)0, false, &(workspace.this_split_NA), &(workspace.this_split_ix));
                    if (
                        (workspace.end - workspace.this_split_ix) < model_params.min_size_categ ||
                        (workspace.this_split_ix - workspace.this_split_NA) < model_params.min_size_categ
                    ) continue;

                } else {

                    divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + col * input_data.nrows, workspace.this_split_NA, workspace.end,
                                        &workspace.buffer_subset_categ[0], input_data.ncat[col], false,
                                        &(workspace.this_split_NA), &(workspace.this_split_ix));
                }

                /* left branch */
                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Categorical, col, InSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col], is_NA_branch);
                workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                              &workspace.ix_arr[0], workspace.this_split_NA, workspace.this_split_ix - 1,
                                                              workspace.ncat_this, model_params.categ_from_maj,
                                                              &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                              &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                              workspace.prop_small_this, workspace.prior_prob,
                                                              &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                              &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                if (workspace.drop_cluster) {
                    workspace.clusters->pop_back();
                    (*workspace.tree)[tree_from].clusters.pop_back();
                }

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, InSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col]);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.st = workspace.this_split_NA;
                    workspace.end = workspace.this_split_ix - 1;
                    recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                    restore_recursion_state(workspace, *state_backup);
                }

                /* right branch */
                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Categorical, col, NotInSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col], is_NA_branch);
                workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                              &workspace.ix_arr[0], workspace.this_split_ix, workspace.end,
                                                              workspace.ncat_this, model_params.categ_from_maj,
                                                              &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                              &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                              workspace.prop_small_this, workspace.prior_prob,
                                                              &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                              &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                if (workspace.drop_cluster) {
                    workspace.clusters->pop_back();
                    (*workspace.tree)[tree_from].clusters.pop_back();
                }

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, NotInSubset, &workspace.buffer_subset_categ[0], input_data.ncat[col]);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.st = workspace.this_split_ix;
                    recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                    restore_recursion_state(workspace, *state_backup);
                }

                if (workspace.this_gain > workspace.best_gain) {
                    workspace.best_gain = workspace.this_gain;
                    workspace.column_type_best = Categorical;
                    workspace.col_best = col;
                    memcpy(&workspace.buffer_subset_categ_best[0], &workspace.buffer_subset_categ[0], input_data.ncat[col] * sizeof(signed char));
                }

            }

        }

    }


    /* then ordinal */
    for (size_t col = 0; col < input_data.ncols_ord; col++) {

        if (curr_depth == 0 && workspace.col_is_bin && workspace.ncat_this > 2 && workspace.already_split_main) break;
        if (input_data.skip_col[col + input_data.ncols_numeric + input_data.ncols_categ]) continue;
        if (workspace.target_col_is_ord && col == (workspace.target_col_num - input_data.ncols_categ)) continue;

        split_ordx_categy(&workspace.ix_arr[0], workspace.st, workspace.end,
                          input_data.ordinal_data + col * input_data.nrows, workspace.untransf_target_col,
                          input_data.ncat_ord[col], workspace.ncat_this,
                          workspace.base_info_orig, &workspace.buffer_cat_cnt[0], &workspace.buffer_crosstab[0], &workspace.buffer_cat_sorted[0],
                          (bool)(input_data.has_NA[col + input_data.ncols_numeric + input_data.ncols_categ]),
                          model_params.min_size_categ, &(workspace.this_gain), &(workspace.this_split_lev));
        if (model_params.gain_as_pct) workspace.this_gain /= workspace.base_info_orig;

        if (workspace.this_gain >= model_params.min_gain) {

            divide_subset_split(&workspace.ix_arr[0], input_data.ordinal_data + col * input_data.nrows, workspace.st, workspace.end,
                                workspace.this_split_lev, (bool)(workspace.buffer_cat_cnt[ input_data.ncat_ord[col] ] > 0),
                                &(workspace.this_split_NA), &(workspace.this_split_ix) );

            /* NA branch */
            if ((workspace.this_split_NA - workspace.st) > model_params.min_size_categ) {

                (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
                workspace.clusters->emplace_back(Ordinal, col, IsNa, (int)0, true);
                workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                              &workspace.ix_arr[0], workspace.st, workspace.this_split_NA - 1,
                                                              workspace.ncat_this, model_params.categ_from_maj,
                                                              &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                              &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                              workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                              model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                              workspace.prop_small_this, workspace.prior_prob,
                                                              &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                              &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
                workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
                if (workspace.drop_cluster) {
                    workspace.clusters->pop_back();
                    (*workspace.tree)[tree_from].clusters.pop_back();
                }

                if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                    (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                    workspace.tree->emplace_back(tree_from, col, (int)-1, IsNa);
                    backup_recursion_state(workspace, *state_backup);
                    workspace.end = workspace.this_split_NA - 1;
                    recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, true);
                    restore_recursion_state(workspace, *state_backup);
                }

            }

            /* left branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Ordinal, col, LessOrEqual, workspace.this_split_lev, is_NA_branch);
            workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                          &workspace.ix_arr[0], workspace.this_split_NA, workspace.this_split_ix - 1,
                                                          workspace.ncat_this, model_params.categ_from_maj,
                                                          &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                          &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                          workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                          model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                          workspace.prop_small_this, workspace.prior_prob,
                                                          &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                          &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
            if (workspace.drop_cluster) {
                workspace.clusters->pop_back();
                (*workspace.tree)[tree_from].clusters.pop_back();
            }

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_lev, LessOrEqual);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_NA;
                workspace.end = workspace.this_split_ix - 1;
                recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }

            /* right branch */
            (*workspace.tree)[tree_from].clusters.push_back(workspace.clusters->size());
            workspace.clusters->emplace_back(Ordinal, col, Greater, workspace.this_split_lev, is_NA_branch);
            workspace.has_outliers = define_categ_cluster(workspace.untransf_target_col,
                                                          &workspace.ix_arr[0], workspace.this_split_ix, workspace.end,
                                                          workspace.ncat_this, model_params.categ_from_maj,
                                                          &workspace.outlier_scores[0], &workspace.outlier_clusters[0], &workspace.outlier_trees[0],
                                                          &workspace.outlier_depth[0], workspace.clusters->back(), *(workspace.clusters),
                                                          workspace.clusters->size() - 1, tree_from, curr_depth + 1,
                                                          model_params.max_perc_outliers, model_params.z_norm, model_params.z_outlier,
                                                          workspace.prop_small_this, workspace.prior_prob,
                                                          &workspace.buffer_cat_cnt[0], &workspace.buffer_cat_sum[0],
                                                          &workspace.buffer_crosstab[0], &workspace.buffer_subset_outlier[0], &(workspace.drop_cluster));
            workspace.lev_has_outliers = workspace.has_outliers? true : workspace.lev_has_outliers;
            if (workspace.drop_cluster) {
                workspace.clusters->pop_back();
                (*workspace.tree)[tree_from].clusters.pop_back();
            }

            if (model_params.follow_all && ((curr_depth + 1) < model_params.max_depth)) {
                (*workspace.tree)[tree_from].all_branches.push_back(workspace.tree->size());
                workspace.tree->emplace_back(tree_from, col, workspace.this_split_lev, Greater);
                backup_recursion_state(workspace, *state_backup);
                workspace.st = workspace.this_split_ix;
                recursive_split_categ(workspace, input_data, model_params, curr_depth + 1, is_NA_branch);
                restore_recursion_state(workspace, *state_backup);
            }


            if (workspace.this_gain > workspace.best_gain) {
                workspace.best_gain = workspace.this_gain;
                workspace.column_type_best = Ordinal;
                workspace.col_best = col;
                workspace.split_lev_best = workspace.this_split_lev;
            }

        }

    }


    /* avoid unnecessary memory usage or repeats */
    workspace.col_has_outliers = workspace.lev_has_outliers? true : workspace.col_has_outliers;
    (*workspace.tree)[tree_from].clusters.shrink_to_fit();
    if ((*workspace.tree)[tree_from].all_branches.size() > 0) (*workspace.tree)[tree_from].all_branches.shrink_to_fit();
    if (curr_depth == 0 && workspace.col_is_bin && workspace.ncat_this > 2 && !workspace.already_split_main)
        workspace.already_split_main = true;


    /* if there is a non-insignificant gain, continue splitting from the branches of the best column */
    if (workspace.best_gain >= model_params.min_gain && !model_params.follow_all) {
        
        curr_depth++;
        if (curr_depth >= model_params.max_depth) goto drop_if_not_needed;

        /* discard outliers if any */
        if (workspace.lev_has_outliers)
            workspace.st = move_outliers_to_front(&workspace.ix_arr[0], &workspace.outlier_scores[0], workspace.st, workspace.end);

        /* assign rows to their corresponding branch */
        switch(workspace.column_type_best) {
            case Numeric:
            {
                divide_subset_split(&workspace.ix_arr[0], input_data.numeric_data + workspace.col_best * input_data.nrows,
                                    workspace.st, workspace.end, workspace.split_point_best,
                                    (bool)(input_data.has_NA[workspace.col_best]),
                                    &(workspace.this_split_NA), &(workspace.this_split_ix) );
                spl1 = LessOrEqual; spl2 = Greater;
                set_tree_as_numeric(workspace.tree->back(), workspace.split_point_best, workspace.col_best);
                break;
            }

            case Ordinal:
            {
                divide_subset_split(&workspace.ix_arr[0], input_data.ordinal_data + workspace.col_best * input_data.nrows,
                                    workspace.st, workspace.end, workspace.split_lev_best,
                                    (bool)(input_data.has_NA[workspace.col_best + input_data.ncols_numeric + input_data.ncols_categ]),
                                    &(workspace.this_split_NA), &(workspace.this_split_ix) );
                spl1 = LessOrEqual; spl2 = Greater;
                set_tree_as_ordinal(workspace.tree->back(), workspace.split_lev_best, workspace.col_best);
                break;
            }

            case Categorical:
            {

                if (input_data.ncat[workspace.col_best] == 2) {

                    divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + workspace.col_best * input_data.nrows,
                                        workspace.st, workspace.end, (int)0,
                                        (bool)(input_data.has_NA[workspace.col_best + input_data.ncols_numeric]),
                                        &(workspace.this_split_NA), &(workspace.this_split_ix) );
                    spl1 = InSubset; spl2 = NotInSubset;
                    set_tree_as_categorical(workspace.tree->back(), workspace.col_best);

                } else if (workspace.col_is_bin || model_params.cat_bruteforce_subset) {

                    divide_subset_split(&workspace.ix_arr[0], input_data.categorical_data + workspace.col_best * input_data.nrows,
                                        workspace.st, workspace.end, &workspace.buffer_subset_categ_best[0], input_data.ncat[workspace.col_best],
                                        (bool)(input_data.has_NA[workspace.col_best + input_data.ncols_numeric]),
                                        &(workspace.this_split_NA), &(workspace.this_split_ix) );
                    spl1 = InSubset; spl2 = NotInSubset;
                    set_tree_as_categorical(workspace.tree->back(), input_data.ncat[workspace.col_best],
                                            &workspace.buffer_subset_categ_best[0], workspace.col_best);

                } else {
                    spl1 = SingleCateg;
                    workspace.temp_ptr_x = input_data.categorical_data + workspace.col_best * input_data.nrows;
                    std::sort(&workspace.ix_arr[0] + workspace.st, &workspace.ix_arr[0] + workspace.end + 1,
                              [&workspace](const size_t a, const size_t b){return workspace.temp_ptr_x[a] < workspace.temp_ptr_x[b];});
                    set_tree_as_categorical(workspace.tree->back(), workspace.col_best, input_data.ncat[workspace.col_best]);

                    for (size_t row = workspace.st; row <= workspace.end; row++) {
                        if (workspace.temp_ptr_x[ workspace.ix_arr[row] ] >= 0) {
                            workspace.this_split_NA = row;
                            break;
                        }
                    }
                }
                break;
            }

            default:
            {
                unexpected_error();
            }
        }


        ix1 = workspace.this_split_NA;
        ix2 = workspace.this_split_ix;
        ix3 = workspace.end;

        /* NA branch */
        if (workspace.st > workspace.this_split_NA &&
            (workspace.st - workspace.this_split_NA) >= 2 * model_params.min_size_categ) {

            workspace.end = ix1 - 1;
            (*workspace.tree)[tree_from].tree_NA = workspace.tree->size();
            workspace.tree->emplace_back(tree_from, IsNa);
            recursive_split_categ(workspace, input_data, model_params, curr_depth, true);
        }

        if (spl1 == SingleCateg) {

            /* TODO: this should be done instead in a loop per category looking for the start and end positions
               in ix_arr of each category using std::lower_bound */

            /* TODO: it's not necessary to backup everything like when using 'follow_all', only need 'best_col' and 'temp_ptr_x' */
            state_backup = std::unique_ptr<RecursionState>(new RecursionState);
            for (int cat = 1; cat < input_data.ncat[workspace.col_best]; cat++) {

                /*  TODO: this is inefficient when some categories are not present, should instead at first do a pass over 'ix_arr'
                    to calculate the start and end indices of each category, then loop over that array instead */
                for (size_t row = ix1 + 1; row < ix3; row++) {
                    if (workspace.temp_ptr_x[ workspace.ix_arr[row] ] == cat) {
                        if ((row - ix1) >= 2 * model_params.min_size_categ) {
                            (*workspace.tree)[tree_from].binary_branches[cat-1] = workspace.tree->size();
                            workspace.tree->emplace_back(tree_from, spl1);
                            backup_recursion_state(workspace, *state_backup);
                            workspace.st = ix1;
                            workspace.end = row - 1;
                            recursive_split_categ(workspace, input_data, model_params, curr_depth, is_NA_branch);
                            restore_recursion_state(workspace, *state_backup);
                        }
                        ix1 = row;
                        break;
                    }
                    else if (workspace.temp_ptr_x[ workspace.ix_arr[row] ] > cat) {
                        ix1 = row;
                        break;
                    }
                }

            }
            /* last category is given by the end index */
            if ((ix3 - ix1) >= 2 * model_params.min_size_categ) {
                (*workspace.tree)[tree_from].binary_branches[input_data.ncat[workspace.col_best]-1] = workspace.tree->size();
                workspace.tree->emplace_back(tree_from, spl1);
                workspace.st = ix1;
                workspace.end = ix3;
                recursive_split_categ(workspace, input_data, model_params, curr_depth, is_NA_branch);
            } else {
                (*workspace.tree)[tree_from].binary_branches.push_back(0);
            }

        } else {
            /* numeric, ordinal, and subset split */

            /* left branch */
            if ((ix2 - ix1) >= 2 * model_params.min_size_categ) {
                workspace.st = ix1;
                workspace.end = ix2 - 1;
                (*workspace.tree)[tree_from].tree_left = workspace.tree->size();
                workspace.tree->emplace_back(tree_from, spl1);
                recursive_split_categ(workspace, input_data, model_params, curr_depth, is_NA_branch);
            }

            /* right branch */
            if ((ix3 - ix2) > 2 * model_params.min_size_categ) {
                workspace.st = ix2;
                workspace.end = ix3;
                (*workspace.tree)[tree_from].tree_right = workspace.tree->size();
                workspace.tree->emplace_back(tree_from, spl2);
                recursive_split_categ(workspace, input_data, model_params, curr_depth, is_NA_branch);
            }

        }


    }


    /* if tree has no clusters and no subtrees, disconnect it from parent and then drop */
    drop_if_not_needed:
        if (check_tree_is_not_needed((*workspace.tree)[tree_from])) {

            if (tree_from == 0) {
                workspace.tree->clear();
            } else if ((*workspace.tree)[(*workspace.tree)[tree_from].parent].all_branches.size() > 0) {
                (*workspace.tree)[(*workspace.tree)[tree_from].parent].all_branches.pop_back();
                workspace.tree->pop_back();
            } else {
                switch((*workspace.tree)[tree_from].parent_branch) {

                    case IsNa:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_NA = 0;
                        workspace.tree->pop_back();
                        break;
                    }

                    case LessOrEqual:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_left = 0;
                        workspace.tree->pop_back();
                        break;
                    }

                    case Greater:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_right = 0;
                        workspace.tree->pop_back();
                        break;
                    }

                    case InSubset:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_left = 0;
                        workspace.tree->pop_back();
                        break;
                    }

                    case NotInSubset:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].tree_right = 0;
                        workspace.tree->pop_back();
                        break;
                    }

                    case SingleCateg:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].binary_branches.back() = 0;
                        workspace.tree->pop_back();
                        break;
                    }

                    case SubTrees:
                    {
                        (*workspace.tree)[(*workspace.tree)[tree_from].parent].binary_branches.pop_back();
                        workspace.tree->pop_back();
                        break;
                    }

                    default:
                    {
                        unexpected_error();
                    }
                }
            }
        }

}
