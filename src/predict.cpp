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


/*    Check if new data would be outliers according to previously-fit model
*    
*    Note that the new data must follow the exact same column order, and must also be passsed as arrays
*    order by columns (like Fortran arrays, not like C arrays). For data with < 10,000 rows, it's usually
*    faster to run it single-threaded. The outputs are pased in arrays within the 'ModelOutputs' struct,
*    just like when fitting the model. Outpus for rows from previous calls to this function or to the model-fitting
*    function will be overwriten.
*    
*    Parameters:
*    - numeric_data[n * m1] (in)
*        Array with numerical columns in the data. Must be ordered by columns like Fortran arrays.
*        Missing values should be encoded as NaN. Infinite values in most sections are treated as NaN too.
*        If there are no numerical columns, pass NULL.
*    - categorical_data[n * m2] (in)
*        Array with categorical columns in the data. Must be ordered by columns like Fortran arrays.
*        Negative numbers will be interpreted as missing values. Numeration must start at zero and be
*        contiguous (i.e. if there's category 2, must also have category 1).
*        If there are no categorical columns, pass NULL.
*    - ordinal_data[n * m3] (in)
*        Array with ordinal categorical columns in the data. Must be ordered by columns like Fortran arrays.
*        Same rules as for categorical data. Note that the order will only be taken into consideration when
*        producing splits by these columns, but outliers are still detected in the same way as for categoricals.
*        If there are no ordinal columns, pass NULL.
*    - nrows (in)
*        Number of rows (n) in the arrays passed above.
*    - nthreads (in)
*        Number of parallel threads to use.
*    - model_outputs (in, out)
*        Struct containing the data from the fitted model necessary to make new predictions,
*        and buffer vectors where to store the details of the potential outliers found.
*    
*    Returns:
*        Whether there were any outliers identified in the data passed here. Their details will be inside the
*        'ModelOutputs' struct.
*/
bool find_new_outliers(double *restrict numeric_data,
                       int    *restrict categorical_data,
                       int    *restrict ordinal_data,
                       size_t nrows, int nthreads, ModelOutputs &model_outputs)
{
    size_t tot_cols = model_outputs.ncols_numeric + model_outputs.ncols_categ + model_outputs.ncols_ord;
    double num_val_this;
    int cat_val_this;
    bool col_is_num;

    bool found_outliers = false;
    if (nrows < (size_t)nthreads)
        nthreads = (int) nrows;
    #if defined(_OPENMP)
        std::vector<char> outliers_thread(nthreads, false);
    #endif

    /* reset the output data structures */
    allocate_row_outputs(model_outputs, nrows, model_outputs.max_depth);

    /* put data into a struct and pass it by reference */
    PredictionData prediction_data = {numeric_data, categorical_data, ordinal_data, nrows};

    /*    Note: if parallelizing by columns instead of by rows, need to switch on the `#pragma omp critical`
        in the block that assigns the cluster to an observation */

    /* see if any value is an outlier */
    // #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(model_outputs, outliers_thread, nrows, tot_cols, prediction_data) private(col_is_num, num_val_this, cat_val_this)
    for (size_t_for col = 0; col < tot_cols; col++) {
        
        if (model_outputs.all_trees[col].size() == 0 || model_outputs.all_clusters[col].size() == 0) continue;
        col_is_num = col < model_outputs.ncols_numeric;

        /* Note: earlier versions of OpenMP (like v2 released in 2000 and still used by MSVC in 2019) don't support max reduction, hence this code */
        #pragma omp parallel for schedule(dynamic) num_threads(nthreads) shared(model_outputs, outliers_thread, nrows, prediction_data) \
                    firstprivate(col_is_num, col) private(num_val_this, cat_val_this)
        for (size_t_for row = 0; row < nrows; row++) {
            
            /* first make a pre-check that the value could be flagged as outlier in some cluster */
            if (col < model_outputs.ncols_numeric) {

                num_val_this = prediction_data.numeric_data[row + col * nrows];
                if (is_na_or_inf(num_val_this)) continue;
                if ((num_val_this < model_outputs.max_outlier_any_cl[col]) && (num_val_this > model_outputs.min_outlier_any_cl[col])) continue;

            } else if (col < (model_outputs.ncols_numeric + model_outputs.ncols_categ)) {
                
                cat_val_this = prediction_data.categorical_data[row + (col - model_outputs.ncols_numeric) * nrows];
                if (cat_val_this < 0) continue;
                if (cat_val_this >= model_outputs.ncat[col - model_outputs.ncols_numeric]) continue;
                if (!model_outputs.cat_outlier_any_cl[col - model_outputs.ncols_numeric][cat_val_this]) continue;

            } else {
                
                cat_val_this = prediction_data.ordinal_data[row + (col - model_outputs.ncols_numeric - model_outputs.ncols_categ) * nrows];
                if (cat_val_this < 0) continue;
                if (cat_val_this >= model_outputs.ncat_ord[col - model_outputs.ncols_numeric - model_outputs.ncols_categ]) continue;
                if (!model_outputs.cat_outlier_any_cl[col - model_outputs.ncols_numeric][cat_val_this]) continue;

            }

            #ifdef _OPENMP
                outliers_thread[omp_get_thread_num()] = follow_tree(model_outputs, prediction_data, 0, 0, row, col, col_is_num, num_val_this, cat_val_this)?
                                                        true : outliers_thread[omp_get_thread_num()];
            #else 
                found_outliers = std::max(found_outliers, follow_tree(model_outputs, prediction_data, 0, 0, row, col, col_is_num, num_val_this, cat_val_this));
            #endif

        }
    }

    #if defined(_OPENMP)
        for (size_t tid = 0; tid < outliers_thread.size(); tid++) {
            if (outliers_thread[tid] != 0) found_outliers = true;
        }
    #endif

    if (found_outliers)
        calc_min_decimals_to_print(model_outputs, prediction_data.numeric_data, nthreads);

    return found_outliers;
}

bool follow_tree(ModelOutputs &model_outputs, PredictionData &prediction_data, size_t curr_tree, size_t curr_depth,
                 size_t_for row, size_t_for col, bool col_is_num, double num_val_this, int cat_val_this)
{
    bool found_outliers = check_is_outlier_in_tree(model_outputs.all_trees[col][curr_tree].clusters,
                                                   curr_depth, curr_tree, model_outputs, prediction_data,
                                                   row, col, col_is_num, num_val_this, cat_val_this);

    /*    if there's outliers at this level and it's in a non-NA branch, there's no point in continuing
        further down the tree as deeper clusters are not preferred */
    if (
        found_outliers &&
        !model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
    ) return true;

    /* check if the tree is a dead-end */
    if (
        model_outputs.all_trees[col][curr_tree].tree_NA == 0 &&
        model_outputs.all_trees[col][curr_tree].tree_left == 0 &&
        model_outputs.all_trees[col][curr_tree].tree_right == 0 &&
        (
            model_outputs.all_trees[col][curr_tree].binary_branches.size() == 0 ||
            *std::max_element(
                                model_outputs.all_trees[col][curr_tree].binary_branches.begin(),
                                model_outputs.all_trees[col][curr_tree].binary_branches.end()
                                ) == 0
        ) &&
        (
            model_outputs.all_trees[col][curr_tree].all_branches.size() == 0 ||
            *std::max_element(
                                model_outputs.all_trees[col][curr_tree].all_branches.begin(),
                                model_outputs.all_trees[col][curr_tree].all_branches.end()
                                ) == 0
            )
        ) return false;

    /* try to follow trees according to the value of the columns they look at */
    double num_val_other;
    int cat_val_other;

    /* if using 'follow_all', follow on all possible branches */
    if (model_outputs.all_trees[col][curr_tree].all_branches.size() > 0) {

        for (size_t br : model_outputs.all_trees[col][curr_tree].all_branches) {
            if (br > 0) {
                switch(model_outputs.all_trees[col][br].column_type) {

                    case Numeric:
                    {
                        num_val_other = prediction_data.numeric_data[row + model_outputs.all_trees[col][br].col_num * prediction_data.nrows];
                        switch (model_outputs.all_trees[col][br].split_this_branch) {
                            case IsNa:
                            {
                                if (isnan(num_val_other))
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case LessOrEqual:
                            {
                                if (!isnan(num_val_other) && num_val_other <= model_outputs.all_trees[col][br].split_point)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case Greater:
                            {
                                if (!isnan(num_val_other) && num_val_other > model_outputs.all_trees[col][br].split_point)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            default:
                            {
                                assert(0);
                            }
                        }
                        break;
                    }

                    case Categorical:
                    {
                        cat_val_other = prediction_data.categorical_data[row + model_outputs.all_trees[col][br].col_num * prediction_data.nrows];
                        if (cat_val_other >= model_outputs.ncat[model_outputs.all_trees[col][br].col_num]) continue;
                        switch (model_outputs.all_trees[col][br].split_this_branch) {
                            case IsNa:
                            {
                                if (cat_val_other < 0)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case InSubset:
                            {
                                if (cat_val_other >= 0 && model_outputs.all_trees[col][br].split_subset[cat_val_other] == 1)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case NotInSubset:
                            {
                                if (cat_val_other >= 0 && model_outputs.all_trees[col][br].split_subset[cat_val_other] == 0)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case Equal:
                            {
                                if (cat_val_other == model_outputs.all_trees[col][br].split_lev)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case NotEqual:
                            {
                                if (cat_val_other >= 0 && cat_val_other != model_outputs.all_trees[col][br].split_lev)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            default:
                            {
                                assert(0);
                            }
                        }
                        break;
                    }

                    case Ordinal:
                    {
                        cat_val_other = prediction_data.ordinal_data[row + model_outputs.all_trees[col][br].col_num * prediction_data.nrows];
                        if (cat_val_other >= model_outputs.ncat_ord[model_outputs.all_trees[col][br].col_num]) continue;
                        switch (model_outputs.all_trees[col][br].split_this_branch) {
                            case IsNa:
                            {
                                if (cat_val_other < 0)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case LessOrEqual:
                            {
                                if (cat_val_other >= 0 && cat_val_other <= model_outputs.all_trees[col][br].split_lev)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case Greater:
                            {
                                if (cat_val_other > model_outputs.all_trees[col][br].split_lev)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case Equal:
                            {
                                if (cat_val_other == model_outputs.all_trees[col][br].split_lev)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            case NotEqual:
                            {
                                if (cat_val_other >= 0 && cat_val_other != model_outputs.all_trees[col][br].split_lev)
                                    found_outliers = follow_tree(model_outputs, prediction_data, br, curr_depth + 1,
                                                                 row, col, col_is_num, num_val_this, cat_val_this)?
                                                     true : found_outliers;
                                break;
                            }

                            default:
                            {
                                assert(0);
                            }
                        }
                        break;
                    }

                    default: {}
                }
            }
        }
        return found_outliers;
    }

    /* regular case (not using 'follow_all') - follow the corresponding branch */
    switch(model_outputs.all_trees[col][curr_tree].column_type) {

        case NoType:
        {
            if (model_outputs.all_trees[col][curr_tree].binary_branches.size() > 0) {
                for (size_t tree_follow : model_outputs.all_trees[col][curr_tree].binary_branches) {
                        if (tree_follow > 0)
                            found_outliers = follow_tree(model_outputs, prediction_data, tree_follow, curr_depth,
                                                         row, col, col_is_num, num_val_this, cat_val_this)?
                                             true : found_outliers;
                    }
                    return found_outliers;
            }
            break;
        }

        case Numeric:
        {
            num_val_other = prediction_data.numeric_data[row + model_outputs.all_trees[col][curr_tree].col_num * prediction_data.nrows];
            if (isnan(num_val_other)) {

                if (model_outputs.all_trees[col][curr_tree].tree_NA > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_NA, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);

            } else if (num_val_other <= model_outputs.all_trees[col][curr_tree].split_point) {

                if (model_outputs.all_trees[col][curr_tree].tree_left > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_left, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);

            } else {

                if (model_outputs.all_trees[col][curr_tree].tree_right > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_right, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);
                
            }
            break;
        }

        case Categorical:
        {
            cat_val_other = prediction_data.categorical_data[row + model_outputs.all_trees[col][curr_tree].col_num * prediction_data.nrows];
            if (cat_val_other >= model_outputs.ncat[model_outputs.all_trees[col][curr_tree].col_num]) return false;
            if (cat_val_other < 0) {
                if (model_outputs.all_trees[col][curr_tree].tree_NA > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_NA, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);
                else return false;
            }


            if (model_outputs.all_trees[col][curr_tree].binary_branches.size() > 0) {

                if (curr_tree == 0 && model_outputs.all_trees[col][curr_tree].column_type == NoType) {
                    /* binarized branches in the main tree */
                    for (size_t tree_follow : model_outputs.all_trees[col][curr_tree].binary_branches) {

                        if (tree_follow > 0)
                            found_outliers = follow_tree(model_outputs, prediction_data, tree_follow, curr_depth,
                                                         row, col, col_is_num, num_val_this, cat_val_this)?
                                             true : found_outliers;
                    }
                    return found_outliers;

                } else {

                    /* single-category branch in a categorical-by-categorical split */
                    if (model_outputs.all_trees[col][curr_tree].binary_branches[cat_val_other] > 0) {
                        return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].binary_branches[cat_val_other], curr_depth + 1,
                                           row, col, col_is_num, num_val_this, cat_val_this);
                    }
                }

            }

            else if (model_outputs.all_trees[col][curr_tree].split_lev != INT_MAX) {

                if (model_outputs.all_trees[col][curr_tree].split_lev == cat_val_other) {

                    if (model_outputs.all_trees[col][curr_tree].tree_left > 0)
                        return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_left, curr_depth + 1,
                                           row, col, col_is_num, num_val_this, cat_val_this);

                } else {

                    if (model_outputs.all_trees[col][curr_tree].tree_right > 0)
                        return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_right, curr_depth + 1,
                                           row, col, col_is_num, num_val_this, cat_val_this);

                }

            }

            else {

                if (model_outputs.all_trees[col][curr_tree].split_subset[cat_val_other] == 1) {

                    if (model_outputs.all_trees[col][curr_tree].tree_left > 0)
                        return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_left, curr_depth + 1,
                                           row, col, col_is_num, num_val_this, cat_val_this);
                    
                } else if (model_outputs.all_trees[col][curr_tree].split_subset[cat_val_other] == 0) {

                    if (model_outputs.all_trees[col][curr_tree].tree_right > 0)
                        return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_right, curr_depth + 1,
                                           row, col, col_is_num, num_val_this, cat_val_this);

                }

            }
            break;
        }

        case Ordinal:
        {
            cat_val_other = prediction_data.ordinal_data[row + model_outputs.all_trees[col][curr_tree].col_num * prediction_data.nrows];
            if (cat_val_other >= model_outputs.ncat_ord[model_outputs.all_trees[col][curr_tree].col_num]) return false;
            if (cat_val_other < 0) {

                if (model_outputs.all_trees[col][curr_tree].tree_NA > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_NA, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);
            
            } else if (cat_val_other <= model_outputs.all_trees[col][curr_tree].split_lev) {

                if (model_outputs.all_trees[col][curr_tree].tree_left > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_left, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);

            } else {

                if (model_outputs.all_trees[col][curr_tree].tree_right > 0)
                    return follow_tree(model_outputs, prediction_data, model_outputs.all_trees[col][curr_tree].tree_right, curr_depth + 1,
                                       row, col, col_is_num, num_val_this, cat_val_this);
            }
            break;
        }

    }

    return false;
}

bool check_is_outlier_in_tree(std::vector<size_t> &clusters_in_tree, size_t curr_depth, size_t curr_tree,
                              ModelOutputs &model_outputs, PredictionData &prediction_data, size_t_for row, size_t_for col,
                              bool col_is_num, double num_val_this, int cat_val_this)
{


    bool tree_has_outliers = false;
    bool flag_this_cluster;
    double outlier_score;
    size_t cluster_size;
    size_t cluster_depth;
    double num_val_other;
    int cat_val_other;

    if (clusters_in_tree.size() > 0) {

        /* see if it would be an outlier under any of the clusters from this tree */
        for (const size_t cl : clusters_in_tree) {

            if (col_is_num) {
                if (
                    num_val_this > model_outputs.all_clusters[col][cl].lower_lim &&
                    num_val_this < model_outputs.all_clusters[col][cl].upper_lim
                ) continue;
            } else {
                if (model_outputs.all_clusters[col][cl].subset_common[cat_val_this] == 0)
                    continue;
            }

            /* if so, then check if it actually belongs into the cluster */
            flag_this_cluster = false;
            switch(model_outputs.all_clusters[col][cl].column_type) {

                case NoType:
                {
                    flag_this_cluster = true;
                    break;
                }

                case Numeric:
                {
                    num_val_other = prediction_data.numeric_data[row + model_outputs.all_clusters[col][cl].col_num * prediction_data.nrows];
                    switch(model_outputs.all_clusters[col][cl].split_type) {
                        case IsNa:
                        {
                            if (isnan(num_val_other)) flag_this_cluster = true;
                            break;
                        }

                        case LessOrEqual:
                        {
                            if (!isnan(num_val_other) && num_val_other <= model_outputs.all_clusters[col][cl].split_point) flag_this_cluster = true;
                            break;
                        }

                        case Greater:
                        {
                            if (!isnan(num_val_other) && num_val_other > model_outputs.all_clusters[col][cl].split_point) flag_this_cluster = true;
                            break;
                        }

                        default:
                        {
                            assert(0);
                        }
                    }
                    break;
                }

                case Categorical:
                {
                    cat_val_other = prediction_data.categorical_data[row + model_outputs.all_clusters[col][cl].col_num * prediction_data.nrows];
                    if (cat_val_other >= model_outputs.ncat[model_outputs.all_clusters[col][cl].col_num]) continue;
                    switch(model_outputs.all_clusters[col][cl].split_type) {
                        case IsNa:
                        {
                            if (cat_val_other < 0) flag_this_cluster = true;
                            break;
                        }

                        case InSubset:
                        {
                            if (cat_val_other >=0 && model_outputs.all_clusters[col][cl].split_subset[cat_val_other] == 1) flag_this_cluster = true;
                            break;
                        }

                        case NotInSubset:
                        {
                            if (cat_val_other >=0 && model_outputs.all_clusters[col][cl].split_subset[cat_val_other] == 0) flag_this_cluster = true;
                            break;
                        }

                        case Equal:
                        {
                            if (cat_val_other == model_outputs.all_clusters[col][cl].split_lev) flag_this_cluster = true;
                            break;
                        }

                        case NotEqual:
                        {
                            if (cat_val_other >=0 && cat_val_other != model_outputs.all_clusters[col][cl].split_lev) flag_this_cluster = true;
                            break;
                        }

                        default:
                        {
                            assert(0);
                        }

                        /* Note: type 'SingleCateg' is only used temporarily, later gets converted to 'Equal' */
                    }
                    break;
                }

                case Ordinal:
                {
                    cat_val_other = prediction_data.ordinal_data[row + model_outputs.all_clusters[col][cl].col_num * prediction_data.nrows];
                    if (cat_val_other >= model_outputs.ncat_ord[model_outputs.all_clusters[col][cl].col_num]) continue;
                    switch(model_outputs.all_clusters[col][cl].split_type) {
                        case IsNa:
                        {
                            if (cat_val_other < 0) flag_this_cluster = true;
                            break;
                        }

                        case LessOrEqual:
                        {
                            if (cat_val_other >=0 && cat_val_other <= model_outputs.all_clusters[col][cl].split_lev) flag_this_cluster = true;
                            break;
                        }

                        case Greater:
                        {
                            if (cat_val_other >=0 && cat_val_other > model_outputs.all_clusters[col][cl].split_lev) flag_this_cluster = true;
                            break;
                        }

                        case Equal:
                        {
                            if (cat_val_other == model_outputs.all_clusters[col][cl].split_lev) flag_this_cluster = true;
                            break;
                        }

                        case NotEqual:
                        {
                            if (cat_val_other >=0 && cat_val_other != model_outputs.all_clusters[col][cl].split_lev) flag_this_cluster = true;
                            break;
                        }

                        default:
                        {
                            assert(0);
                        }
                    }
                    break;
                }
            }
            if (flag_this_cluster) {

                tree_has_outliers = true;
                cluster_size = model_outputs.all_clusters[col][cl].cluster_size;
                cluster_depth = curr_depth + ((model_outputs.all_clusters[col][cl].column_type == NoType)? 0 : 1);
                if (col_is_num) {
                    outlier_score = chebyshyov_bound(z_score(
                                                            (model_outputs.col_transf[col] == NoTransf)? num_val_this :
                                                                (model_outputs.col_transf[col] == Log)?
                                                                    log(num_val_this - model_outputs.transf_offset[col]) :
                                                                    exp( (num_val_this - model_outputs.transf_offset[col]) / model_outputs.sd_div[col] ),
                                                            model_outputs.all_clusters[col][cl].cluster_mean,
                                                            model_outputs.all_clusters[col][cl].cluster_sd
                                                            )
                    );
                    if (is_na_or_inf(outlier_score))
                        outlier_score = 1. - 1e-15;
                } else {
                    outlier_score = model_outputs.all_clusters[col][cl].score_categ[cat_val_this];
                }

                /* if this is the best cluster so far, remember it */
                /* Note: if parallelizing by columns, must turn this into a critical section as the previously-assigned column can change in the meantime */
                // #pragma omp critical
                if (
                    model_outputs.outlier_scores_final[row] >= 1.0 ||
                    (
                        cluster_depth < model_outputs.outlier_depth_final[row] &&
                        (
                            model_outputs.all_clusters[col][cl].has_NA_branch
                                ==
                            model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
                        )
                    ) ||
                        (
                        model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch &&
                        !model_outputs.all_clusters[col][cl].has_NA_branch
                        ) ||
                        (
                        cluster_depth == model_outputs.outlier_depth_final[row] &&
                            (
                                model_outputs.all_clusters[col][cl].has_NA_branch
                                    ==
                                model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
                            ) &&
                        cluster_size > model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].cluster_size
                        ) ||
                        (
                        cluster_depth == model_outputs.outlier_depth_final[row] &&
                        cluster_size == model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].cluster_size &&
                        (
                            model_outputs.all_clusters[col][cl].has_NA_branch
                                ==
                            model_outputs.all_clusters[model_outputs.outlier_columns_final[row]][model_outputs.outlier_clusters_final[row]].has_NA_branch
                        ) &&
                        outlier_score < model_outputs.outlier_scores_final[row]
                        )
                )
                {
                    model_outputs.outlier_columns_final[row] = col;
                    model_outputs.outlier_scores_final[row] = outlier_score;
                    model_outputs.outlier_clusters_final[row] = cl;
                    model_outputs.outlier_trees_final[row] = curr_tree;
                    model_outputs.outlier_depth_final[row] = cluster_depth;
                }


            }

        }

    }

    return tree_has_outliers;

}
