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



/*    Characterize a homogenous 1-dimensional cluster
*    
*    Calcualtes limits and display statistics on the distribution of one numerical variable,
*    flagging potential outliers if found. Can be run on the full data or on subsets obtained from splitting
*    by other variables.
*    
*    In order to flag an observation as outlier, it must:
*        * Be in a very small/large percentile of the subset passed here.
*        * Have a large absolute Z value (standardized and centered).
*        * Have a large gap in the Z value with respect to the next largest/smallest ovservation.
*        * Not be in a long tail (unless the variable was transformed by exponentiating or taking logarithm).
*    
*    Parameters:
*    - x[n] (in)
*        Variable for which to define the cluster.
*    - ix_arr[n] (in)
*        Indices to take from the array above.
*    - st (in)
*        Position at which ix_arr starts (inclusive).
*    - end (in)
*        Position at which ix_arr ends (inclusive).
*    - outlier_scores[n] (in, out)
*        Outlier scores (based on chebyshyov's inequality) that are already assigned to the observations from this column
*        from previous runs of this function in larger subsets (should be started to 1).
*    - outlier_clusters[n] (in, out)
*        Cluster number under which an observation is the most anomalous.
*    - outlier_trees[n] (in, out)
*        Tree under which the outlier cluster assigned lies.
*    - outlier_depth[n] (in, out)
*        Tree depth at which the outlier cluster assigned is found.
*    - cluster (in, out)
*        Outlier cluster object with statistics and limits.
*    - clusters (in)
*        Vector containing all cluster already generated.
*    - cluster_num (in)
*        Number to give to this cluster.
*    - tree_num (in)
*        Number of the tree under which this cluster is to be found.
*    - tree_depth (in)
*        Distance form the tree root at which this tree is to be found.
*    - is_log_transf (in)
*        Whether the column 'x' has undergone a logarithmic transformation.
*    - log_minval (in)
*        Value that was added to 'x' before taking its logarithm (if it was log-transformed).
*    - is_exp_transf (in)
*        Whether the column 'x' has undergone an exponential transformation on its standardized values.
*    - orig_mean (in)
*        Mean of the variable 'x' before being standardized (if it was exponentiated).
*    - orig_sd (in)
*        Standard deviation of the variable 'x'  before being standardized (if it was exponentiated).
*    - left_tail (in)
*        Value of 'x' after which it is considered a long tail, in which outliers will not be searched for.
*    - rught_tail (in)
*        Value of 'x' before which it is considered a long tail, in which outliers will not be searched for.
*    - orig_x (in)
*        Original values of 'x' if it was transformed (log or exp).
*    - max_perc_outliers (in)
*        Model parameter. Default is 0.01.
*    - z_norm (in)
*        Model parameter. Default is 2.67.
*    - z_outlier (in)
*        Model parameter. Default is 8.0. Must be greater than z_norm.
*    
*    Returns:
*        - Whether there were any outliers detected.
*/
bool define_numerical_cluster(double *restrict x, size_t *restrict ix_arr, size_t st, size_t end,
                              double *restrict outlier_scores, size_t *restrict outlier_clusters, size_t *restrict outlier_trees,
                              size_t *restrict outlier_depth, Cluster &cluster, std::vector<Cluster> &clusters,
                              size_t cluster_num, size_t tree_num, size_t tree_depth,
                              bool is_log_transf, double log_minval, bool is_exp_transf, double orig_mean, double orig_sd,
                              double left_tail, double right_tail, double *restrict orig_x,
                              double max_perc_outliers, double z_norm, double z_outlier)
{

    /*  TODO: this function could try to determine if the distribution is multimodal, and if so,
        take only the most extreme means/sd for outlier comparisons */

    /*  TODO: statistics like SD, mean; are already available from the splitting function which
        is called right before this, so these should *only* need to be recalculated them if the column
        has undergone log or exp transform */

    /* NAs and Inf should have already been removed, and outliers with fewer conditionals already discarded */
    bool has_low_values  = false;
    bool has_high_values = false;
    long double running_mean = 0;
    long double running_ssq  = 0;
    long double mean_prev    = 0;
    double xval;
    double mean;
    double sd;
    size_t cnt;
    size_t tail_size     = (size_t) calculate_max_outliers((long double)(end - st + 1), max_perc_outliers);
    size_t st_non_tail   = st  + tail_size;
    size_t end_non_tail  = end - tail_size;
    size_t st_normals    = 0;
    size_t end_normals   = 0;
    double min_gap = z_outlier - z_norm;

    double curr_gap, next_gap, eps, lim_by_orig;

    /* Note: there is no good reason and no theory behind these numbers.
       TODO: find a better way of setting this */
    double min_gap_orig_scale = log(sqrtl((long double)(end - st + 1))) / 2.;
    min_gap_orig_scale = std::fmax(1.1, min_gap_orig_scale);
    min_gap_orig_scale = std::fmin(2.5, min_gap_orig_scale);

    /* TODO: here it's not necessary to sort the whole data, only top/bottom N */

    /* sort the data */
    std::sort(ix_arr + st, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});

    /* calculate statistics with tails and previous outliers excluded */
    cnt = end_non_tail - st_non_tail + 1;
    mean_prev = x[ ix_arr[st_non_tail] ];
    for (size_t row = st_non_tail; row <= end_non_tail; row++) {
        xval = x[ ix_arr[row] ];
        running_mean += (xval - running_mean) / (long double)(row - st_non_tail + 1);
        running_ssq  += (xval - running_mean) * (xval - mean_prev);
        mean_prev     = running_mean;

    }
    mean = (double) running_mean;
    sd   = (double) sqrtl(running_ssq / (long double)(cnt - 1));

    /* adjust SD heuristically to account for reduced size, by (N + tail)/(N-tail) --- note that cnt = N-2*tail */
    sd *= (long double)(cnt + 3 * tail_size) / (long double)(cnt + tail_size);
    /* re-adjust if there's a one-sided tail and no transformation was applies */
    if ((!isinf(left_tail) || !isinf(right_tail)) && !is_log_transf && !is_exp_transf) {
        sd *= 0.5;
    }
    sd = std::fmax(sd, 1e-15);
    while (std::numeric_limits<double>::epsilon() > sd*std::fmin(min_gap, z_norm))
        sd *= 4;
    cluster.cluster_mean = mean;
    cluster.cluster_sd = sd;
    cnt = end - st + 1;

    /* TODO: review how to better set this limit */
    tail_size = std::min(tail_size, (size_t)ceill(log2l((long double)(end - st + 1))));

    /* see if the minimum and/or maximum values qualify for outliers */
    if (-z_score(x[ix_arr[st]],  mean, sd) >= z_outlier && x[ix_arr[st]]  > left_tail)  has_low_values  = true;
    if ( z_score(x[ix_arr[end]], mean, sd) >= z_outlier && x[ix_arr[end]] < right_tail) has_high_values = true;

    /* look for a large gap in the z-scores */
    if (has_low_values) {
        for (size_t row = st; row < st + tail_size; row++) {

            if (( z_score(x[ix_arr[row + 1]], mean, sd) - z_score(x[ix_arr[row]], mean, sd) ) >= min_gap) {
                
                /* if the variable was transformed, check that the gap is still wide in the original scale */
                if (is_exp_transf || is_log_transf) {
                    curr_gap = orig_x[ix_arr[row + 1]] - orig_x[ix_arr[row]];
                    next_gap = 0;
                    for (size_t rr = row + 1; rr < end; rr++) {
                        if (orig_x[ix_arr[rr+1]] > orig_x[ix_arr[rr]]) {
                            next_gap = orig_x[ix_arr[rr+1]] - orig_x[ix_arr[rr]];
                            break;
                        }
                    }

                    if (next_gap > 0 && curr_gap/next_gap < min_gap_orig_scale)
                        continue;
                }

                st_normals = row + 1;
                if (is_exp_transf) {
                    cluster.lower_lim = log(x[ix_arr[row + 1]] - min_gap * sd) * orig_sd + orig_mean;
                } else if (is_log_transf) {
                    cluster.lower_lim = exp(x[ix_arr[row + 1]] - min_gap * sd) + log_minval;
                } else {
                    cluster.lower_lim = x[ix_arr[row + 1]] - min_gap * sd;
                }
                cluster.display_lim_low = orig_x[ix_arr[row + 1]];
                cluster.perc_above = (long double)(end - st_normals + 1) / (long double)(end - st + 1);

                eps = 1e-15;
                while (cluster.display_lim_low <= cluster.lower_lim) {
                    cluster.lower_lim -= eps;
                    eps *= 4;
                }
                break;
            }
            if (z_score(x[ix_arr[row]], mean, sd) > -z_outlier) break;

        }
        if (st_normals == 0) {
            has_low_values = false;
        } else {
            for (size_t row = st; row < st_normals; row++) {

                /* assign outlier if it's a better cluster than previously assigned */
                if (
                        outlier_scores[ix_arr[row]] >= 1.0 ||
                        (clusters[outlier_clusters[ix_arr[row]]].has_NA_branch && !cluster.has_NA_branch) ||
                        (
                            cluster.has_NA_branch == clusters[outlier_clusters[ix_arr[row]]].has_NA_branch
                                &&
                            (
                                tree_depth < outlier_depth[ix_arr[row]] ||
                                (
                                    tree_depth == outlier_depth[ix_arr[row]] &&
                                    clusters[outlier_clusters[ix_arr[row]]].cluster_size < (cnt - 2 * tail_size)
                                )
                            )
                        )
                    )
                {
                    outlier_scores[ix_arr[row]] = chebyshyov_bound(z_score(x[ix_arr[row]], mean, sd));
                    if (is_na_or_inf(outlier_scores[ix_arr[row]])) outlier_scores[ix_arr[row]] = 0;
                    outlier_clusters[ix_arr[row]] = cluster_num;
                    outlier_trees[ix_arr[row]] = tree_num;
                    outlier_depth[ix_arr[row]] = tree_depth;
                }

            }
        }
    }
    if (!has_low_values) {
        cluster.perc_above = 1.0;
        if (!is_log_transf && !is_exp_transf) {

            if (isinf(left_tail)) {
                cluster.lower_lim = x[ix_arr[st]] - min_gap * sd;
            } else {
                cluster.lower_lim = -HUGE_VAL;
            }

        } else if (is_exp_transf) {
            cluster.lower_lim = log(x[ix_arr[st]] - min_gap * sd) * orig_sd + orig_mean;
        } else {
            cluster.lower_lim = exp(x[ix_arr[st]] - min_gap * sd) + log_minval;
        }

        if (cluster.lower_lim > -HUGE_VAL) {
            eps = 1e-15;
            while (cluster.lower_lim >= orig_x[ix_arr[st]]) {
                cluster.lower_lim -= eps;
                eps *= 4.;
            }
        }

        if (is_exp_transf || is_log_transf) {
            for (size_t row = st; row < end; row++) {
                if (orig_x[ix_arr[row+1]] > orig_x[ix_arr[row]]) {
                    curr_gap = orig_x[ix_arr[row+1]] - orig_x[ix_arr[row]];
                    lim_by_orig = orig_x[ix_arr[st]] - min_gap_orig_scale * curr_gap;
                    cluster.lower_lim = std::fmin(cluster.lower_lim, lim_by_orig);
                    break;
                }
            }
        }

        cluster.display_lim_low = orig_x[ix_arr[st]];

    }

    if (has_high_values) {
        for (size_t row = end; row > (end - tail_size); row--) {

            if (( z_score(x[ix_arr[row]], mean, sd) - z_score(x[ix_arr[row - 1]], mean, sd) ) >= min_gap) {
                
                /* if the variable was transformed, check that the gap is still wide in the original scale */
                if (is_exp_transf || is_log_transf) {
                    curr_gap = orig_x[ix_arr[row]] - orig_x[ix_arr[row - 1]];
                    next_gap = 0;
                    for (size_t rr = row-1; rr > st; rr--) {
                        if (orig_x[ix_arr[rr]] > orig_x[ix_arr[rr-1]]) {
                            next_gap = orig_x[ix_arr[rr]] - orig_x[ix_arr[rr-1]];
                            break;
                        }
                    }

                    if (next_gap > 0 && curr_gap/next_gap < min_gap_orig_scale)
                        continue;
                }

                end_normals = row - 1;
                if (is_exp_transf) {
                    cluster.upper_lim = log(x[ix_arr[row - 1]] + min_gap * sd) * orig_sd + orig_mean;
                } else if (is_log_transf) {
                    cluster.upper_lim = exp(x[ix_arr[row - 1]] + min_gap * sd) + log_minval;
                } else {
                    cluster.upper_lim = x[ix_arr[row - 1]] + min_gap * sd;
                }
                cluster.display_lim_high = orig_x[ix_arr[row - 1]];
                cluster.perc_below = (long double)(end_normals - st + 1) / (long double)(end - st + 1);

                eps = 1e-15;
                while (cluster.display_lim_high >= cluster.upper_lim) {
                    cluster.upper_lim += eps;
                    eps *= 4;
                }
                break;
            }
            if (z_score(x[ix_arr[row]], mean, sd) < z_outlier) break;

        }
        if (end_normals == 0) {
            has_high_values = false;
        } else {
            for (size_t row = end; row > end_normals; row--) {

                /*  assign outlier if it's a better cluster than previously assigned - Note that it might produce slight mismatches
                    against the predict function (the latter is more trustable) due to the size of the cluster not yet being known
                    at the moment of determinining whether to overwrite previous in here */
                if (
                        outlier_scores[ix_arr[row]] >= 1.0 ||
                        (clusters[outlier_clusters[ix_arr[row]]].has_NA_branch && !cluster.has_NA_branch) ||
                        (
                            cluster.has_NA_branch == clusters[outlier_clusters[ix_arr[row]]].has_NA_branch
                            &&
                            (
                                tree_depth < outlier_depth[ix_arr[row]] ||
                                (
                                    tree_depth == outlier_depth[ix_arr[row]] &&
                                    clusters[outlier_clusters[ix_arr[row]]].cluster_size < (cnt - 2 * tail_size)
                                )
                            )
                        )
                    )
                {
                    outlier_scores[ix_arr[row]] = chebyshyov_bound(z_score(x[ix_arr[row]], mean, sd));
                    if (is_na_or_inf(outlier_scores[ix_arr[row]])) outlier_scores[ix_arr[row]] = 0;
                    outlier_clusters[ix_arr[row]] = cluster_num;
                    outlier_trees[ix_arr[row]] = tree_num;
                    outlier_depth[ix_arr[row]] = tree_depth;
                }

            }
        }
    }
    if (!has_high_values) {
        cluster.perc_below = 1.0;
        if (!is_log_transf && !is_exp_transf) {

            if (isinf(right_tail)) {
                cluster.upper_lim = x[ix_arr[end]] + min_gap * sd;
            } else {
                cluster.upper_lim = HUGE_VAL;
            }
        } else if (is_exp_transf) {
            cluster.upper_lim = log(x[ix_arr[end]] + min_gap * sd) * orig_sd + orig_mean;
        } else {
            cluster.upper_lim = exp(x[ix_arr[end]] + min_gap * sd) + log_minval;
        }

        if (cluster.upper_lim < HUGE_VAL) {
            eps = 1e-15;
            while (cluster.upper_lim <= orig_x[ix_arr[end]]) {
                cluster.upper_lim += eps;
                eps *= 4.;
            }
        }

        if (is_exp_transf || is_log_transf) {
            for (size_t row = end; row < st; row--) {
                if (orig_x[ix_arr[row]] > orig_x[ix_arr[row-1]]) {
                    curr_gap = orig_x[ix_arr[row]] - orig_x[ix_arr[row-1]];
                    lim_by_orig = orig_x[ix_arr[end]] + min_gap_orig_scale * curr_gap;
                    cluster.upper_lim = std::fmax(cluster.upper_lim, lim_by_orig);
                    break;
                }
            }
        }

        cluster.display_lim_high = orig_x[ix_arr[end]];
    }

    /* save displayed statistics for cluster */
    if (has_high_values || has_low_values || is_log_transf || is_exp_transf) {
        size_t st_disp  = has_low_values?  st_normals  : st;
        size_t end_disp = has_high_values? end_normals : end;
        running_mean = 0;
        running_ssq  = 0;
        mean_prev    = orig_x[ix_arr[st_disp]];
        for (size_t row = st_disp; row <= end_disp; row++) {
            xval = orig_x[ix_arr[row]];
            running_mean += (xval - running_mean) / (long double)(row - st_disp + 1);
            running_ssq  += (xval - running_mean) * (xval - mean_prev);
            mean_prev     = running_mean;
        }
        cluster.cluster_size = end_disp - st_disp + 1;
        cluster.display_mean = (double) running_mean;
        cluster.display_sd   = (double) sqrtl(running_ssq / (long double)(cluster.cluster_size - 1));
    } else {
        cluster.display_mean = cluster.cluster_mean;
        cluster.display_sd   = cluster.cluster_sd;
        cluster.cluster_size = end - st + 1;
    }

    /* report whether outliers were found or not */
    return has_low_values || has_high_values;
}


/*    Characterize a homogeneous categorical cluster from the *full* data
*    
*    Function is meant for the data as it comes, before splitting it, as once split, it will
*    not be able to detect these outliers. As such, it takes fewer parameters, since it can only
*    be the first tree and cluster in a column. It assumes the outliers have already been identified.
*    
*    Parameters:
*    - x[n]
*        Array indicating the category to which each observation belongs.
*    - ix_arr[n] (in)
*        Indices to take from the array above.
*    - st (in)
*        Position at which ix_arr starts (inclusive).
*    - end (in)
*        Position at which ix_arr ends (inclusive).
*    - ncateg (in)
*        Number of categories in this column.
*    - outlier_scores[n] (in, out)
*        Array where to assign outlier scores (based on proportion) to each observation belonging to an outlier category.
*    - outlier_clusters[n] (in, out)
*        Array where to assign cluster number to each observation belonging to an outlier category.
*    - outlier_trees[n] (in, out)
*        Array where to assign tree number to each observation belonging to an outlier category.
*    - outlier_depth[n] (in, out)
*        Array where to assign tree depth to each observation belonging to an outlier category.
*    - cluster (in, out)
*        Outlier cluster object with statistics and classifications.
*    - categ_counts[ncateg] (in)
*        Array with the frequencies of each category in the data.
*    - is_outlier[ncateg] (in)
*        Array indicating which categories are to be considered as outliers (must be already calculated).
*    - perc_next_most_comm (in)
*        Proportion of the least common non-outlier category (must be already calculated).
*/
void define_categ_cluster_no_cond(int *restrict x, size_t *restrict ix_arr, size_t st, size_t end, size_t ncateg,
                                  double *restrict outlier_scores, size_t *restrict outlier_clusters, size_t *restrict outlier_trees,
                                  size_t *restrict outlier_depth, Cluster &cluster,
                                  size_t *restrict categ_counts, signed char *restrict is_outlier, double perc_next_most_comm)
{
    size_t cnt_common = end - st + 1;
    cluster.cluster_size = cnt_common;
    double pct_outl;
    cluster.subset_common.assign(is_outlier, is_outlier + ncateg);
    cluster.score_categ.resize(ncateg, 0);


    for (size_t row = st; row <= end; row++) {
        if (is_outlier[x[ix_arr[row]]]) {
            cnt_common--;
            pct_outl = (long double)categ_counts[ x[ix_arr[row]] ] / (long double)cluster.cluster_size;
            pct_outl = pct_outl + sqrt(pct_outl * (1 - pct_outl) / (long double)cluster.cluster_size);
            cluster.score_categ[ x[ix_arr[row]] ] = pct_outl;
            outlier_scores[ix_arr[row]] = pct_outl;
            outlier_clusters[ix_arr[row]] = 0;
            outlier_trees[ix_arr[row]] = 0;
            outlier_depth[ix_arr[row]] = 0;
        }
    }
    cluster.perc_in_subset = (long double)cnt_common / (long double)cluster.cluster_size;
    cluster.perc_next_most_comm = perc_next_most_comm;
}


/*    Characterize a homogeneous categorical cluster form a subset of the data, or report if it's not homogeneous
*    
*    Function is meant to be called with subsets of the data only. Will calculate the counts inside it.
*    In order to consider a category as outlier, it must:
*        * Have a proportion smaller than its prior probability and than a condifence interval of its prior.
*        * Have a large gap with respect to the next most-common category.
*        * Be in a cluster in which few or no observations belong to a category meeting such conditions.
*    It's oftentimes not possible to create a cluster with category frequencies that would produce outliers,
*    in which case it will report whether the cluster should be dropped.
*    
*    Parameters:
*    - x[n]
*        Array indicating the category to which each observation belongs.
*    - ix_arr[n] (in)
*        Indices to take from the array above.
*    - st (in)
*        Position at which ix_arr starts (inclusive).
*    - end (in)
*        Position at which ix_arr ends (inclusive).
*    - ncateg (in)
*        Number of categories in this column.
*    - by_maj (in)
*        Model parameter. Default is 'false'. Indicates whether to detect outliers according to the number of non-majority
*        obsevations compared to the expected number for each category.
*    - outlier_scores[n] (in, out)
*        Outlier scores (based on observed category proportion) that are already assigned to the observations from this column
*        from previous runs of this function in larger subsets (should be started to 1).
*    - outlier_clusters[n] (in, out)
*        Cluster number under which an observation is the most anomalous.
*    - outlier_trees[n] (in, out)
*        Tree under which the outlier cluster assigned lies.
*    - outlier_depth[n] (in, out)
*        Tree depth at which the outlier cluster assigned is found.
*    - cluster (in, out)
*        Outlier cluster object with statistics and limits.
*    - clusters (in)
*        Vector containing all cluster already generated.
*    - cluster_num (in)
*        Number to give to this cluster.
*    - tree_num (in)
*        Number of the tree under which this cluster is to be found.
*    - tree_depth (in)
*        Distance form the tree root at which this tree is to be found.
*    - max_perc_outliers (in)
*        Model parameter. Default is 0.01.
*    - z_norm (in)
*        Model parameter. Default is 2.67.
*    - z_outlier (in)
*        Model parameter. Default is 8.0.
*    - perc_threshold[ncateg] (in)
*        Observed proportion below which a category can be considered as outlier.
*    - prop_prior[ncateg] (in)
*        Prior probability of each category in the full data (only used when passing 'by_maj' = 'true').
*    - buffer_categ_counts[ncateg] (temp)
*        Buffer where to save the observed frequencies of each category.
*    - buffer_categ_pct[ncateg] (temp)
*        Buffer where to save the observed proportion of each category.
*    - buffer_categ_ix[ncateg] (temp)
*        Buffer where to save the category numbers sorted by proportion.
*    - buffer_outliers[ncateg] (temp)
*        Buffer where to save the results of which categories are flagged as outliers
*        before copying it to the cluster (will not copy if none is flagged).
*    - drop_cluster (out)
*        Whethet the cluster should be dropped (i.e. it was not possible to flag any present
*        or non-present category as outlier).
*    
*    Returns:
*        - Whether it identified any outliers or not.
*/
bool define_categ_cluster(int *restrict x, size_t *restrict ix_arr, size_t st, size_t end, size_t ncateg, bool by_maj,
                          double *restrict outlier_scores, size_t *restrict outlier_clusters, size_t *restrict outlier_trees,
                          size_t *restrict outlier_depth, Cluster &cluster, std::vector<Cluster> &clusters,
                          size_t cluster_num, size_t tree_num, size_t tree_depth,
                          double max_perc_outliers, double z_norm, double z_outlier,
                          long double *restrict perc_threshold, long double *restrict prop_prior,
                          size_t *restrict buffer_categ_counts, long double *restrict buffer_categ_pct,
                          size_t *restrict buffer_categ_ix, signed char *restrict buffer_outliers,
                          bool *restrict drop_cluster)
{
    bool found_outliers, new_is_outlier;
    size_t tot = end - st + 1;
    size_t sz_maj = tot;
    long double tot_dbl = (long double) tot;
    size_t tail_size = (size_t) calculate_max_outliers(tot_dbl, max_perc_outliers);
    cluster.perc_in_subset = 1;
    double pct_outl;

    /* calculate category counts */
    memset(buffer_categ_counts, 0, ncateg * sizeof(size_t));
    for (size_t row = st; row <= end; row++) {
        buffer_categ_counts[ x[ix_arr[row]] ]++;
    }

    /* flag categories as outliers if appropriate */
    if (!by_maj)
        find_outlier_categories(buffer_categ_counts, ncateg, tot, max_perc_outliers,
                                perc_threshold, buffer_categ_ix, buffer_categ_pct,
                                z_norm, buffer_outliers, &found_outliers,
                                &new_is_outlier, &cluster.perc_next_most_comm);
    else
        find_outlier_categories_by_maj(buffer_categ_counts, ncateg, tot, max_perc_outliers,
                                       prop_prior, z_outlier, buffer_outliers,
                                       &found_outliers, &new_is_outlier, &cluster.categ_maj);

    if (found_outliers) {
        for (size_t row = st; row <= end; row++) {
            if (buffer_outliers[ x[ix_arr[row]] ]) {

                /* follow usual rules for preferring this cluster over others */
                if (
                        outlier_scores[ix_arr[row]] >= 1.0 ||
                        (clusters[outlier_clusters[ix_arr[row]]].has_NA_branch && !cluster.has_NA_branch) ||
                        (
                            cluster.has_NA_branch == clusters[outlier_clusters[ix_arr[row]]].has_NA_branch
                            &&
                            (
                                tree_depth < outlier_depth[ix_arr[row]] ||
                                (
                                    tree_depth == outlier_depth[ix_arr[row]] &&
                                    clusters[outlier_clusters[ix_arr[row]]].cluster_size < (tot - tail_size)
                                )
                            )
                        )
                    )
                {
                    if (!by_maj) {
                        pct_outl = (long double)buffer_categ_counts[ x[ix_arr[row]] ] / tot_dbl;
                        pct_outl = pct_outl + sqrt(pct_outl * (1 - pct_outl) / tot_dbl);
                        outlier_scores[ix_arr[row]] = pct_outl;
                    } else {
                        pct_outl = (long double)(tot - buffer_categ_counts[cluster.categ_maj]) / (tot_dbl * prop_prior[ x[ix_arr[row]] ]);
                        outlier_scores[ix_arr[row]] = square(pct_outl);
                    }
                    outlier_clusters[ix_arr[row]] = cluster_num;
                    outlier_trees[ix_arr[row]] = tree_num;
                    outlier_depth[ix_arr[row]] = tree_depth;
                }
                sz_maj--;

            }
        }
        cluster.perc_in_subset = (long double)sz_maj / tot_dbl;
    }

    if (new_is_outlier && !found_outliers) {
        cluster.perc_in_subset = 1.0;
    }

    if (new_is_outlier || found_outliers) {
        *drop_cluster = false;
        cluster.cluster_size = sz_maj;
        cluster.subset_common.assign(buffer_outliers, buffer_outliers + ncateg);
        cluster.score_categ.resize(ncateg, 0);
        if (!by_maj) {

            for (size_t cat = 0; cat < ncateg; cat++) {
                if (cluster.subset_common[cat] > 0) {
                    pct_outl = (long double)buffer_categ_counts[cat] / tot_dbl;
                    cluster.score_categ[cat] = pct_outl + sqrt(pct_outl * (1 - pct_outl) / tot_dbl);
                } else if (cluster.subset_common[cat] < 0) {
                    pct_outl = (long double)1 / (long double)(tot + 2);
                    cluster.score_categ[cat] = pct_outl + sqrt(pct_outl * (1 - pct_outl) / (long double)(tot + 2));
                }
            }

        } else {

            cluster.perc_in_subset = (long double) buffer_categ_counts[cluster.categ_maj] / tot_dbl;
            for (size_t cat = 0; cat < ncateg; cat++) {
                if (cat == cluster.categ_maj)
                    continue;
                if (cluster.subset_common[cat] != 0) {
                    cluster.score_categ[cat] = (long double)(tot - buffer_categ_counts[cluster.categ_maj] + 1)
                                                            / ((long double)(tot + 2) * prop_prior[cat]);
                    cluster.score_categ[cat] = square(cluster.score_categ[cat]);
                }
            }

        }
    } else {
        *drop_cluster = true;
    }

    return found_outliers;
}

/* Convert in/not-in conditions to 'equals' or 'not equals' when they look for only 1 category */
void simplify_when_equal_cond(std::vector<Cluster> &clusters, int ncat_ord[])
{

    int col_equal;
    size_t size_subset;
    size_t size_subset_excl;
    for (size_t clust = 0; clust < clusters.size(); clust++) {
        if (clusters[clust].split_type == IsNa) continue;

        switch(clusters[clust].column_type) {

            case Categorical:
            {

                col_equal = -1;
                if (clusters[clust].split_subset.size() == 2) {

                    switch(col_equal = clusters[clust].split_type) {
                        case InSubset:
                        {
                            col_equal = clusters[clust].split_subset[0]? 0 : 1;
                            break;
                        }

                        case NotInSubset:
                        {
                            col_equal = clusters[clust].split_subset[0]? 1 : 0;
                            break;
                        }

                        case SingleCateg:
                        {
                            col_equal = clusters[clust].split_subset[0]? 0 : 1;
                            break;
                        }
                    }
                    clusters[clust].split_type = Equal;

                } else {

                    size_subset_excl = std::accumulate(clusters[clust].split_subset.begin(), clusters[clust].split_subset.end(), (size_t)0,
                                                       [](const size_t a, const signed char b){return a + ((b < 0)? 1 : 0);});
                    if (size_subset_excl > 0) continue;
                    size_subset = std::accumulate(clusters[clust].split_subset.begin(), clusters[clust].split_subset.end(), (size_t)0,
                                                  [](const size_t a, const signed char b){return a + ((b > 0)? 1 : 0);});
                    if (size_subset == 1) {

                        do {col_equal++;} while (clusters[clust].split_subset[col_equal] <= 0);
                        if (clusters[clust].split_type == InSubset || clusters[clust].split_type == SingleCateg)
                            clusters[clust].split_type = Equal;
                        else
                            clusters[clust].split_type = NotEqual;

                    } else if (size_subset == (clusters[clust].split_subset.size() - 1)) {

                        do {col_equal++;} while (clusters[clust].split_subset[col_equal] != 0);
                        if (clusters[clust].split_type == NotInSubset)
                            clusters[clust].split_type = Equal;
                        else
                            clusters[clust].split_type = NotEqual;

                    }

                }
                if (col_equal >= 0) {
                    clusters[clust].split_subset.resize(0);
                    clusters[clust].split_lev = col_equal;
                }
                break;
            }


            case Ordinal: 
            {

                if (clusters[clust].split_lev == 0) {

                    if (clusters[clust].split_type == LessOrEqual)
                        clusters[clust].split_type = Equal;
                    else
                        clusters[clust].split_type = NotEqual;

                }

                else if (clusters[clust].split_lev == (ncat_ord[clusters[clust].col_num] - 2)) {

                    clusters[clust].split_lev++;
                    if (clusters[clust].split_type == Greater)
                        clusters[clust].split_type = Equal;
                    else
                        clusters[clust].split_type = NotEqual;

                }
                break;
            }

            default: {}
        }

    }

}

/*    
*    Convert in/not-in conditions to 'equals' when they look for only 1 category
*    Note: unlike in the case of clusters, trees do not store the split type, but rather
*    always assume left is in/l.e. and right the opposite, so it's not possible to
*    simplify ordinal splits to equals (as the tree will not distinguish between
*    an ordinal split with equals and another with l.e./g.e.). Thus, this part needs
*    to be done in the function that prints the outlier conditions.
*/
void simplify_when_equal_cond(std::vector<ClusterTree> &trees, int ncat_ord[])
{

    int col_equal;
    size_t size_subset;
    size_t size_subset_excl;
    size_t temp_swap;
    for (size_t tree = 0; tree < trees.size(); tree++) {

        if (trees[tree].all_branches.size() == 0 && trees[tree].tree_left == 0 && trees[tree].tree_right == 0) continue;
        if (trees[trees[tree].parent].all_branches.size() > 0 && trees[tree].split_this_branch == IsNa) continue;
        switch(trees[tree].column_type) {

            case Categorical:
            {
                size_subset_excl = std::accumulate(trees[tree].split_subset.begin(), trees[tree].split_subset.end(), (size_t)0,
                                                   [](const size_t a, const signed char b){return a + ((b < 0)? 1 : 0);});
                if (size_subset_excl > 0) continue;

                col_equal = -1;
                if (trees[tree].split_subset.size() == 2) {

                    col_equal = 0;
                    if (trees[tree].split_subset[0] == 0) {
                        temp_swap = trees[tree].tree_left;
                        trees[tree].tree_left = trees[tree].tree_right;
                        trees[tree].tree_right = temp_swap;
                    }
                    if (trees[tree].tree_left > 0)
                        trees[trees[tree].tree_left].parent_branch = Equal;
                    if (trees[tree].tree_right > 0)
                        trees[trees[tree].tree_right].parent_branch = NotEqual;

                    if (trees[trees[tree].parent].all_branches.size() > 0) {
                        switch(trees[tree].split_this_branch) {
                            case InSubset:
                            {
                                trees[tree].split_this_branch = Equal;
                                break;
                            }

                            case NotInSubset:
                            {
                                trees[tree].split_this_branch = NotEqual;
                                break;
                            }

                            case SingleCateg:
                            {
                                trees[tree].split_this_branch = Equal;
                                break;
                            }

                            default: {}
                        }
                    }

                }

                else {

                    size_subset = std::accumulate(trees[tree].split_subset.begin(), trees[tree].split_subset.end(), (size_t)0,
                                                  [](const size_t a, const signed char b){return a + ((b > 0)? 1 : 0);});
                    if (size_subset == 1) {

                        do {col_equal++;} while (trees[tree].split_subset[col_equal] <= 0);
                        if (trees[trees[tree].parent].all_branches.size() > 0) {
                            switch(trees[tree].split_this_branch) {
                                case InSubset:
                                {
                                    trees[tree].split_this_branch = Equal;
                                    break;
                                }

                                case NotInSubset:
                                {
                                    trees[tree].split_this_branch = NotEqual;
                                    break;
                                }

                                case SingleCateg:
                                {
                                    trees[tree].split_this_branch = Equal;
                                    break;
                                }

                                default: {}
                            }
                        }


                    } else if (size_subset == (trees[tree].split_subset.size() - 1)) {

                        do {col_equal++;} while (trees[tree].split_subset[col_equal] != 0);
                        temp_swap = trees[tree].tree_left;
                        trees[tree].tree_left = trees[tree].tree_right;
                        trees[tree].tree_right = temp_swap;
                        if (trees[trees[tree].parent].all_branches.size() > 0) {
                            switch(trees[tree].split_this_branch) {
                                case InSubset:
                                {
                                    trees[tree].split_this_branch = NotEqual;
                                    break;
                                }

                                case NotInSubset:
                                {
                                    trees[tree].split_this_branch = Equal;
                                    break;
                                }

                                default: {}
                            }
                        }

                    }

                }

                if (col_equal >= 0) {
                    trees[tree].split_subset.resize(0);
                    trees[tree].split_lev = col_equal;
                    if (trees[tree].tree_left > 0)
                        trees[trees[tree].tree_left].parent_branch = Equal;
                    if (trees[tree].tree_right > 0)
                        trees[trees[tree].tree_right].parent_branch = NotEqual;

                }
                break;
            }


            case Ordinal:
            {
                if (trees[trees[tree].parent].all_branches.size() == 0) continue;

                if (trees[tree].split_lev == 0) {

                    if (trees[tree].split_this_branch == LessOrEqual)
                        trees[tree].split_this_branch = Equal;
                    else
                        trees[tree].split_this_branch = NotEqual;

                }

                else if (trees[tree].split_lev == (ncat_ord[trees[tree].col_num] - 2)) {

                    trees[tree].split_lev++;
                    if (trees[tree].split_this_branch == Greater)
                        trees[tree].split_this_branch = Equal;
                    else
                        trees[tree].split_this_branch = NotEqual;

                }
                break;
            }

            default: {}
        }

    }

}

#ifdef TEST_MODE_DEFINE
/*    
*    Goodie to help with testing and debugging (not used in the final code)
*    
*    This function tries to unconnect unnecessary trees so that, if a tree has no clusters and its children
*    don't have any clusters either, such tree would not be reached at prediction time. It will drop trees from the vector
*    if they happen to lie at the end of it, but otherwise will just leave them there so as not to have to recalculate
*    the tree indexes and avoid having to update them everywhere they are referenced (such as in identified outliers).
*
*    This is only for categorical and ordinal columns, as numerical columns will always produce produce clusters when
*    they have children.
*    
*    This is supposed to be done with the conditions at the end of each recursive function, but this piece of
*    code can provide help in identifying errors when the code is modified.
*/
void prune_unused_trees(std::vector<ClusterTree> &trees)
{
    /* TODO: when using 'follow_all', function should delete instead of disconnect by setting to zero */
    if (trees.size() == 0) return;
    for (size_t t = trees.size() - 1; t >= 0; t--) {

        if (trees[t].binary_branches.size() > 0) {
            for (size_t br = 0; br < trees[t].binary_branches.size(); br++) {
                if (trees[t].binary_branches[br] == 0) continue;
                if (trees[t].binary_branches[br] >= trees.size()) trees[t].binary_branches[br] = 0;
                if (check_tree_is_not_needed(trees[trees[t].binary_branches[br]])) trees[t].binary_branches[br] = 0;
            }
        }

        if (trees[t].all_branches.size() > 0) {
            for (size_t br = 0; br < trees[t].all_branches.size(); br++) {
                if (trees[t].all_branches[br] == 0) continue;
                if (trees[t].all_branches[br] >= trees.size()) trees[t].all_branches[br] = 0;
                if (check_tree_is_not_needed(trees[trees[t].all_branches[br]])) trees[t].all_branches[br] = 0;
            }
        }


        if (check_tree_is_not_needed(trees[t])) {

            /* disconnect tree from parent */
            switch(trees[t].parent_branch) {
                case IsNa:
                {
                    trees[trees[t].parent].tree_NA = 0;
                    break;
                }

                case LessOrEqual:
                {
                    trees[trees[t].parent].tree_left = 0;
                    break;
                }

                case Greater:
                {
                    trees[trees[t].parent].tree_right = 0;
                    break;
                }

                case InSubset:
                {
                    trees[trees[t].parent].tree_left = 0;
                    break;
                }

                case NotInSubset:
                {
                    trees[trees[t].parent].tree_right = 0;
                    break;
                }

            }

            if (t == (trees.size() - 1)) trees.pop_back();
        }
        if (t == 0) break;
    }
}
#endif

/* Check whether a tree has no clusters and no children with clusters either */
bool check_tree_is_not_needed(ClusterTree &tree)
{
    return 
        tree.tree_NA == 0 && tree.tree_left == 0 && tree.tree_right == 0 &&
        tree.clusters.size() == 0 &&
        (tree.binary_branches.size() == 0 || *std::max_element(tree.binary_branches.begin(), tree.binary_branches.end()) == 0) &&
        (tree.all_branches.size() == 0 || *std::max_element(tree.all_branches.begin(), tree.all_branches.end()) == 0)
        ;
}

/*    
*    These functions simply check what's the minimum/maximum value that could identify an observation
*    as outlier in any cluster, or which categories could be possibly flagged as outliers in any cluster.
*    This info is redundant, as outliers can be identified by following splits, but it can help speed up
*    things at prediction time by not having to even bother checking a column if the value is within
*    non-flaggable limits.
*/
void calculate_cluster_minimums(ModelOutputs &model_outputs, size_t col)
{
    for (size_t cl = 0; cl < model_outputs.all_clusters[col].size(); cl++) {
        model_outputs.min_outlier_any_cl[col] = fmax(model_outputs.min_outlier_any_cl[col], model_outputs.all_clusters[col][cl].lower_lim);
        model_outputs.max_outlier_any_cl[col] = fmin(model_outputs.max_outlier_any_cl[col], model_outputs.all_clusters[col][cl].upper_lim);
    }

}

void calculate_cluster_poss_categs(ModelOutputs &model_outputs, size_t col, size_t col_rel)
{
    if (model_outputs.all_clusters[col].size() == 0) return;
    model_outputs.cat_outlier_any_cl[col_rel].resize(model_outputs.all_clusters[col][0].subset_common.size(), 0);
    for (size_t cl = 0; cl < model_outputs.all_clusters[col].size(); cl++) {
        for (size_t cat = 0; cat < model_outputs.all_clusters[col][cl].subset_common.size(); cat++) {
            if (model_outputs.all_clusters[col][cl].subset_common[cat] != 0) model_outputs.cat_outlier_any_cl[col_rel][cat] = true;
        }
    }
}
