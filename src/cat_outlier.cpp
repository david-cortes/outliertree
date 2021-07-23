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


/*    Check whether to consider any category as outlier, based on current counts and prior probabilities
*    
*    Function is to be applied to some subset of the data obtained by splitting by one or more columns.
*    For outliers before any split there is a separate function. Note that since it required current
*    probability to be lower than prior probability in order to consider as outlier, it cannot be
*    used with the full data (only with subsets).
*    
*    Parameters:
*    - categ_counts[ncateg] (in)
*        Counts of each category in the subset (including non-present categories).
*    - ncateg (in)
*        Number of categories for this column (including non-present categories).
*    - tot (in)
*        Number of rows in the subset.
*    - max_perc_outliers (in)
*        Model parameter. Default value is 0.01.
*    - perc_threshold[ncateg] (in)
*        Threshold for the proportion/probability of each category below which it can be considered
*        to be an outlier in a subset of the data. Note that in addition it will build a confidence
*        interval here which might make it even smaller.
*    - buffer_ix[ncateg] (temp)
*        Buffer where to store indices of categories sorted by proportion.
*    - buffer_perc[ncateg] (temp)
*        Buffer where to store proportions of counts.
*    - z_norm (in)
*        Model parameter. Default value is 2.67.
*    - is_outlier[ncateg] (out)
*        Array where to define whether any category is an outlier. Values will be as follows:
*            (-1) -> Category had zero count, but would be an outlier if it appeared among this group
*              0  -> Category is not an outlier
*            (+1) -> Category is an outlier
*    - found_outliers (out)
*        Whether there were any outliers identified among the counts.
*    - new_is_outlier (out)
*        Whether any of the categories with zero count would be flagged as outlier if they appeared in this group.
*    - next_most_comm (out)
*        Proportion of the least common category that is not flagged as outlier.
*/
void find_outlier_categories(size_t categ_counts[], size_t ncateg, size_t tot, double max_perc_outliers,
                             long double perc_threshold[], size_t buffer_ix[], long double buffer_perc[],
                             double z_norm, signed char is_outlier[], bool *found_outliers, bool *new_is_outlier,
                             double *next_most_comm)
{
    //TODO: must also establish bounds for new, unseen categories

    /* initialize parameters as needed */
    *found_outliers = false;
    *new_is_outlier = false;
    size_t st_non_zero = 0;
    size_t end_tail = 0;
    size_t max_outliers = (size_t) calculate_max_cat_outliers((long double)tot, max_perc_outliers, z_norm);
    long double tot_dbl = (long double) tot;
    long double pct_unseen = (long double)1 / (long double)(tot + 1);
    size_t size_tail = 0;

    /* reset the temporary arrays and fill them */
    memset(is_outlier, 0, ncateg * sizeof(signed char));
    for (size_t cat = 0; cat < ncateg; cat++) {
        buffer_ix[cat] = cat;
        buffer_perc[cat] = (categ_counts[cat] > 0)? ((long double)categ_counts[cat] / tot_dbl) : 0;
    }

    /* sort the categories by counts */
    std::sort(buffer_ix, buffer_ix + ncateg,
              [&categ_counts](const size_t a, const size_t b){return categ_counts[a] < categ_counts[b];});

    /* find the first non-zero */
    for (size_t cat = 0; cat < ncateg; cat++) {
        if (categ_counts[ buffer_ix[cat] ] > 0) {
            st_non_zero = cat;
            break;
        }
    }

    /* check that least common is not common enough to be normal */
    if (categ_counts[ buffer_ix[st_non_zero] ] > max_outliers) return;

    /*    find tail among non-zero proportions
    *    a tail is considered to be so if:
    *    - the difference is above z_norm sd's of either proportion
    *    - the difference is greater than some fraction of the larger
    *    - the actual proportion here is lower than a CI of the prior proportion
    *    - the actual proportion here is half or less of the prior proportion
    */
    for (size_t cat = st_non_zero; cat < ncateg - 1; cat++) {
        if (
                (
                    (buffer_perc[buffer_ix[cat + 1]] - buffer_perc[buffer_ix[cat]])
                        >
                    z_norm * sqrtl(
                                    fmaxl(
                                            buffer_perc[buffer_ix[cat + 1]] * ((long double)1 - buffer_perc[buffer_ix[cat + 1]]),
                                            buffer_perc[buffer_ix[cat]] * ((long double)1 - buffer_perc[buffer_ix[cat]])
                                        )
                                        / tot_dbl
                                )
                )
                &&
                (
                    buffer_perc[buffer_ix[cat + 1]] * 0.5  >  buffer_perc[buffer_ix[cat]]
                )
        )
        {
            end_tail = cat;
            *next_most_comm = buffer_perc[buffer_ix[cat + 1]];
            break;
        }
    }

    /* if the tail is too long, don't identify any as outlier, but see if unseen categories (with prior > 0) would create a new tail */
    for (size_t cat = st_non_zero; cat <= end_tail; cat++) size_tail += categ_counts[ buffer_ix[cat] ];

    if (size_tail >= max_outliers) {

        if (
            st_non_zero == 0 ||
            // ((long double)buffer_ix[buffer_ix[st_non_zero]] / (tot_dbl + 1)) * 0.5 <= pct_unseen ||
            ( ((long double)buffer_ix[buffer_ix[st_non_zero]] * 0.5) / (tot_dbl + 1)) <= pct_unseen ||
            ((long double)(buffer_ix[buffer_ix[st_non_zero]] - 1) / (tot_dbl + 1))
                - (long double)z_norm * sqrtl(buffer_perc[buffer_ix[st_non_zero]] * ((long double)1 - buffer_perc[buffer_ix[st_non_zero]]) / tot_dbl)
                    >= pct_unseen
            ) return;

        for (size_t cat = 0; cat < st_non_zero; cat++) {
            if (perc_threshold[buffer_ix[cat]] > pct_unseen) {
                *new_is_outlier = true;
                is_outlier[buffer_ix[cat]] = -1;
            }
        }
        *next_most_comm = buffer_perc[buffer_ix[st_non_zero]];
        return;

    }

    /* now determine if any category in the tail is an outlier */
    for (size_t cat = st_non_zero; cat <= end_tail; cat++) {

        /* must have a proportion below CI and below half of prior */
        if (buffer_perc[buffer_ix[cat]] < perc_threshold[buffer_ix[cat]]) {
            is_outlier[buffer_ix[cat]] = 1;
            *found_outliers = true;
        }
    }

    /* check if any new categories would be outliers */
    if (st_non_zero > 0) {
        for (size_t cat = 0; cat < st_non_zero; cat++) {
            if (perc_threshold[buffer_ix[cat]] > pct_unseen) {
                *new_is_outlier = true;
                is_outlier[buffer_ix[cat]] = -1;
            }
        }
    }
    if (*new_is_outlier && !(*found_outliers)) {
        *next_most_comm = buffer_perc[buffer_ix[st_non_zero]];
    }

}

/*    Check whether to consider any category as outlier, based on majority category and prior probabilties
*    
*    Function is to be applied to some subset of the data obtained by splitting by one or more columns.
*    For outliers before any split there is a separate function. This is an alternative to the "tail"
*    approach above which is more in line with GritBot.
*    
*    Parameters:
*    - categ_counts[ncateg] (in)
*        Counts of each category in the subset (including non-present categories).
*    - ncateg (in)
*        Number of categories for this column (including non-present categories).
*    - tot (in)
*        Number of rows in the subset.
*    - max_perc_outliers (in)
*        Model parameter. Default value is 0.01.
*    - prior_prob[ncateg] (in)
*        Proportions that each category had in the full data.
*    - z_outlier (in)
*        Model parameter. Default value is 8.0
*    - is_outlier[ncateg] (out)
*        Array where to define whether any category is an outlier. Values will be as follows:
*            (-1) -> Category had zero count, but would be an outlier if it appeared among this group
*              0  -> Category is not an outlier
*            (+1) -> Category is an outlier
*    - found_outliers (out)
*        Whether there were any outliers identified among the counts.
*    - new_is_outlier (out)
*        Whether any of the categories with zero count would be flagged as outlier if they appeared in this group.
*    - categ_maj (out)
*        Category to which the majority of the observations belong.
*/
void find_outlier_categories_by_maj(size_t categ_counts[], size_t ncateg, size_t tot, double max_perc_outliers,
                                    long double prior_prob[], double z_outlier, signed char is_outlier[],
                                    bool *found_outliers, bool *new_is_outlier, int *categ_maj)
{
    /* initialize parameters as needed */
    *found_outliers = false;
    *new_is_outlier = false;
    memset(is_outlier, 0, ncateg * sizeof(signed char));
    size_t max_outliers = (size_t) calculate_max_outliers((long double)tot, max_perc_outliers);
    long double tot_dbl = (long double) (tot + 1);
    size_t n_non_maj;
    long double thr_prop = (double)1 / square(z_outlier);

    /* check if any can be considered as outlier */
    size_t *ptr_maj = std::max_element(categ_counts, categ_counts + ncateg);
    *categ_maj = (int)(ptr_maj - categ_counts);
    n_non_maj = tot - *ptr_maj;
    if (n_non_maj > max_outliers)
        return;

    /* determine proportions and check for outlierness */
    long double n_non_maj_dbl = (long double) n_non_maj;
    for (size_t cat = 0; cat < ncateg; cat++) {

        if ((int)cat == *categ_maj) continue;

        if ( (n_non_maj_dbl / (tot_dbl * prior_prob[cat])) < thr_prop ) {
            if (categ_counts[cat]) {
                is_outlier[cat] = 1;
                *found_outliers = true;
            } else {
                is_outlier[cat] = -1;
                *new_is_outlier = true;
            }
        }
    }

    /* TODO: implement formula for flagging unsen categories (not in the sample, nor the full data) as outliers */
}


/*    Check whether to consider any category as outlier before splitting, based on prior counts
*    
*    Follows very rough criteria: there can be at most 1-3 outliers depending on size of dataset,
*    and the next most common category must have a count of at least 250.
*    
*    Parameters:
*    - categ_counts[ncateg] (in)
*        Frequencies of each category in the full data.
*    - ncateg (in)
*        Number of categories with non-zero count.
*    - tot (in)
*        Number of rows.
*    - is_outlier[ncateg] (out)
*        Array indicating whether any category is outlier (0 = non-outlier, 1 = outlier).
*    - next_most_comm (out)
*        Proportion of the least common non-outlier category.
*/
bool find_outlier_categories_no_cond(size_t categ_counts[], size_t ncateg, size_t tot,
                                     signed char is_outlier[], double *next_most_comm)
{
    /* if sample is too small, don't flag any as outliers */
    if (tot < 1000) return false;

    /* set a very low outlier threshold with a hard limit of 3 */
    size_t max_outliers = (tot < 10000)? 1 : ((tot < 100000)? 2 : 3);

    /* will only consider a category as outlier if the next most common is very common */
    size_t max_next_most_comm = 250;

    /* look if there's any category meeting the first condition and none meeting the second one */
    bool has_outlier_cat = false;
    memset(is_outlier, 0, sizeof(signed char) * ncateg);
    for (size_t cat = 0; cat < ncateg; cat++) {
        if (categ_counts[cat] > max_outliers && categ_counts[cat] < max_next_most_comm) {
            has_outlier_cat = false;
            break;
        }

        if (categ_counts[cat] > 0 && categ_counts[cat] <= max_outliers) {
            /* can only have 1 outlier category in the whole column */
            if (has_outlier_cat) { has_outlier_cat = false; break; }

            has_outlier_cat = true;
            is_outlier[cat] = 1;
        }

    }

    /* if outlier is found, find next most common frequency for printed statistics */
    if (has_outlier_cat) {
        size_t next_most_comm_cat = INT_MAX;
        for (size_t cat = 0; cat < ncateg; cat++) {
            if (categ_counts[cat] > 0 && !is_outlier[cat]) {
                next_most_comm_cat = std::min(next_most_comm_cat, categ_counts[cat]);
            }
        }
        *next_most_comm = (long double)next_most_comm_cat / (long double)tot;
    }

    return has_outlier_cat;
}
