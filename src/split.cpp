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


/* TODO: don't divide the gains by tot at every calculation as it makes it slower */

/* TODO: sorting here is the slowest thing, so it could be improved by using radix sort for categorical/ordinal and timsort for numerical */

/* TODO: columns that split by numeric should output the sum/sum_sq to pass it to the cluster functions, instead of recalculating them later */


void subset_to_onehot(size_t ix_arr[], size_t n_true, size_t n_tot, signed char onehot[])
{
    memset(onehot, 0, sizeof(bool) * n_tot);
    for (size_t i = 0; i <= n_true; i++) onehot[ix_arr[i]] = 1;
}

size_t move_zero_count_to_front(size_t *restrict cat_sorted, size_t *restrict cat_cnt, size_t ncat_x)
{
    size_t temp_ix;
    size_t st_cat = 0;
    for (size_t cat = 0; cat < ncat_x; cat++) {
        if (cat_cnt[cat] == 0) {
            temp_ix = cat_sorted[st_cat];
            cat_sorted[st_cat] = cat;
            cat_sorted[cat] = temp_ix;
            st_cat++;
        }
    }
    return st_cat;
}

void flag_zero_counts(signed char split_subset[], size_t buffer_cat_cnt[], size_t ncat_x)
{
    for (size_t cat = 0; cat < ncat_x; cat++)
        if (buffer_cat_cnt[cat] == 0) split_subset[cat] = -1;
}

long double calc_sd(size_t cnt, long double sum, long double sum_sq)
{
    if (cnt < 3) return 0;
    return sqrtl( (sum_sq - (square(sum) / (long double) cnt) + SD_REG) / (long double) (cnt - 1) );
}

long double calc_sd(NumericBranch &branch)
{
    if (branch.cnt < 3) return 0;
    return sqrtl((branch.sum_sq - (square(branch.sum) / (long double) branch.cnt) + SD_REG) / (long double) (branch.cnt - 1));
}

long double calc_sd(size_t ix_arr[], double *restrict x, size_t st, size_t end, double *restrict mean)
{
    long double running_mean = 0;
    long double running_ssq  = 0;
    long double mean_prev    = x[ix_arr[st]];
    double xval;
    for (size_t row = st; row <= end; row++) {
        xval = x[ix_arr[row]];
        running_mean += (xval - running_mean) / (long double)(row - st + 1);
        running_ssq  += (xval - running_mean) * (xval - mean_prev);
        mean_prev     = running_mean;
    }
    *mean = (double) running_mean;
    return sqrtl(running_ssq / (long double)(end - st));

}

long double numeric_gain(NumericSplit &split_info, long double tot_sd)
{
    long double tot = (long double)(split_info.NA_branch.cnt + split_info.left_branch.cnt + split_info.right_branch.cnt);
    long double residual = 
        ((long double) split_info.NA_branch.cnt)    * calc_sd(split_info.NA_branch)   +
        ((long double) split_info.left_branch.cnt)  * calc_sd(split_info.left_branch) +
        ((long double) split_info.right_branch.cnt) * calc_sd(split_info.right_branch);

    return tot_sd - (residual / tot);
}

long double numeric_gain(long double tot_sd, long double info_left, long double info_right, long double info_NA, long double cnt)
{
    return tot_sd - (info_left + info_right + info_NA) / cnt;
}

long double total_info(size_t categ_counts[], size_t ncat)
{
    long double s   = 0;
    size_t tot = 0;
    for (size_t cat = 0; cat < ncat; cat++) {
        if (categ_counts[cat] > 0) {
            s   += (long double)categ_counts[cat] * logl((long double)categ_counts[cat]);
            tot += categ_counts[cat];
        }
    }
    if (tot == 0) return 0;
    return (long double)tot * logl((long double)tot) - s;
}

long double total_info(size_t categ_counts[], size_t ncat, size_t tot)
{
    if (tot == 0) return 0;
    long double s   = 0;
    for (size_t cat = 0; cat < ncat; cat++) {
        if (categ_counts[cat] > 1) {
            s   += (long double)categ_counts[cat] * logl((long double)categ_counts[cat]);
        }
    }
    return (long double) tot * logl((long double) tot) - s;
    /* tot = sum(categ_counts[]) */
}

long double total_info(size_t *restrict ix_arr, int *restrict x, size_t st, size_t end, size_t ncat, size_t *restrict buffer_cat_cnt)
{
    long double info = (long double)(end - st + 1) * logl((long double)(end - st + 1));
    memset(buffer_cat_cnt, 0, ncat * sizeof(size_t));
    for (size_t row = st; row <= end; row++) {
        buffer_cat_cnt[ x[ix_arr[row]] ]++;
    }
    for (size_t cat = 0; cat < ncat; cat++) {
        if (buffer_cat_cnt[cat] > 1) {
            info -= (long double)buffer_cat_cnt[cat] * logl((long double)buffer_cat_cnt[cat]);
        }
    }
    return info;
}

long double categ_gain(CategSplit split_info, long double base_info)
{
    return (
            base_info -
            total_info(split_info.NA_branch,    split_info.ncat, split_info.size_NA) -
            total_info(split_info.left_branch,  split_info.ncat, split_info.size_left) -
            total_info(split_info.right_branch, split_info.ncat, split_info.size_right)
           ) / (long double) split_info.tot ;
}

long double categ_gain(size_t *restrict categ_counts, size_t ncat, size_t *restrict ncat_col, size_t maxcat, long double base_info, size_t tot)
{
    long double info = 0;
    for (size_t cat = 0; cat < ncat; cat++) {
        if (categ_counts[cat] > 0) {
            info += total_info(categ_counts + cat * maxcat, ncat_col[cat]);
        }
    }

    /* last entry in the array corresponds to NA values */
    if (categ_counts[ncat] > 0) {
        info += total_info(categ_counts + ncat * maxcat, ncat_col[ncat]);
    }

    return (base_info - info) / (long double) tot;
}

long double categ_gain_from_split(size_t *restrict ix_arr, int *restrict x, size_t st, size_t st_non_na, size_t split_ix, size_t end,
                                  size_t ncat, size_t *restrict buffer_cat_cnt, long double base_info)
{
    long double gain = base_info;
    memset(buffer_cat_cnt, 0, ncat * sizeof(size_t));
    if (st_non_na > st) {
        for (size_t row = st; row < st_non_na; row++) {
            buffer_cat_cnt[ x[ix_arr[row]] ]++;
        }
        gain -= total_info(buffer_cat_cnt, ncat, st_non_na - st);
        memset(buffer_cat_cnt, 0, ncat * sizeof(size_t));
    }

    for (size_t row = st_non_na; row < split_ix; row++) {
        buffer_cat_cnt[ x[ix_arr[row]] ]++;
    }
    gain -= total_info(buffer_cat_cnt, ncat, split_ix - st_non_na);
    memset(buffer_cat_cnt, 0, ncat * sizeof(size_t));

    for (size_t row = split_ix; row <= end; row++) {
        buffer_cat_cnt[ x[ix_arr[row]] ]++;
    }
    gain -= total_info(buffer_cat_cnt, ncat, end - split_ix + 1);

    return gain / (long double)(end - st + 1);
}

/*    Calculate gain from splitting a numeric column by another numeric column
*    
*    Function splits into buckets (NA, <= threshold, > threshold)
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Numeric column from which a split predicting 'y' will be calculated.
*    - y[n] (in)
*        Numeric column whose distribution wants to be split by 'x'.
*        Must not contain missing values.
*    - sd_y (in)
*        Standard deviation of 'y' between the indices considered here.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - min_size (in)
*        Minimum number of elements that can be in a split.
*    - buffer_sd[n] (in)
*        Buffer where to write temporary sd/information at each split point in a first pass.
*    - gain (out)
*        Gain calculated on the best split found. If no split is possible, will return -Inf.
*    - split_point (out)
*        Threshold for splitting on values of 'x'. If no split is posible, will return -Inf.
*    - split_left (out)
*        Index at which the data is split between the two branches (includes last from left branch).
*    - split_NA (out)
*        Index at which the NA data is separated from the other branches
*/
void split_numericx_numericy(size_t *restrict ix_arr, size_t st, size_t end, double *restrict x, double *restrict y,
                             long double sd_y, bool has_na, size_t min_size, bool take_mid, long double *restrict buffer_sd,
                             long double *restrict gain, double *restrict split_point, size_t *restrict split_left, size_t *restrict split_NA)
{

    *gain = -HUGE_VAL;
    *split_point = -HUGE_VAL;
    size_t st_non_na;
    long double this_gain;
    long double cnt_dbl = (long double)(end - st + 1);
    long double running_mean = 0;
    long double running_ssq  = 0;
    long double mean_prev    = 0;
    double xval;
    long double info_left;
    long double info_NA = 0;

    /* check that there are enough observations for a split */
    if ((end - st + 1) < (2 * min_size)) return;

    /* move all NAs of X to the front */
    if (has_na) {
        st_non_na = move_NAs_to_front(ix_arr, x, st, end, false);
    } else { st_non_na = st; }
    *split_NA = st_non_na;

    /* assign NAs to their own branch */
    if (st_non_na > st) {

        /* first check that it's still possible to split */
        if ((end - st_non_na + 1) < (2 * min_size)) return;

        info_NA = (long double)(st_non_na - st) * calc_sd(ix_arr, y, st, st_non_na-1, &xval); /* last arg is not used */
    }

    /* sort the remaining non-NA values in ascending order */
    std::sort(ix_arr + st_non_na, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});

    /* calculate SD*N backwards first, then forwards */
    mean_prev = y[ix_arr[end]];
    for (size_t i = end; i >= st_non_na; i--) {
        xval = y[ix_arr[i]];
        running_mean += (xval - running_mean) / (long double)(end - i + 1);
        running_ssq  += (xval - running_mean) * (xval - mean_prev);
        mean_prev     =  running_mean;
        buffer_sd[i]  = (long double)(end - i + 1) * sqrtl(running_ssq / (long double)(end - i));
        /* could also avoid div by n-1, would be faster */

        if (i == st_non_na) break; /* be aware unsigned integer overflow */
    }

    /* look for the best split point, by moving one observation at a time to the left branch*/
    running_mean = 0;
    running_ssq  = 0;
    mean_prev    = y[ix_arr[st_non_na]];
    for (size_t i = st_non_na; i <= (end - min_size); i++) {
        xval = y[ix_arr[i]];
        running_mean += (xval - running_mean) / (long double)(i - st_non_na + 1);
        running_ssq  += (xval - running_mean) * (xval - mean_prev);
        mean_prev     =  running_mean;

        /* check that split meets minimum criteria (size on right branch is controlled in loop condition) */
        if ((i - st_non_na + 1) < min_size) continue;

        /* check that value is not repeated next -- note that condition in loop prevents out-of-bounds access */
        if (x[ix_arr[i]] == x[ix_arr[i + 1]]) continue;

        /* evaluate gain at this split point */
        info_left = (long double)(i - st_non_na + 1) * sqrtl(running_ssq / (long double)(i - st_non_na));
        this_gain = numeric_gain(sd_y, info_left, buffer_sd[i + 1], info_NA, cnt_dbl);
        if (this_gain > *gain) {
            *gain = this_gain;
            *split_point = take_mid? (avg_between(x[ix_arr[i]], x[ix_arr[i + 1]])) : (x[ix_arr[i]]);
            *split_left = i;
        }
    }
}

/*    Calculate gain from splitting a numeric column by a categorical column
*    
*    Function splits into two subsets + NAs on their own branch
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Categorical column from which a split predicting 'y' will be calculated.
*        Missing values should be encoded as negative integers.
*    - y[n] (in)
*        Numeric column whose distribution wants to be split by 'x'.
*        Must not contain missing values.
*    - sd_y (in)
*        Standard deviation of 'y' between the indices considered here.
*    - x_is_ordinal (in)
*        Whether the 'x' column has ordered categories, in which case the split will be a
*        <= that respects this order.
*    - ncat_x (in)
*        Number of categories in 'x' (excluding NA).
*    - buffer_cat_cnt[ncat_x + 1] (temp)
*        Array where temporary data for each category will be written into.
*        Must have one additional entry anove the number of categories to account for NAs.
*    - buffer_cat_sum[ncat_x + 1] (temp)
*        See above.
*    - buffer_cat_sum_sq[ncat_x + 1] (temp)
*        See above.
*    - buffer_cat_sorted[ncat_x] (temp)
*        See above. This one doesn't need an extra entry.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - min_size (in)
*        Minimum number of elements that can be in a split.
*    - gain (out)
*        Gain calculated on the best split found. If no split is possible, will return -Inf.
*    - split_subset[ncat_x] (out)
*        Array that will indicate which categories go into the left branch in the chosen split.
*        (value of 1 means it's on the left branch, 0 in the right branch, -1 not applicable)
*    - split_point (out)
*        Split level for ordinal X variables (left branch is <= this)
*/
void split_categx_numericy(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, double *restrict y, long double sd_y, double ymean,
                           bool x_is_ordinal, size_t ncat_x, size_t *restrict buffer_cat_cnt, long double *restrict buffer_cat_sum,
                           long double *restrict buffer_cat_sum_sq, size_t *restrict buffer_cat_sorted,
                           bool has_na, size_t min_size, long double *gain, signed char *restrict split_subset, int *restrict split_point)
{

    /* output parameters and variables to use */
    *gain = -HUGE_VAL;
    long double this_gain;
    NumericSplit split_info;
    size_t st_cat = 0;
    double sd_y_d = (double) sd_y;

    /* reset the buffers */
    memset(split_subset,      0, sizeof(signed char)   *  ncat_x);
    memset(buffer_cat_cnt,    0, sizeof(size_t) * (ncat_x + 1));
    memset(buffer_cat_sum,    0, sizeof(long double) * (ncat_x + 1));
    memset(buffer_cat_sum_sq, 0, sizeof(long double) * (ncat_x + 1));

    /* calculate summary info for each category */
    if (has_na) {

        for (size_t i = st; i <= end; i++) {

            /* NAs are encoded as negative integers, and go at the last slot */
            if (x[ix_arr[i]] < 0) {
                buffer_cat_cnt[ncat_x]++;
                buffer_cat_sum[ncat_x]    += z_score(y[ix_arr[i]], ymean, sd_y_d);
                buffer_cat_sum_sq[ncat_x] += square(z_score(y[ix_arr[i]], ymean, sd_y_d));
            } else {
                buffer_cat_cnt[ x[ix_arr[i]] ]++;
                buffer_cat_sum[ x[ix_arr[i]] ]    += z_score(y[ix_arr[i]], ymean, sd_y_d);
                buffer_cat_sum_sq[ x[ix_arr[i]] ] += square(z_score(y[ix_arr[i]], ymean, sd_y_d));
            }
        }

    } else {

        buffer_cat_cnt[ncat_x] = 0;
        for (size_t i = st; i <= end; i++) {
            buffer_cat_cnt[ x[ix_arr[i]] ]++;
            buffer_cat_sum[ x[ix_arr[i]] ]    += z_score(y[ix_arr[i]], ymean, sd_y_d);
            buffer_cat_sum_sq[ x[ix_arr[i]] ] += square(z_score(y[ix_arr[i]], ymean, sd_y_d));
        }

    }

    /* set NAs to their own branch */
    if (buffer_cat_cnt[ncat_x] > 0) {
        split_info.NA_branch = {buffer_cat_cnt[ncat_x], buffer_cat_sum[ncat_x], buffer_cat_sum_sq[ncat_x]};
    }

    /* easy case: binary split (only one possible split point) */
    if (ncat_x == 2) {

        /* must still meet minimum size requirements */
        if (buffer_cat_cnt[0] < min_size || buffer_cat_cnt[1] < min_size) return;

        split_info.left_branch = {buffer_cat_cnt[0], buffer_cat_sum[0], buffer_cat_sum_sq[0]};
        split_info.right_branch = {buffer_cat_cnt[1], buffer_cat_sum[1], buffer_cat_sum_sq[1]};
        *gain = numeric_gain(split_info, 1.0) * sd_y;
        split_subset[0] = 1;
    }

    /* subset and ordinal splits */
    else {

        /* put all the categories on the right branch */
        for (size_t cat = 0; cat < ncat_x; cat++) {
            split_info.right_branch.cnt    += buffer_cat_cnt[cat];
            split_info.right_branch.sum    += buffer_cat_sum[cat];
            split_info.right_branch.sum_sq += buffer_cat_sum_sq[cat];
        }

        /* if it's an ordinal variable, must respect the order */
        for (size_t cat = 0; cat < ncat_x; cat++) buffer_cat_sorted[cat] = cat;

        if (!x_is_ordinal) {
            /* otherwise, sort the categories according to their mean of y */

            /* first remove zero-counts */
            st_cat = move_zero_count_to_front(buffer_cat_sorted, buffer_cat_cnt, ncat_x);

            /* then sort */
            std::sort(buffer_cat_sorted + st_cat, buffer_cat_sorted + ncat_x,
                      [&buffer_cat_sum, &buffer_cat_cnt](const size_t a, const size_t b)
                      {
                          return (buffer_cat_sum[a] / (long double) buffer_cat_cnt[a]) >
                                 (buffer_cat_sum[b] / (long double) buffer_cat_cnt[b]);
                      });
        }

        /* try moving each category to the left branch in the given order */
        for (size_t cat = st_cat; cat < ncat_x; cat++) {
            split_info.right_branch.cnt    -= buffer_cat_cnt[ buffer_cat_sorted[cat] ];
            split_info.right_branch.sum    -= buffer_cat_sum[ buffer_cat_sorted[cat] ];
            split_info.right_branch.sum_sq -= buffer_cat_sum_sq[ buffer_cat_sorted[cat] ];

            split_info.left_branch.cnt     += buffer_cat_cnt[ buffer_cat_sorted[cat] ];
            split_info.left_branch.sum     += buffer_cat_sum[ buffer_cat_sorted[cat] ];
            split_info.left_branch.sum_sq  += buffer_cat_sum_sq[ buffer_cat_sorted[cat] ];

            /* see if it meets minimum split sizes */
            if (split_info.left_branch.cnt < min_size || split_info.right_branch.cnt < min_size) continue;

            /* calculate the gain */
            this_gain = numeric_gain(split_info, 1.0);
            if (this_gain > *gain) {
                *gain = this_gain * sd_y;
                if (!x_is_ordinal)
                    subset_to_onehot(buffer_cat_sorted, cat, ncat_x, split_subset);
                else
                    *split_point = (int) cat;
            }
        }

        /* if it's categorical, set the non-present categories to -1 */
        if (!is_na_or_inf(*gain) && !x_is_ordinal) flag_zero_counts(split_subset, buffer_cat_cnt, ncat_x);

    }

}



/*    Calculate gain from splitting a categorical column by a numeric column
*    
*    Function splits into two subsets + NAs on their own branch
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Numerical column from which a split predicting 'y' will be calculated.
*    - y[n] (in)
*        Categorical column whose distributions are to be split by 'x'.
*        Must not contain missing values (which are encoded as negative integers).
*    - ncat_y (in)
*        Number of categories in 'y' (excluding NAs, which are encoded as negative integers).
*    - base_info (in)
*        Base information for the 'y' counts before splitting.
*        (:= N*log(N) - sum_i..m N_i*log(N_i))
*    - buffer_cat_cnt[ncat_y * 3] (temp)
*        Array where temporary data for each category will be written into.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - min_size (in)
*        Minimum number of elements that can be in a split.
*    - gain (out)
*        Gain calculated on the best split found. If no split is possible, will return -Inf.
*    - split_point (out)
*        Threshold for splitting on values of 'x'. If no split is posible, will return -Inf.
*    - split_left (out)
*        Index at which the data is split between the two branches (includes last from left branch).
*    - split_NA (out)
*        Index at which the NA data is separated from the other branches
*/
void split_numericx_categy(size_t *restrict ix_arr, size_t st, size_t end, double *restrict x, int *restrict y,
                           size_t ncat_y, long double base_info, size_t *restrict buffer_cat_cnt,
                           bool has_na, size_t min_size, bool take_mid, long double *restrict gain, double *restrict split_point,
                           size_t *restrict split_left, size_t *restrict split_NA)
{
    *gain = -HUGE_VAL;
    *split_point = -HUGE_VAL;
    size_t st_non_na;
    long double this_gain;
    CategSplit split_info;
    split_info.ncat = ncat_y;
    split_info.tot = end - st + 1;

    /* check that there are enough observations for a split */
    if ((end - st + 1) < (2 * min_size)) return;

    /* will divide into 3 branches: NA, <= p, > p */
    memset(buffer_cat_cnt, 0, 3 * ncat_y * sizeof(size_t));
    split_info.NA_branch    = buffer_cat_cnt;
    split_info.left_branch  = buffer_cat_cnt + ncat_y;
    split_info.right_branch = buffer_cat_cnt + 2 * ncat_y;

    /* move all NAs of X to the front */
    if (has_na) {
        st_non_na = move_NAs_to_front(ix_arr, x, st, end, false);
    } else { st_non_na = st; }
    *split_NA = st_non_na;

    /* assign NAs to their own branch */
    split_info.size_NA = st_non_na - st;
    if (st_non_na > st) {

        /* first check that it's still possible to split */
        if ((end - st_non_na + 1) < (2 * min_size)) return;

        for (size_t i = st; i < st_non_na; i++) split_info.NA_branch[ y[ix_arr[i]] ]++;
    }

    /* sort the remaining non-NA values in ascending order */
    std::sort(ix_arr + st_non_na, ix_arr + end + 1, [&x](const size_t a, const size_t b){return x[a] < x[b];});

    /* put all observations on the right branch */
    for (size_t i = st_non_na; i <= end; i++) split_info.right_branch[ y[ix_arr[i]] ]++;

    /* look for the best split point, by moving one observation at a time to the left branch*/
    for (size_t i = st_non_na; i <= (end - min_size); i++) {
        split_info.right_branch[ y[ix_arr[i]] ]--;
        split_info.left_branch [ y[ix_arr[i]] ]++;
        split_info.size_left  = i - st_non_na + 1;
        split_info.size_right = end - i;

        /* check that split meets minimum criteria (size on right branch is controlled in loop condition) */
        if (split_info.size_left < min_size) continue;

        /* check that value is not repeated next -- note that condition in loop prevents out-of-bounds access */
        if (x[ix_arr[i]] == x[ix_arr[i + 1]]) continue;

        /* evaluate gain at this split point */
        this_gain = categ_gain(split_info, base_info);
        if (this_gain > *gain) {
            *gain = this_gain;
            *split_point = take_mid? (avg_between(x[ix_arr[i]], x[ix_arr[i + 1]])) : (x[ix_arr[i]]);
            *split_left = i;
        }
    }
}

/*    Calculate gain from splitting a categorical column by an ordinal column
*    
*    Function splits into two subsets + NAs on their own branch
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Ordinal column from which a split predicting 'y' will be calculated.
*        Missing values must be encoded as negative integers.
*    - y[n] (in)
*        Categorical column whose distributions are to be split by 'x'.
*        Must not contain missing values (which are encoded as negative integers).
*    - ncat_y (in)
*        Number of categories in 'y' (excluding NAs, which are encoded as negative integers).
*    - ncat_x (in)
*        Number of categories in 'x' (excluding NAs, which are encoded as negative integers).
*    - base_info (in)
*        Base information for the 'y' counts before splitting.
*        (:= N*log(N) - sum_i..m N_i*log(N_i))
*    - buffer_cat_cnt[ncat_y * 3] (temp)
*        Array where temporary data for each category will be written into.
*    - buffer_crosstab[ncat_x * ncat_y] (temp)
*        See above.
*    - buffer_ord_cnt[ncat_x] (temp)
*        See above.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - min_size (in)
*        Minimum number of elements that can be in a split.
*    - gain (out)
*        Gain calculated on the best split found. If no split is possible, will return -Inf.
*    - split_point (out)
*        Threshold for splitting on values of 'x'. If no split is posible, will return -1.
*/
void split_ordx_categy(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                       size_t ncat_y, size_t ncat_x, long double base_info,
                       size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab, size_t *restrict buffer_ord_cnt,
                       bool has_na, size_t min_size, long double *gain, int *split_point)
{
    *gain = -HUGE_VAL;
    *split_point = -1;
    size_t st_non_na;
    long double this_gain;
    CategSplit split_info;
    split_info.ncat = ncat_y;
    split_info.tot = end - st + 1;

    /* check that there are enough observations for a split */
    if ((end - st + 1) < (2 * min_size)) return;

    /* will divide into 3 branches: NA, <= p, > p */
    memset(buffer_cat_cnt, 0, 3 * ncat_y * sizeof(size_t));
    split_info.NA_branch    = buffer_cat_cnt;
    split_info.left_branch  = buffer_cat_cnt + ncat_y;
    split_info.right_branch = buffer_cat_cnt + 2 * ncat_y;

    /* move all NAs of X to the front */
    if (has_na) {
        st_non_na = move_NAs_to_front(ix_arr, x, st, end);
    } else { st_non_na = st; }

    /* assign NAs to their own branch */
    split_info.size_NA = st_non_na - st;
    if (st_non_na > st) {

        /* first check that it's still possible to split */
        if ((end - st_non_na + 1) < (2 * min_size)) return;

        for (size_t i = st; i < st_non_na; i++) split_info.NA_branch[ y[ix_arr[i]] ]++;
    }

    /* calculate cross-table on the non-missing cases, and put all observations in the right branch */
    memset(buffer_crosstab, 0, ncat_y * ncat_x * sizeof(size_t));
    memset(buffer_ord_cnt,  0, ncat_x * sizeof(size_t));
    for (size_t i = st_non_na; i <= end; i++) {
        buffer_crosstab[ y[ix_arr[i]] + ncat_y * x[ix_arr[i]] ]++;
        buffer_ord_cnt [ x[ix_arr[i]] ]++;
        split_info.right_branch[ y[ix_arr[i]] ]++;
    }
    split_info.size_right = end - st_non_na + 1;
    split_info.size_left  = 0;

    /* look for the best split point, by moving one observation at a time to the left branch*/
    for (size_t ord_cat = 0; ord_cat < (ncat_x - 1); ord_cat++) {
        
        for (size_t moved_cat = 0; moved_cat < ncat_y; moved_cat++) {
            split_info.right_branch[ moved_cat ] -= buffer_crosstab[ moved_cat + ncat_y * ord_cat ];
            split_info.left_branch [ moved_cat ] += buffer_crosstab[ moved_cat + ncat_y * ord_cat ];
        }
        split_info.size_right -= buffer_ord_cnt[ord_cat];
        split_info.size_left  += buffer_ord_cnt[ord_cat];

        /* check that split meets minimum criteria */
        if (split_info.size_left < min_size || split_info.size_right < min_size) continue;

        /* evaluate gain at this split point */
        this_gain = categ_gain(split_info, base_info);
        if (this_gain > *gain) {
            *gain = this_gain;
            *split_point = ord_cat;
        }
    }
}


/*    Calculate gain from splitting a binary column by a categorical column
*    
*    Function splits into two subsets + NAs on their own branch
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Categorical column from which a split predicting 'y' will be calculated.
*        Missing values must be encoded as negative integers.
*    - y[n] (in)
*        Binary column whose distributions are to be split by 'x'.
*        Must not contain missing values (which are encoded as negative integers).
*    - ncat_x (in)
*        Number of categories in 'x' (excluding NAs, which are encoded as negative integers).
*    - base_info (in)
*        Base information for the 'y' counts before splitting.
*        (:= N*log(N) - sum_i..m N_i*log(N_i))
*    - buffer_cat_cnt[ncat_x] (temp)
*        Array where temporary data for each category will be written into.
*    - buffer_crosstab[2 * ncat_x] (temp)
*        See above.
*    - buffer_cat_sorted[ncat_x] (temp)
*        See above.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - min_size (in)
*        Minimum number of elements that can be in a split.
*    - gain (out)
*        Gain calculated on the best split found. If no split is possible, will return -Inf.
*    - split_subset[ncat_x] (out)
*        Array that will indicate which categories go into the left branch in the chosen split.
*        (value of 1 means it's on the left branch, 0 in the right branch, -1 not applicable)
*/
void split_categx_biny(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                       size_t ncat_x, long double base_info,
                       size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab, size_t *restrict buffer_cat_sorted,
                       bool has_na, size_t min_size, long double *gain, signed char *restrict split_subset)
{
    *gain = -HUGE_VAL;
    size_t st_non_na;
    long double this_gain;
    size_t buffer_fixed_size[6] = {0};
    CategSplit split_info;
    size_t st_cat;
    split_info.ncat = 2;
    split_info.tot = end - st + 1;

    /* check that there are enough observations for a split */
    if ((end - st + 1) < (2 * min_size)) return;

    /* will divide into 3 branches: NA, <= p, > p */
    split_info.NA_branch    = buffer_fixed_size;
    split_info.left_branch  = buffer_fixed_size + 2;
    split_info.right_branch = buffer_fixed_size + 2 * 2;

    /* move all NAs of X to the front */
    if (has_na) {
        st_non_na = move_NAs_to_front(ix_arr, x, st, end);
    } else { st_non_na = st; }

    /* assign NAs to their own branch */
    split_info.size_NA = st_non_na - st;
    if (st_non_na > st) {

        /* first check that it's still possible to split */
        if ((end - st_non_na + 1) < (2 * min_size)) return;

        for (size_t i = st; i < st_non_na; i++) split_info.NA_branch[ y[ix_arr[i]] ]++;
    }

    /* calculate cross-table on the non-missing cases, and put all observations in the right branch */
    memset(buffer_crosstab, 0, 2 * ncat_x * sizeof(size_t));
    memset(buffer_cat_cnt,  0, ncat_x * sizeof(size_t));
    for (size_t i = st_non_na; i <= end; i++) {
        buffer_crosstab[ y[ix_arr[i]] + 2 * x[ix_arr[i]] ]++;
        buffer_cat_cnt [ x[ix_arr[i]] ]++;
        split_info.right_branch[ y[ix_arr[i]] ]++;
    }
    split_info.size_right = end - st_non_na + 1;
    split_info.size_left  = 0;

    /* sort the categories according to their mean of y */
    for (size_t cat = 0; cat < ncat_x; cat++) buffer_cat_sorted[cat] = cat;
    st_cat = move_zero_count_to_front(buffer_cat_sorted, buffer_cat_cnt, ncat_x);
    std::sort(buffer_cat_sorted + st_cat, buffer_cat_sorted + ncat_x,
              [&buffer_crosstab, &buffer_cat_cnt](const size_t a, const size_t b)
              {
                  return ((long double) buffer_crosstab[2 * a] / (long double) buffer_cat_cnt[a]) >
                         ((long double) buffer_crosstab[2 * b] / (long double) buffer_cat_cnt[b]);
              });

    /* look for the best split subset, by moving one category at a time to the left branch*/
    for (size_t cat = st_cat; cat < (ncat_x - 1); cat++) {

        split_info.right_branch[0] -= buffer_crosstab[2 * buffer_cat_sorted[cat]];
        split_info.right_branch[1] -= buffer_crosstab[2 * buffer_cat_sorted[cat] + 1];
        split_info.left_branch [0] += buffer_crosstab[2 * buffer_cat_sorted[cat]];
        split_info.left_branch [1] += buffer_crosstab[2 * buffer_cat_sorted[cat] + 1];
        split_info.size_right      -= buffer_cat_cnt [buffer_cat_sorted[cat]];
        split_info.size_left       += buffer_cat_cnt [buffer_cat_sorted[cat]];

        /* check that split meets minimum criteria */
        if (split_info.size_left < min_size || split_info.size_right < min_size) continue;

        /* evaluate gain at this split point */
        this_gain = categ_gain(split_info, base_info);
        if (this_gain > *gain) {
            *gain = this_gain;
            subset_to_onehot(buffer_cat_sorted, cat, ncat_x, split_subset);
        }
    }
    if (!is_na_or_inf(*gain)) flag_zero_counts(split_subset, buffer_cat_cnt, ncat_x);
}


/*    Calculate gain from splitting a categorical columns by another categorical column
*    
*    Function splits into one branch per category of 'x'
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Categorical column from which a split predicting 'y' will be calculated.
*        Missing values must be encoded as negative integers.
*    - y[n] (in)
*        Categorical column whose distributions are to be split by 'x'.
*        Must not contain missing values (which are encoded as negative integers).
*    - ncat_x (in)
*        Number of categories in 'x' (excluding NAs, which are encoded as negative integers).
*    - ncat_y (in)
*        Number of categories in 'y'.
*    - base_info (in)
*        Base information for the 'y' counts before splitting.
*        (:= N*log(N) - sum_i..m N_i*log(N_i))
*    - buffer_cat_cnt[ncat_x + 1] (temp)
*        Array where temporary data for each category will be written into.
*    - buffer_crosstab[(ncat_x + 1) * ncat_y] (temp)
*        See above.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - gain (out)
*        Gain calculated on the split. If no split is possible, will return -Inf.
*/
void split_categx_categy_separate(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                                  size_t ncat_x, size_t ncat_y, long double base_info,
                                  size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab,
                                  bool has_na, size_t min_size, long double *gain)
{
    long double this_gain = 0;
    size_t st_non_na;

    /* move all NAs of X to the front */
    if (has_na) {
        st_non_na = move_NAs_to_front(ix_arr, x, st, end);
    } else { st_non_na = st; }

    /* calculate cross-table on the non-missing cases */
    memset(buffer_crosstab, 0, ncat_y * (ncat_x + 1) * sizeof(size_t));
    memset(buffer_cat_cnt,  0, (ncat_x + 1) * sizeof(size_t));
    for (size_t i = st_non_na; i <= end; i++) {
        buffer_crosstab[ y[ix_arr[i]] + ncat_y * x[ix_arr[i]] ]++;
        buffer_cat_cnt [ x[ix_arr[i]] ]++;
    }

    /* if no category meets the minimum split size, end here */
    if (*std::max_element(buffer_cat_cnt, buffer_cat_cnt + (ncat_x + 1)) < min_size) {
        *gain = -HUGE_VAL;
        return;
    }

    /* calculate gain for splitting at each category */
    for (size_t cat = 0; cat < ncat_x; cat++) {
        this_gain += total_info(buffer_crosstab + cat * ncat_y, ncat_y, buffer_cat_cnt[cat]);
    }

    /* add the split on missing x */
    if (st_non_na > st) {
        for (size_t i = st; i < st_non_na; i++) {
            buffer_crosstab[ y[ix_arr[i]] + ncat_y * ncat_x ]++;
            buffer_cat_cnt [ ncat_x ]++;
        }
        this_gain += total_info(buffer_crosstab + ncat_x * ncat_y, ncat_y, buffer_cat_cnt[ncat_x]);
    }

    /* return calculated gain */
    *gain = (base_info - this_gain) / (long double) (end - st + 1);
}


/*    Calculate gain from splitting a categorical column by another categorical column
*    
*    Function splits into two subsets + NAs on their own branch
*    
*    Parameters
*    - ix_arr[n] (in)
*        Array containing the indices at which 'x' and 'y' can be accessed, considering only the
*        elements between st and end (i.e. ix_arr[st:end], inclusive of both ends)
*        (Note: will be modified in-place)
*    - st (in)
*        See above.
*    - end (in)
*        See above.
*    - x[n] (in)
*        Categorical column from which a split predicting 'y' will be calculated.
*        Missing values must be encoded as negative integers.
*    - y[n] (in)
*        Categorical column whose distributions are to be split by 'x'.
*        Must not contain missing values (which are encoded as negative integers).
*    - ncat_x (in)
*        Number of categories in 'x' (excluding NAs, which are encoded as negative integers).
*    - ncat_y (in)
*        Number of categories in 'x'.
*    - base_info (in)
*        Base information for the 'y' counts before splitting.
*        (:= N*log(N) - sum_i..m N_i*log(N_i))
*    - buffer_cat_cnt[ncat_x] (temp)
*        Array where temporary data for each category will be written into.
*    - buffer_crosstab[ncat_x * ncat_y] (temp)
*        See above.
*    - buffer_split[3 * ncat_y] (temp)
*        See above.
*    - has_na (in)
*        Whether 'x' can have missing values or not.
*    - min_size (in)
*        Minimum number of elements that can be in a split.
*    - gain (out)
*        Gain calculated on the best split found. If no split is possible, will return -Inf.
*    - split_subset[ncat_x] (out)
*        Array that will indicate which categories go into the left branch in the chosen split.
*        (value of 1 means it's on the left branch, 0 in the right branch, -1 not applicable)
*/
void split_categx_categy_subset(size_t *restrict ix_arr, size_t st, size_t end, int *restrict x, int *restrict y,
                                size_t ncat_x, size_t ncat_y, long double base_info,
                                size_t *restrict buffer_cat_cnt, size_t *restrict buffer_crosstab, size_t *restrict buffer_split,
                                bool has_na, size_t min_size, long double *gain, signed char *restrict split_subset)
{
    *gain = -HUGE_VAL;
    long double this_gain;
    size_t best_subset;
    CategSplit split_info;
    split_info.tot = end - st + 1;
    split_info.ncat = ncat_y;
    size_t st_non_na;

    /* will divide into 3 branches: NA, within subset, outside subset */
    memset(buffer_split, 0, 3 * ncat_y * sizeof(size_t));
    split_info.NA_branch    = buffer_split;
    split_info.left_branch  = buffer_split + ncat_y;
    split_info.right_branch = buffer_split + 2 * ncat_y;

    /* move all NAs of X to the front */
    if (has_na) {
        st_non_na = move_NAs_to_front(ix_arr, x, st, end);
    } else { st_non_na = st; }
    split_info.size_NA = st_non_na - st;

    /* calculate cross-table */
    memset(buffer_crosstab, 0, ncat_y * ncat_x * sizeof(size_t));
    memset(buffer_cat_cnt,  0, ncat_x * sizeof(size_t));
    for (size_t i = st_non_na; i <= end; i++) {
        buffer_crosstab[ y[ix_arr[i]] + ncat_y * x[ix_arr[i]] ]++;
        buffer_cat_cnt [ x[ix_arr[i]] ]++;
    }
    if (st_non_na > st) {
        for (size_t i = st; i < st_non_na; i++) {
            split_info.NA_branch[ y[ix_arr[i]] ]++;
        }
    }

    /* put all categories on the right branch */
    memset(split_info.left_branch,   0, ncat_y * sizeof(size_t));
    memset(split_info.right_branch,  0, ncat_y * sizeof(size_t));
    split_info.size_left = 0;
    split_info.size_right = 0;
    for (size_t catx = 0; catx < ncat_x; catx++) {
        for (size_t caty = 0; caty < ncat_y; caty++) {
            split_info.right_branch[caty] += buffer_crosstab[caty + catx * ncat_y];
        }
        split_info.size_right += buffer_cat_cnt[catx];
    }

    /* TODO: don't loop over categories with zero-counts everywhere */

    /* do a brute-force search over all possible subset splits (there's [2^ncat_x - 2] of them) */
    size_t curr_exponent = 0;
    size_t last_bit;
    size_t ncomb = pow2(ncat_x) - 1;

    /* iteration is done by putting a category in the left branch if the bit at its
       position in the binary representation of the combination number is a 1 */
    /* TODO: this would be faster with a depth-first search routine */
    for (size_t combin = 1; combin < ncomb; combin++) {

        /* at each iteration, move the bits that differ from one branch to the other */
        /* note however than when there are few categories, it's actually faster to recalculate
           the counts based on the bitset -- this code however still follows this more "smart" way
           of moving cateogries when needed, which makes it slightly more scalable */

        /* at any given number, the bits can only vary up a certain bit from an increase by one,
           which can be obtained from calculating the maximum power of two that is smaller than
           the combination number */
        if (combin == pow2(curr_exponent)) {
            curr_exponent++;
            last_bit = (size_t) curr_exponent - 1;

            /* when this happens, this specific bit will change from a zero to a one,
               while the ones before will change from ones to zeros */
            for (size_t caty = 0; caty < ncat_y; caty++) {
                split_info.right_branch[caty] -= buffer_crosstab[caty + last_bit * ncat_y];
                split_info.left_branch [caty] += buffer_crosstab[caty + last_bit * ncat_y];
            }
            split_info.size_left  += buffer_cat_cnt[last_bit];
            split_info.size_right -= buffer_cat_cnt[last_bit];

            for (size_t catx = 0; catx < last_bit; catx++) {
                for (size_t caty = 0; caty < ncat_y; caty++) {
                    split_info.left_branch [caty] -= buffer_crosstab[caty + catx * ncat_y];
                    split_info.right_branch[caty] += buffer_crosstab[caty + catx * ncat_y];
                }
                split_info.size_left  -= buffer_cat_cnt[catx];
                split_info.size_right += buffer_cat_cnt[catx];
            }

        } else {

            /* in the regular case, just inspect the bits that come before the exponent in the current
               power of two that is less than the combination number, and see if a category needs to be moved */
            for (size_t catx = 0; catx < last_bit; catx++) {
                if (extract_bit(combin, catx) != extract_bit(combin - 1, catx)) {

                    if (extract_bit(combin - 1, catx)) {
                        for (size_t caty = 0; caty < ncat_y; caty++) {
                            split_info.left_branch [caty] -= buffer_crosstab[caty + catx * ncat_y];
                            split_info.right_branch[caty] += buffer_crosstab[caty + catx * ncat_y];
                        }
                        split_info.size_left  -= buffer_cat_cnt[catx];
                        split_info.size_right += buffer_cat_cnt[catx];
                    } else {
                        for (size_t caty = 0; caty < ncat_y; caty++) {
                            split_info.left_branch [caty] += buffer_crosstab[caty + catx * ncat_y];
                            split_info.right_branch[caty] -= buffer_crosstab[caty + catx * ncat_y];
                        }
                        split_info.size_left  += buffer_cat_cnt[catx];
                        split_info.size_right -= buffer_cat_cnt[catx];
                    }

                }
            }

        }

        /* check that split meets minimum criteria */
        if (split_info.size_left < min_size || split_info.size_right < min_size) continue;

        /* now evaluate the subset */
        this_gain = categ_gain(split_info, base_info);
        if (this_gain > *gain) {
            *gain = this_gain;
            best_subset = combin;
        }
        
    }

    /* now convert the best subset into a proper array */
    if (*gain > -HUGE_VAL) {
        for (size_t catx = 0; catx < ncat_x; catx++) {
            split_subset[catx] = extract_bit(best_subset, catx);
        }
        flag_zero_counts(split_subset, buffer_cat_cnt, ncat_x);
    }

}
