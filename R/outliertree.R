#' @title Outlier Tree
#' @description Fit Outlier Tree model to normal data with perhaps some outliers.
#' @param df  Data Frame with normal data that might contain some outliers. See details for allowed column types.
#' @param max_depth Maximum depth of the trees to grow. Can also pass zero, in which case it will only look
#' for outliers with no conditions (i.e. takes each column as a 1-d distribution and looks for outliers in
#' there independently of the values in other columns).
#' @param min_gain Minimum gain that a split has to produce in order to consider it (both in terms of looking
#' for outliers in each branch, and in considering whether to continue branching from them). Note that default
#' value for GritBot is 1e-6, with `gain_as_pct` = `FALSE`. Recommended to pass higher values (e.g. 1e-1) when using
#' `gain_as_pct` = `FALSE`.
#' @param z_norm Maximum Z-value (from standard normal distribution) that can be considered as a normal
#' observation. Note that simply having values above this will not automatically flag observations as outliers,
#' nor does it assume that columns follow normal distributions. Also used for categorical and ordinal columns
#' for building approximate confidence intervals of proportions.
#' @param z_outlier Minimum Z-value that can be considered as an outlier. There must be a large gap in the
#' Z-value of the next observation in sorted order to consider it as outlier, given by (z_outlier - z_norm).
#' Decreasing this parameter is likely to result in more observations being flagged as outliers.
#' Ignored for categorical and ordinal columns.
#' @param pct_outliers Approximate max percentage of outliers to expect in a given branch.
#' @param min_size_numeric Minimum size that branches need to have when splitting a numeric column. In order to look for
#' outliers in a given branch for a numeric column, it must have a minimum of twice this number
#' of observations.
#' @param min_size_categ Minimum size that branches need to have when splitting a categorical or ordinal column. In order to
#' look for outliers in a given branch for a categorical, ordinal, or boolean column, it must have a minimum of twice
#' this number of observations.
#' @param categ_split How to produce categorical-by-categorical splits. Options are:
#' \itemize{
#'   \item `"binarize"` : Will binarize the target variable according to whether it's equal to each present category
#'   within it (greater/less for ordinal), and split each binarized variable separately.
#'   \item `"bruteforce"` : Will evaluate each possible binary split of the categories (that is, it evaluates 2^n potential
#'   splits every time). Note that trying this when there are many categories in a column will result
#'   in exponential computation time that might never finish.
#'   \item `"separate"` : Will create one branch per category of the splitting variable (this is how GritBot handles them).
#' }
#' @param categ_outliers How to look for outliers in categorical variables. Options are:
#' \itemize{
#'   \item `"tail"` : Will try to flag outliers if there is a large gap between proportions in sorted order, and this
#'   gap is unexpected given the prior probabilities. Such criteria tends to sometimes flag too many
#'   uninteresting outliers, but is able to detect more cases and recognize outliers when there is no
#'   single dominant category.
#'   \item `"majority"` : Will calculate an equivalent to z-value according to the number of observations that do not
#'   belong to the non-majority class, according to formula '(n-n_maj)/(n * p_prior) < 1/z_outlier^2'.
#'   Such criteria  tends to miss many interesting outliers and will only be able to flag outliers in
#'   large sample sizes. This is the approach used by GritBot.
#' }
#' @param cols_ignore Vector containing columns which will not be split, but will be evaluated for usage
#' in splitting other columns. Can pass either a logical (boolean) vector with the same number of columns
#' as `df`, or a character vector of column names (must match with those of `df`).
#' Pass `NULL` to use all columns.
#' @param follow_all Whether to continue branching from each split that meets the size and gain criteria.
#' This will produce exponentially many more branches, and if depth is large, might take forever to finish.
#' Will also produce a lot more spurious outiers. Not recommended.
#' @param gain_as_pct Whether the minimum gain above should be taken in absolute terms, or as a percentage of
#' the standard deviation (for numerical columns) or shannon entropy (for categorical columns). Taking it in
#' absolute terms will prefer making more splits on columns that have a large variance, while taking it as a
#' percentage might be more restrictive on them and might create deeper trees in some columns. For GritBot
#' this parameter would always be `FALSE`. Recommended to pass higher values for `min_gain` when passing `FALSE`
#' here. Not that when `gain_as_pct` = `FALSE`, the results will be sensitive to the scales of variables.
#' @param save_outliers Whether to store outliers detected in `df` in the object that is returned.
#' These outliers can then be extracted from the returned object through function
#' `extract.training.outliers`.
#' @param outliers_print Maximum number of flagged outliers in the training data to print after fitting
#' the model. Pass zero or `NULL` to avoid printing any. Outliers can be printed from resulting data frame
#' afterwards through the `predict` method, or through the `print` method (on the extracted outliers, not on
#' the model object) if passing `save_outliers` = `TRUE`.
#' @param nthreads Number of parallel threads to use. When fitting the model, it will only use up to one
#' thread per column, while for prediction it will use up to one thread per row. The more threads that are
#' used, the more memory will be required and allocated, so using more threads will not always lead to better
#' speed. Can be changed after the object is already initialized.
#' @return An object with the fitted model that can be used to detect more outliers in new data, and from
#' which outliers in the training data can be extracted (when passing `save_outliers` = `TRUE`).
#' @details Explainable outlier detection through decision-tree grouping. Tries to detect outliers by
#' generating decision trees that attempt to "predict" the values of each column based on each other column,
#' testing in each branch of every tried split (if it meets some minimum criteria) whether there are
#' observations that seem too distant from the others in a 1-D distribution for the column that the split
#' tries to "predict" (unlike other methods, this will not generate a score for each observation).
#' 
#' Splits are based on gain, while outlierness is based on confidence intervals. Similar in spirit to the GritBot
#' software developed by RuleQuest research.
#' 
#' Supports columns of types numeric (either as type `numeric` or `integer`), categorical (either as type `character` or
#' `factor` with unordered levels), boolean (as type `logical`), and ordinal (as type `factor` with ordered levels as checked by
#' `is.ordered`). Can handle missing values in any of them. Can also pass dates/timestamps that will get converted to
#' numeric but shown as dates/timestamps in the output. Offers option to set columns to be used only for generating
#' conditions without looking at outliers in them.
#' 
#' Infinite values will be taken into consideration when the column is used to split another column
#' (that is, +inf will go into the branch that is greater than something, -inf into the other branch),
#' but when a column is the target of the split, they will be taken as missing - that is, it will not report
#' infinite values as outliers. 
#' @references \itemize{
#'   \item GritBot software: \url{https://www.rulequest.com/gritbot-info.html}
#'   \item Cortes, David. "Explainable outlier detection through decision tree conditioning." arXiv preprint arXiv:2001.00636 (2020).
#' }
#' @seealso \link{predict.outliertree} \link{extract.training.outliers} \link{hypothyroid} \link{unpack.outlier.tree}
#' @examples 
#' library(outliertree)
#' 
#' ### example dataset with interesting outliers
#' data(hypothyroid)
#' 
#' ### fit the model and get a print of outliers
#' model <- outlier.tree(hypothyroid,
#'   outliers_print=10,
#'   save_outliers=TRUE,
#'   nthreads=1)
#' 
#' ### extract outlier info as R list
#' outliers <- extract.training.outliers(model)
#' summary(outliers)
#' 
#' ### information for row 745
#' outliers[[745]]
#' 
#' ### use custom row names
#' df.w.names <- hypothyroid
#' row.names(df.w.names) <- paste0("rownum", 1:nrow(hypothyroid))
#' outliers.w.names <- predict(model, df.w.names, return_outliers=TRUE)
#' outliers.w.names[["rownum745"]]
#' @export
outlier.tree <- function(df, max_depth = 4, min_gain = 1e-2, z_norm = 2.67, z_outlier = 8.0,
                         pct_outliers = 0.01, min_size_numeric = 25, min_size_categ = 50,
                         categ_split = "binarize", categ_outliers = "tail", cols_ignore = NULL,
                         follow_all = FALSE, gain_as_pct = TRUE,
                         save_outliers = FALSE, outliers_print = 10,
                         nthreads = parallel::detectCores())
{
    ### validate inputs
    allowed_cs <- c("binarize", "bruteforce", "separate")
    if (NROW(categ_split) != 1 || !(categ_split %in% allowed_cs)) {
        stop(paste0("'categ_split' must be one of ", paste(allowed_cs, collapse = ", ")))
    }
    allowed_co <- c("tail", "majority")
    if (NROW(categ_outliers) != 1 || !(categ_outliers %in% allowed_co)) {
        stop(paste0("'categ_outliers' must be one of ", paste(allowed_co, collapse = ", ")))
    }
    if (max_depth < 0) { stop("'max_depth' must be >= 0.") }
    if (!("numeric" %in% class(min_gain)))     { stop("'min_gain' must be a decimal number.")     }
    if (!("numeric" %in% class(z_norm)))       { stop("'z_norm' must be a decimal number.")       }
    if (!("numeric" %in% class(z_outlier)))    { stop("'z_outlier' must be a decimal number.")    }
    if (!("numeric" %in% class(pct_outliers))) { stop("'pct_outliers' must be a decimal number.") }
    if (z_norm <= 0)          { stop("'z_norm' must be > 0.")                     }
    if (z_outlier < z_norm)   { stop("'z_outlier' must be >= 'z_norm'.")          }
    if (pct_outliers < 0)     { stop("'pct_outliers' must be greater than zero.") }
    if (pct_outliers > 0.1)   { stop("'pct_outliers' passed is too large.")       }
    if (min_size_numeric < 5) { stop("'min_size_numeric' is too small.")          }
    if (min_size_categ < 5)   { stop("'min_size_categ' is too small.")            }
    nthreads <- check.nthreads(nthreads)
    
    ### cast inputs
    if (is.null(outliers_print)) { outliers_print <- FALSE }
    if (is.na(outliers_print))   { outliers_print <- FALSE }
    if (outliers_print <= 0)     { outliers_print <- FALSE }
    if (outliers_print)          { outliers_print <- as.integer(outliers_print) }
    max_depth <- as.integer(max_depth)
    min_gain  <- as.numeric(min_gain)
    z_norm    <- as.numeric(z_norm)
    z_outlier <- as.numeric(z_outlier)
    pct_outliers     <- as.numeric(pct_outliers)
    min_size_numeric <- as.integer(min_size_numeric)
    min_size_categ   <- as.integer(min_size_categ)
    categ_split      <- categ_split
    categ_outliers   <- categ_outliers
    follow_all  <- as.logical(follow_all)
    gain_as_pct <- as.logical(gain_as_pct)

    ### decompose data into C arrays and names, then pass to C++
    model_data <- split.types(df, cols_ignore, nthreads)
    model_data$obj_from_cpp <- fit_OutlierTree(model_data$arr_num, model_data$ncol_num,
                                               model_data$arr_cat, model_data$ncol_cat, model_data$ncat,
                                               model_data$arr_ord, model_data$ncol_ord, model_data$ncat_ord,
                                               model_data$nrow, model_data$cols_ign, nthreads,
                                               categ_split == "binarize", categ_split == "binarize",
                                               categ_split == "bruteforce", categ_outliers == "majority",
                                               max_depth, pct_outliers, min_size_numeric, min_size_categ,
                                               min_gain, follow_all, gain_as_pct, z_norm, z_outlier,
                                               as.logical(save_outliers | outliers_print),
                                               lapply(model_data$cat_levels, as.character),
                                               lapply(model_data$ord_levels, as.character),
                                               as.character(c(model_data$cols_num, model_data$cols_date, model_data$cols_ts)),
                                               as.character(c(model_data$cols_cat, model_data$cols_bool)),
                                               as.character(model_data$cols_ord),
                                               model_data$date_min,
                                               model_data$ts_min)
    names(model_data$obj_from_cpp$bounds) <- get.cols.ordered(model_data)
    model_data$obj_from_cpp$bounds        <- model_data$obj_from_cpp$bounds[names(df)]
    
    ### print or store the outliers if requested
    if (outliers_print > 0) {
        if (model_data$obj_from_cpp$found_outliers) {
                report.outliers(model_data$obj_from_cpp$outliers_info, row.names(df), outliers_print)
            } else {
                report.no.outliers()
            }
    }
    if (save_outliers) {
        if (model_data$obj_from_cpp$found_outliers) {
                model_data$outliers_data <- outliers.to.list(df, model_data$obj_from_cpp$outliers_info)
            } else {
                model_data$outliers_data <- produce.empty.outliers(row.names(df))
            }
    } else {
        model_data$outliers_data <- NULL
    }
    model_data$obj_from_cpp$outliers_info <-NULL
    
    ### return object with corresponding class
    model_data$nthreads <- nthreads
    class(model_data)   <- "outliertree"
    return(model_data)
}

#' @title Predict method for Outlier Tree
#' @param object An Outlier Tree object as returned by `outlier.tree`.
#' @param newdata A Data Frame in which to look for outliers according to the fitted model.
#' @param outliers_print How many outliers to print. Pass zero or `NULL` to avoid printing them. Must pass
#' at least one of `outliers_print` and `return_outliers`.
#' @param return_outliers Whether to return the outliers in an R object (otherwise will just print them).
#' @param ... Not used.
#' @return If passing `return_outliers` = `TRUE`, will return a list of lists with the outliers and their
#' information (each row is an entry in the first list, with the same names as the rows in the input data
#' frame), which can be printed into a human-readable format after-the-fact through functions
#' `print` and `summary` (they do the same thing).
#' Otherwise, will not return anything, but will print the outliers if any are detected.
#' Note that, while the object that is returned will display a short summary of only some observations
#' when printing it in the console, it actually contains information for all rows, and can be subsetted
#' to obtain information specific to one row.
#' @details Note that after loading a serialized object from `outlier.tree` through `readRDS` or `load`,
#' it will only de-serialize the underlying C++ object upon running `predict` or `print`, so the first run will
#' be slower, while subsequent runs will be faster as the C++ object will already be in-memory.
#' @seealso \link{outlier.tree} \link{print.outlieroutputs} \link{unpack.outlier.tree}
#' @examples 
#' library(outliertree)
#' ### random data frame with an obvious outlier
#' nrows = 100
#' set.seed(1)
#' df = data.frame(
#'     numeric_col1 = c(rnorm(nrows - 1), 1e6),
#'     numeric_col2 = rgamma(nrows, 1),
#'     categ_col    = sample(c('categA', 'categB', 'categC'),
#'         size = nrows, replace = TRUE)
#'     )
#'     
#' ### test data frame with another obvious outlier
#' nrows_test = 50
#' df_test = data.frame(
#'     numeric_col1 = rnorm(nrows_test),
#'     numeric_col2 = c(-1e6, rgamma(nrows_test - 1, 1)),
#'     categ_col    = sample(c('categA', 'categB', 'categC'),
#'         size = nrows_test, replace = TRUE)
#' )
#'     
#' ### fit model on training data
#' outliers_model = outlier.tree(df, outliers_print=FALSE, nthreads=1)
#' 
#' ### find the test outlier
#' test_outliers = predict(outliers_model, df_test,
#'     outliers_print = 1, return_outliers = TRUE)
#' 
#' ### retrieve the outlier info (for row 1) as an R list
#' test_outliers[[1]]
#' @export 
predict.outliertree <- function(object, newdata, outliers_print = 15, return_outliers = FALSE, ...) {
    check.is.model.obj(object)
    if (check_null_ptr_model(object$obj_from_cpp$ptr_model)) {
        ptr_new <- deserialize_OutlierTree(object$obj_from_cpp$serialized_obj)
        eval.parent(substitute(object$obj_from_cpp$ptr_model <- ptr_new))
        object$obj_from_cpp$ptr_model <- ptr_new
    }
    outliers_print  <- check.outliers.print(outliers_print)
    return_outliers <- as.logical(return_outliers)
    if (NROW(newdata) == 0) {
        if (outliers_print) {
            report.no.outliers()
        }
        if (return_outliers) { return(produce.empty.outliers(row.names(newdata))) } else { return(invisible(NULL)) }
    }
    
    c_arr_data    <- split.types.new(newdata, object)
    outliers_info <- predict_OutlierTree(object$obj_from_cpp$ptr_model, NROW(newdata), object$nthreads,
                                         c_arr_data$arr_num, c_arr_data$arr_cat, c_arr_data$arr_ord,
                                         object$cat_levels,
                                         object$ord_levels,
                                         as.character(c(object$cols_num, object$cols_date, object$cols_ts)),
                                         as.character(c(object$cols_cat, object$cols_bool)),
                                         as.character(object$cols_ord),
                                         object$date_min,
                                         object$ts_min)
    if (outliers_print > 0) {
        if (outliers_info$found_outliers) {
                report.outliers(outliers_info, row.names(newdata), outliers_print)
            } else {
                report.no.outliers()
            }
    }
    if (return_outliers) {
        outliers_info$found_outliers <- NULL
        return(outliers.to.list(newdata, outliers_info))
    }
}

#' @title Print outliers in human-readable format
#' @description Pretty-prints outliers as output by the `predict` function from an Outlier Tree
#' model (as generated by function `outlier.tree`), or by `extract.training.outliers`.
#' Same as function `summary`.
#' @param x Outliers as returned by predict method on an object from `outlier.tree`.
#' @param outliers_print Maximum number of outliers to print.
#' @param only_these_rows Specific rows to print (either numbers if the row names in the original
#' data frame were null, or the row names they had if non-null). Pass `NULL` to print information
#' about potentially all rows
#' @param ... Not used.
#' @return No return value.
#' @seealso \link{outlier.tree} \link{predict.outliertree}
#' @examples 
#' ### Example re-printing results for selected rows
#' library(outliertree)
#' data("hypothyroid")
#' 
#' ### Fit model
#' otree <- outlier.tree(hypothyroid,
#'   nthreads=1,
#'   outliers_print=0)
#'   
#' ### Store predictions
#' pred <- predict(otree,
#'   hypothyroid,
#'   outliers_print=0,
#'   return_outliers=TRUE)
#'   
#' ### Print stored predictions
#' ### Row 531 is an outlier, but 532 is not
#' print(pred, only_these_rows = c(531, 532))
#' @export 
print.outlieroutputs <- function(x, outliers_print = 15, only_these_rows = NULL, ...) {
    if (NROW(x) == 0) { report.no.outliers(); return(invisible(NULL)); }
    outliers_print <- check.outliers.print(outliers_print)
    if (!outliers_print) { stop("Must pass a positive integer for 'outliers_print'.") }
    if (is.null(only_these_rows)) {
        outliers_info <- list.to.outliers(x)
        report.outliers(outliers_info, names(x), outliers_print)
    } else {
        outliers_info <- list.to.outliers(x[only_these_rows])
        report.outliers(outliers_info, names(x[only_these_rows]), outliers_print)
    }
}

#' @title Extract outliers found in training data
#' @description Extracts outliers from a model generated by `outlier.tree` if
#' it was passed parameter `save_outliers` = `TRUE`.
#' @param outlier_tree_model An Outlier Tree object as returned by `outlier.tree`.
#' @return A data frame with the outliers, which can be pretty-printed by function
#' `print` from this same package.
#' @export 
extract.training.outliers <- function(outlier_tree_model) {
    check.is.model.obj(outlier_tree_model)
    if (is.null(outlier_tree_model$outliers_data)) {
        stop("Outlier Tree model object has no recorded outliers. Pass 'save_outliers = TRUE' to keep them.")
    }
    return(outlier_tree_model$outliers_data)
}

#' @title Check values that could potentially flag an observation as outlier
#' @description Returns, for each numeric/date/timestamp column, a range of values *outside* of which
#' observations could potentially be flagged as being an outlier in some cluster, and
#' for categorical/ordinal/boolean columns, the factor levels that can be flagged as
#' being an outlier in some cluster. If the lower bound is higher than the upper bound, it means
#' any value can potentially be flagged as outlier.
#' @param outlier_tree_model An Outlier Tree model object as generated by `outlier.tree`.
#' @return A list with column as the names and the bounds or categories as values.
#' @export
check.outlierness.bounds <- function(outlier_tree_model) {
    check.is.model.obj(outlier_tree_model)
    return(outlier_tree_model$obj_from_cpp$bounds)
}

#' @title Unpack Outlier Tree model after de-serializing
#' @description  After persisting an outlier tree model object through `saveRDS`, `save`, or restarting a session, the
#' underlying C++ objects that constitute the outlier tree model and which live only on the C++ heap memory are not saved along,
#' thus not restored after loading a saved model through `readRDS` or `load`.
#' 
#' The model object however keeps serialized versions of the C++ objects as raw bytes, from which the C++ objects can be
#' reconstructed, and are done so automatically after calling `predict`, `print`, or `summary` on the
#' freshly-loaded object from `readRDS` or `load`.
#' 
#' But due to R's environments system (as opposed to other systems such as Python which can use pass-by-reference), they will
#' only be re-constructed in the environment that is calling `predict`, `print`, etc. and not in higher-up environments
#' (i.e. if you call `predict` on the object from inside different functions, each function will have to reconstruct the
#' C++ objects independently and they will only live within the function that called `predict`).
#' 
#' This function serves as an environment-level unpacker that will reconstruct the C++ object in the environment in which
#' it is called (i.e. if you need to call `predict` from inside multiple functions, use this function before passing the
#' freshly-loaded model object to those other functions, and then they will not need to reconstruct the C++ objects anymore),
#' in the same way as `predict` or `print`, but without producing any outputs or messages.
#' @param model An Outlier Tree object as returned by `outlier.tree`, which has been just loaded from a disk
#' file through `readRDS`, `load`, or a session restart.
#' @return No return value. Object is modified in-place.
#' @examples 
#' ### Warning: this example will generate a temporary .Rds
#' ### file in your temp folder, and will then delete it
#' library(outliertree)
#' set.seed(1)
#' df <- as.data.frame(matrix(rnorm(1000), nrow = 250))
#' otree <- outlier.tree(df, outliers_print=0, nthreads=1)
#' temp_file <- file.path(tempdir(), "otree.Rds")
#' saveRDS(otree, temp_file)
#' otree2 <- readRDS(temp_file)
#' file.remove(temp_file)
#' 
#' ### will de-serialize inside, but object is short-lived
#' wrap_predict <- function(model, data) {
#'     pred <- predict(model, data, outliers_print = 0)
#'     cat("pointer inside function is this: ")
#'     print(model$obj_from_cpp$ptr_model)
#'     return(pred)
#' }
#' temp <- wrap_predict(otree2, df)
#' cat("pointer outside function is this: \n")
#' print(otree2$obj_from_cpp$ptr_model) ### pointer to the C++ object
#' 
#' ### now unpack the C++ object beforehand
#' unpack.outlier.tree(otree2)
#' print("after unpacking beforehand")
#' temp <- wrap_predict(otree2, df)
#' cat("pointer outside function is this: \n")
#' print(otree2$obj_from_cpp$ptr_model)
#' @export
unpack.outlier.tree <- function(model)  {
    check.is.model.obj(model)
    if (check_null_ptr_model(model$obj_from_cpp$ptr_model)) {
        ptr_new <- deserialize_OutlierTree(model$obj_from_cpp$serialized_obj)
        eval.parent(substitute(model$obj_from_cpp$ptr_model <- ptr_new))
        model$obj_from_cpp$ptr_model <- ptr_new
    }
    return(invisible(NULL))
}
