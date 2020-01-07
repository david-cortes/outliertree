check.is.df <- function(df) {
    if ("tibble" %in% names(df))        { df <- as.data.frame(df) }
    if (!("data.frame" %in% class(df))) { stop("Input data must be a data.frame.") }
    if (length(class(df)) > 1)          { df <- as.data.frame(df) }
    if (!NCOL(df)) { stop("Input data frame has no columns.")}
    if (!NROW(df)) { stop("Input data frame has no rows.")   }
    return(df)
}

split.types <- function(df, cols_ignore = NULL, nthreads = 1) {
    
    ### validate input data
    df <- check.is.df(df)
    if (NROW(df) < 25)  { stop("Input data has too few row.") }
    supported_types <- c("numeric", "integer", "character", "factor", "ordered", "logical", "Date", "POSIXct")
    coltypes_df     <- Reduce(c, lapply(df, class))
    if (length(setdiff(unique(coltypes_df), c(supported_types, "POSIXt"))) > 0) {
        stop(paste0("Input data can only have the collowing column types: ",
                    paste(supported_types, collapse = ", "),
                    " - got passed the following: ",
                    paste(setdiff(unique(coltypes_df), supported_types), collapse = ", ")))
    }
    if (!is.null(cols_ignore)) {
        if ("character" %in% class(cols_ignore)) {
            if (length(setdiff(cols_ignore, names(df)))) {
                stop(paste0("'cols_ignore' contains names not present in 'df' - head: ",
                            paste(head(setdiff(cols_ignore, names(df)), 3), collapse = ", ")))
            }
        } else if ("logical" %in% class(cols_ignore)) {
            if (length(cols_ignore) != NCOL(df)) {
                stop("'cols_ignore' must have one entry per column of the input data frame.")
            }
        } else {
            stop(paste0("'cols_ignore' must be either a vector with column names to ignore, ",
                        "or a logical (boolean) vector indicating which columns to ignore"))
        }
    }
    
    ### initialize output object
    outp <- list()
    
    ### split by column type
    all_cols <- names(df)
    cols_numeric <- all_cols[sapply(df, function(x) "numeric"   %in% class(x))]
    cols_integer <- all_cols[sapply(df, function(x) "integer"   %in% class(x))]
    cols_boolean <- all_cols[sapply(df, function(x) "logical"   %in% class(x))]
    cols_factor  <- all_cols[sapply(df, function(x) "factor"    %in% class(x))]
    cols_ord     <- all_cols[sapply(df, function(x) "ordered"   %in% class(x))]
    cols_txt     <- all_cols[sapply(df, function(x) "character" %in% class(x))]
    cols_date    <- all_cols[sapply(df, function(x) "Date"      %in% class(x))]
    cols_ts      <- all_cols[sapply(df, function(x) "POSIXct"   %in% class(x))]
    
    outp$cols_num   <- c(cols_numeric, cols_integer)
    outp$cols_cat   <- c(cols_factor,  cols_txt)
    outp$cols_ord   <- cols_ord
    outp$cols_bool  <- cols_boolean
    outp$cols_date  <- cols_date
    outp$cols_ts    <- cols_ts
    outp$date_min   <- as.numeric()
    outp$ts_min     <- as.numeric()
    outp$cat_levels <- list()
    outp$ord_levels <- list()
    
    if (is.null(outp$cols_num))  { outp$cols_num  <- as.character() }
    if (is.null(outp$cols_cat))  { outp$cols_cat  <- as.character() }
    if (is.null(outp$cols_ord))  { outp$cols_ord  <- as.character() }
    if (is.null(outp$cols_bool)) { outp$cols_bool <- as.character() }
    if (is.null(outp$cols_date)) { outp$cols_date <- as.character() }
    if (is.null(outp$cols_ts))   { outp$cols_ts   <- as.character() }
    
    if (NROW(outp$cols_cat) && NROW(outp$cols_ord)) {
        outp$cols_cat <- setdiff(outp$cols_cat, outp$cols_ord)
    }
    
    if (NROW(cols_date)) {
        df[, cols_date] <- as.data.frame(lapply(df[, cols_date, drop = FALSE], as.numeric))
        outp$date_min   <- sapply(df[, cols_date, drop = FALSE], min, na.rm = TRUE)
        if (NROW(cols_date) != NROW(df)) {
            df[, cols_date] <- df[, cols_date, drop = FALSE] - outp$date_min + 1
        } else {
            ### if the number of columns is the same as the number of rows, R will
            ### always make subtraction by-row instead of by-col, regardless of whether
            ### it's passed as a matrix with a certain shape like in NumPy
            df[, cols_date] <- mapply(function(a, b) a - b + 1, df[, cols_date, drop = FALSE], outp$date_min)
        }
        ## the extra 1 is for the way in which package applies log transforms
    }
    
    if (NROW(cols_ts)) {
        df[, cols_ts] <- as.data.frame(lapply(df[, cols_ts, drop = FALSE], as.numeric))
        outp$ts_min   <- sapply(df[, cols_ts, drop = FALSE], min, na.rm = TRUE)
        if (NROW(cols_ts) != NROW(df)) {
            df[, cols_ts] <- df[, cols_ts, drop = FALSE] - outp$ts_min + 1
        } else {
            df[, cols_ts] <- mapply(function(a, b) a - b + 1, df[, cols_ts, drop = FALSE], outp$ts_min)
        }
    }
    
    if (NROW(outp$cols_cat)) {
        df[, outp$cols_cat]  <- as.data.frame(lapply(df[, outp$cols_cat, drop = FALSE], factor))
        outp$cat_levels      <- lapply(df[, outp$cols_cat, drop = FALSE], levels)
        df[, outp$cols_cat]  <- as.data.frame(lapply(df[, outp$cols_cat, drop = FALSE],
                                                     function(x) ifelse(is.na(x), -1, as.integer(x) - 1))
                                              )
    }
    
    if (NROW(outp$cols_ord)) {
        outp$ord_levels     <- lapply(df[, outp$cols_ord, drop = FALSE], levels)
        df[, outp$cols_ord] <- as.data.frame(lapply(df[, outp$cols_ord, drop = FALSE],
                                                    function(x) ifelse(is.na(x), -1, as.integer(x) - 1))
                                             )
        ### check that they have at least 3 levels
        min_ord_levs = min(sapply(outp$ord_levels, length))
        if (min_ord_levs < 3) { stop("Ordinal columns must have at least 3 levels.") }
    }
    
    if (NROW(outp$cols_bool)) {
        df[, outp$cols_bool] <- as.data.frame(lapply(df[, outp$cols_bool, drop = FALSE],
                                                     function(x) ifelse(is.na(x), -1, as.integer(x)))
                                              )
    }
    
    outp$arr_num  <- as.numeric()
    outp$arr_cat  <- as.integer()
    outp$arr_ord  <- as.integer()
    outp$ncol_num <- 0
    outp$ncol_cat <- 0
    outp$ncol_ord <- 0
    outp$nrow     <- NROW(df)
    outp$ncat     <- as.integer()
    outp$ncat_ord <- as.integer()
    outp$cols_ign <- as.logical()
    
    if (NROW(outp$cols_num) || NROW(outp$cols_date) || NROW(outp$cols_ts)) {
        outp$arr_num  <- as.numeric(as.matrix(df[, c(outp$cols_num, outp$cols_date, outp$cols_ts), drop = FALSE]))
        outp$ncol_num <- length(c(outp$cols_num, outp$cols_date, outp$cols_ts))
        
        ### check that they are not binary
        too_few_vals <- check_few_values(outp$arr_num, outp$nrow, outp$ncol_num, nthreads)
        if (any(too_few_vals)) {
            warning(paste0("Passed numeric columns with less than 3 different values - head: ",
                           paste(c(outp$cols_num, outp$cols_date, outp$cols_ts)[too_few_vals], collapse = ", ")))
        }

    }
    
    if (NROW(outp$cols_cat) || NROW(outp$cols_bool)) {
        outp$arr_cat  <- as.integer(as.matrix(df[, c(outp$cols_cat, outp$cols_bool), drop = FALSE]))
        outp$ncol_cat <- length(c(outp$cols_cat, outp$cols_bool))
        outp$ncat     <- sapply(df[, c(outp$cols_cat, outp$cols_bool), drop = FALSE], max) + 1
    }
    
    if (NROW(outp$cols_ord)) {
        outp$arr_ord  <- as.integer(as.matrix(df[, outp$cols_ord, drop = FALSE]))
        outp$ncol_ord <- length(outp$cols_ord)
        outp$ncat_ord <- sapply(df[, outp$cols_ord, drop = FALSE], max) + 1
    }
    
    if (!is.null(cols_ignore)) {
        cols_order <- get.cols.ordered(outp)
        if ("character" %in% class(cols_ignore)) {
            outp$cols_ign <- cols_order %in% cols_ignore
        } else if ("logical" %in% class(cols_ignore)) {
            outp$cols_ign <- as.logical(cols_ignore[match(names(df), cols_order)])
        }
    }
    
    return(outp)
}

get.cols.ordered <- function(model_data) {
    return(c(
                c(model_data$cols_num, model_data$cols_date, model_data$cols_ts),
                c(model_data$cols_cat, model_data$cols_bool),
                model_data$cols_ord
            ))
}

split.types.new <- function(df, model_data) {
    df <- check.is.df(df)
    if (length(setdiff(get.cols.ordered(model_data), names(df)))) {
        stop(paste0("Input data frame is missing some columns - head: ",
                    paste(head(setdiff(get.cols.ordered(model_data), names(df)), 3), collapse = ", ")))
    }
    throw_new_lev_warn <- FALSE
    outp = list(
        arr_num = as.numeric(),
        arr_cat = as.integer(),
        arr_ord = as.integer()
    )
    
    if (NROW(model_data$cols_num)) {
        df[, model_data$cols_num] <- as.data.frame(lapply(df[, model_data$cols_num, drop = FALSE], as.numeric))
    }
    if (NROW(model_data$cols_date)) {
        df[, model_data$cols_date] <- (as.data.frame(lapply(df[, model_data$cols_date, drop = FALSE], as.numeric))
                                       - model_data$date_min + 1)
    }
    if (NROW(model_data$cols_ts)) {
        df[, model_data$cols_ts] <- (as.data.frame(lapply(df[, model_data$cols_ts, drop = FALSE], as.numeric))
                                     - model_data$ts_min + 1)
    }
    if (NROW(model_data$cols_cat)) {
        for (cl in 1:NROW(model_data$cols_cat)) {
            new_levels <- !(df[[model_data$cols_cat[[cl]]]] %in% model_data$cat_levels[[cl]]) &
                          !is.na(df[[model_data$cols_cat[[cl]]]])
            df[[model_data$cols_cat[cl]]] <- factor(df[[model_data$cols_cat[[cl]]]],
                                                    unname(unlist(model_data$cat_levels[[cl]])))
            df[[model_data$cols_cat[cl]]] <- ifelse(is.na(df[[model_data$cols_cat[[cl]]]]),
                                                    -1, as.integer(df[[model_data$cols_cat[[cl]]]]) - 1)
            if (any(new_levels)) {
                df[[model_data$cols_cat[[cl]]]][new_levels] <- length(model_data$cat_levels[[cl]])
                throw_new_lev_warn <- TRUE
            }
        }
    }
    if (NROW(model_data$cols_bool)) {
        df[, model_data$cols_bool] <- as.data.frame(lapply(df[, model_data$cols_bool, drop = FALSE],
                                                           function(x) ifelse(is.na(x), -1, as.integer(as.logical(x)))))
    }
    if (NROW(model_data$cols_ord)) {
        for (cl in 1:NROW(model_data$cols_ord)) {
            new_levels <- !(df[[model_data$cols_ord[[cl]]]] %in% model_data$ord_levels[[cl]]) &
                          !is.na(df[[model_data$cols_ord[[cl]]]])
            df[[model_data$cols_ord[[cl]]]] <- factor(df[[model_data$cols_ord[cl]]], unname(unlist(model_data$ord_levels[[cl]])))
            df[[model_data$cols_ord[[cl]]]] <- ifelse(is.na(df[[model_data$cols_ord[[cl]]]]),
                                                      -1, as.integer(df[[model_data$cols_ord[[cl]]]]) - 1)
            if (any(new_levels)) {
                df[[model_data$cols_ord[[cl]]]][new_levels] <- length(model_data$ord_levels[[cl]])
                throw_new_lev_warn <- TRUE
            }
        }
    }
    
    if (NROW(model_data$cols_num) || NROW(model_data$cols_date) || NROW(model_data$cols_ts)) {
        outp$arr_num  <- as.numeric(as.matrix(df[, c(model_data$cols_num,
                                                     model_data$cols_date,
                                                     model_data$cols_ts),
                                                 drop = FALSE]))
    }
    
    if (NROW(model_data$cols_cat) || NROW(model_data$cols_bool)) {
        outp$arr_cat  <- as.integer(as.matrix(df[, c(model_data$cols_cat, model_data$cols_bool), drop = FALSE]))
    }
    
    if (NROW(model_data$cols_ord)) {
        outp$arr_ord  <- as.integer(as.matrix(df[, model_data$cols_ord, drop = FALSE]))
    }
    
    if (throw_new_lev_warn) {
        warning("Some column(s) contain new factor levels, these will be ignored.")
    }
    
    return(outp)
}

discard.input.data <- function(model_data) {
    model_data$arr_num  <- NULL
    model_data$arr_cat  <- NULL
    model_data$arr_ord  <- NULL
    model_data$ncat     <- NULL
    model_data$ncat_ord <- NULL
    model_data$nrow     <- NULL
    model_data$cols_ign <- NULL
    return(model_data)
}

check.nthreads <- function(nthreads) {
    if (is.null(nthreads)) {
        nthreads <- 1
    } else if (is.na(nthreads)) {
        nthreads <- 1
    } else if (nthreads == "auto") {
        nthreads <- parallel::detectCores()
    } else if (nthreads < 1) {
        nthreads <- parallel::detectCores()
    }
    return(as.integer(nthreads))
}

check.outliers.print <- function(outliers_print) {
    if (NROW(outliers_print) > 1) { stop("Must pass a scalar value for 'outliers_print'.") }
    if (is.null(outliers_print) || is.na(outliers_print) || outliers_print <= 0) {
        return(as.integer(0))
    }
    if ("numeric"   %in% class(outliers_print))  { outliers_print <- as.integer(outliers_print) }
    if (!("integer" %in% class(outliers_print))) { stop("'outliers_print' must be a positive integer.") }
    if (outliers_print <= 0) { stop("'outliers_print' must be a positive integer.") }
    return(outliers_print)
}

check.is.model.obj <- function(model_obj) {
    if (!("outliertree" %in% class(model_obj))) {
        stop("Must pass an Outlier Tree model object as generated by function 'outlier.tree'.")
    }
    if (is.null(model_obj$obj_from_cpp)) {
        stop("Outlier Tree model object has been corrupted.")
    }
}

report.no.outliers <- function() {
    cat("No outliers were found.\n")
}

report.outliers <- function(lst, rnames, outliers_print) {

    if (NROW(lst) == 0) { report.no.outliers(); return(invisible(NULL)); }
    
    suspicous_value  <- lst$suspicous_value
    group_statistics <- lst$group_statistics
    conditions       <- lst$conditions
    
    ### determine which ones to show
    df_outlierness <- data.frame(
        ix_num         = 1:NROW(lst$tree_depth),
        uses_NA_branch = lst$uses_NA_branch,
        tree_depth     = lst$tree_depth,
        outlier_score  = lst$outlier_score
    )
    ### https://stackoverflow.com/questions/1296646/how-to-sort-a-dataframe-by-multiple-columns
    df_outlierness <- df_outlierness[with(df_outlierness, order(uses_NA_branch, tree_depth, outlier_score)), ]
    
    ### if there are no outliers, stop at that
    if (is.na(df_outlierness[1, "tree_depth"]) || NROW(df_outlierness) == 0) {
        report.no.outliers()
        return(invisible(NULL));
    }
    
    ### otherwise, report only the most outlying ones
    df_outlierness <- df_outlierness[!is.na(df_outlierness$uses_NA_branch), ]
    cat(sprintf("Reporting top %d outliers [out of %d found]\n\n",
                min(outliers_print, NROW(df_outlierness)),
                NROW(df_outlierness)))
    df_outlierness <- df_outlierness[1:min(outliers_print, NROW(df_outlierness)), ]
    for (row in 1:NROW(df_outlierness)) {
        row_ix <- df_outlierness$ix_num[row]
        
        ### print suspicious value
        cat(sprintf("row [%s] - suspicious column: [%s] - ", rnames[row_ix], suspicous_value[[row_ix]]$column))
        if ("numeric" %in% class(suspicous_value[[row_ix]]$value)) {
            cat(sprintf("suspicious value: [%.3f]\n", suspicous_value[[row_ix]]$value))
        } else {
            cat(sprintf("suspicious value: [%s]\n", suspicous_value[[row_ix]]$value))
        }
        
        
        ### print distribution
        if ("mean" %in% names(group_statistics[[row_ix]])) {
            if ("upper_thr" %in% names(group_statistics[[row_ix]])) {
                if ("numeric" %in% class(group_statistics[[row_ix]]$upper_thr)) {
                    cat(sprintf("\tdistribution: %.3f%% <= %.3f - [mean: %.3f] - [sd: %.3f] - [norm. obs: %d]\n",
                                group_statistics[[row_ix]]$pct_below * 100,
                                group_statistics[[row_ix]]$upper_thr,
                                group_statistics[[row_ix]]$mean,
                                group_statistics[[row_ix]]$sd,
                                group_statistics[[row_ix]]$n_obs))
                } else {
                    cat(sprintf("\tdistribution: %.3f%% <= [%s] - [mean: %s] - [norm. obs: %d]\n",
                                group_statistics[[row_ix]]$pct_below * 100,
                                group_statistics[[row_ix]]$upper_thr,
                                group_statistics[[row_ix]]$mean,
                                group_statistics[[row_ix]]$n_obs))
                }
            } else {
                if ("numeric" %in% class(group_statistics[[row_ix]]$lower_thr)) {
                    cat(sprintf("\tdistribution: %.3f%% >= %.3f - [mean: %.3f] - [sd: %.3f] - [norm. obs: %d]\n",
                                group_statistics[[row_ix]]$pct_above * 100,
                                group_statistics[[row_ix]]$lower_thr,
                                group_statistics[[row_ix]]$mean,
                                group_statistics[[row_ix]]$sd,
                                group_statistics[[row_ix]]$n_obs))
                } else {
                    cat(sprintf("\tdistribution: %.3f%% >= [%s] - [mean: %s] - [norm. obs: %d]\n",
                                group_statistics[[row_ix]]$pct_above * 100,
                                group_statistics[[row_ix]]$lower_thr,
                                group_statistics[[row_ix]]$mean,
                                group_statistics[[row_ix]]$n_obs))
                }
            }
        } else if ("categs_common" %in% names(group_statistics[[row_ix]])) {
            if (NROW(group_statistics[[row_ix]]$categs_common) == 1) {
                cat(sprintf("\tdistribution: %.3f%% = [%s]\n",
                            group_statistics[[row_ix]]$pct_common * 100,
                            group_statistics[[row_ix]]$categs_common))
            } else {
                cat(sprintf("\tdistribution: %.3f%% in [%s]\n",
                            group_statistics[[row_ix]]$pct_common * 100,
                            paste(group_statistics[[row_ix]]$categs_common, collapse = ", ")))
            }
            if (NROW(conditions[[row_ix]])) {
                cat(sprintf("\t( [norm. obs: %d] - [prior_prob: %.3f%%] - [next smallest: %.3f%%] )\n",
                            group_statistics[[row_ix]]$n_obs,
                            group_statistics[[row_ix]]$prior_prob * 100,
                            group_statistics[[row_ix]]$pct_next_most_comm * 100))
            } else {
                cat(sprintf("\t( [norm. obs: %d] - [next smallest: %.3f%%] )\n",
                            group_statistics[[row_ix]]$n_obs,
                            group_statistics[[row_ix]]$pct_next_most_comm * 100))
            }
        }  else if ("categ_maj" %in% names(group_statistics[[row_ix]])) {
            cat(sprintf("\tdistribution: %.3f%% = [%s]\n",
                        group_statistics[[row_ix]]$pct_common * 100,
                        group_statistics[[row_ix]]$categ_maj))
            cat(sprintf("\t( [norm. obs: %d] - [prior_prob: %.3f%%] )\n",
                        group_statistics[[row_ix]]$n_obs,
                        group_statistics[[row_ix]]$prior_prob * 100))
        } else {
            cat(sprintf("\tdistribution: %.3f%% different [norm. obs: %d]",
                        group_statistics[[row_ix]]$pct_other * 100,
                        group_statistics[[row_ix]]$n_obs))
            if (NROW(conditions[[row_ix]])) {
                cat(sprintf(" - [prior_prob: %.3f%%]", group_statistics[[row_ix]]$prior_prob * 100))
            }
            cat("\n")
        }
        
        
        ### print conditions
        if (NROW(conditions[[row_ix]])) {
            cat("\tgiven:\n")
            conditions_this <- simplify.conditions(conditions[[row_ix]])
            for (cond in conditions_this) {
                switch(cond$comparison,
                       "is NA" = {
                               cat(sprintf("\t\t[%s] is NA\n", cond$column))
                       },
                       "<=" = {
                               if ("numeric" %in% class(cond$value_this)) {
                                   cat(sprintf("\t\t[%s] <= [%.3f] (value: %.3f)\n",
                                               cond$column, cond$value_comp, cond$value_this))
                               } else {
                                   cat(sprintf("\t\t[%s] <= [%s] (value: %s)\n",
                                               cond$column, cond$value_comp, cond$value_this))
                               }
                       },
                       ">" = {
                               if ("numeric" %in% class(cond$value_this)) {
                                   cat(sprintf("\t\t[%s] > [%.3f] (value: %.3f)\n",
                                               cond$column, cond$value_comp, cond$value_this))
                               } else {
                                   cat(sprintf("\t\t[%s] > [%s] (value: %s)\n",
                                               cond$column, cond$value_comp, cond$value_this))
                               }
                       },
                       "between" = {
                               if ("numeric" %in% class(cond$value_this)) {
                                   cat(sprintf("\t\t[%s] between (%.3f, %.3f] (value: %.3f)\n",
                                               cond$column, cond$value_comp[1], cond$value_comp[2], cond$value_this))
                               } else {
                                   cat(sprintf("\t\t[%s] between (%s, %s] (value: %s)\n",
                                               cond$column, cond$value_comp[1], cond$value_comp[2], cond$value_this))
                               }
                       },
                       "=" = {
                               cat(sprintf("\t\t[%s] = [%s]\n", cond$column, cond$value_comp))
                       },
                       "!=" = {
                               cat(sprintf("\t\t[%s] != [%s] (value: %s)\n",
                                           cond$column, cond$value_comp, cond$value_this))
                       },
                       "in" = {
                               cat(sprintf("\t\t[%s] in [%s] (value: %s)\n",
                                           cond$column, paste(cond$value_comp, collapse = ", "), cond$value_this))
                       })
            }
        }
        
        cat("\n\n")
    }
}

simplify.conditions <- function(conditions) {
    if (NROW(conditions) <= 1) { return(conditions) }
    cols_taken <- sapply(conditions, function(x) x$column)
    if (NROW(unique(cols_taken)) < NROW(cols_taken)) {
        repeated_cols <- table(cols_taken, useNA = "no")
        repeated_cols <- names(repeated_cols)[repeated_cols > 1]
        replacing_cond <- list()
        for (cl in repeated_cols) {
            
            n_le <- 0
            n_gt <- 0
            n_in <- 0
            n_eq <- 0
            n_neq       <- 0
            lowest_le   <-  Inf
            highest_gt  <- -Inf
            val_eq      <- NA
            val_neq     <- NA
            smallest_in <- NULL
            
            for (cn in 1:NROW(conditions)) {
                if (conditions[[cn]]$column == cl) {
                    val_this <- conditions[[cn]]$value_this
                    switch(conditions[[cn]]$comparison,
                           "<=" = {
                                   n_le <- n_le + 1
                                   if (conditions[[cn]]$value_comp < lowest_le) {
                                       lowest_le <- conditions[[cn]]$value_comp
                                   }
                           },
                           ">" = {
                                   n_gt <- n_gt + 1
                                   if (conditions[[cn]]$value_comp > highest_gt) {
                                       highest_gt <- conditions[[cn]]$value_comp
                                   }
                           },
                           "in" = {
                                   n_in <- n_in + 1
                                   if (is.null(smallest_in)) {
                                       smallest_in <- conditions[[cn]]$value_comp
                                   } else {
                                       smallest_in <- intersect(smallest_in, conditions[[cn]]$value_comp)
                                   }
                           },
                           "=" = {
                                   n_eq   <- n_eq + 1
                                   val_eq <- conditions[[cn]]$value_comp
                           },
                           "!=" = {
                                   n_neq   <- n_neq + 1
                                   val_neq <- conditions[[cn]]$value_comp
                           }
                    )
                }
            }
            
            if        (n_le  > 0 & n_gt == 0) {
                replacing_cond[[NROW(replacing_cond) + 1]] <- list(column = cl, value_this = val_this,
                                                                   comparison = "<=", value_comp = lowest_le)
            } else if (n_gt  > 0 & n_le == 0) {
                replacing_cond[[NROW(replacing_cond) + 1]] <- list(column = cl, value_this = val_this,
                                                                   comparison = ">", value_comp = highest_gt)
            } else if (n_le  > 0 & n_gt  > 0) {
                replacing_cond[[NROW(replacing_cond) + 1]] <- list(column = cl, value_this = val_this,
                                                                   comparison = "between", value_comp = c(highest_gt, lowest_le))
            } else if (n_in  > 0 & n_eq == 0 & n_neq == 0) {
                replacing_cond[[NROW(replacing_cond) + 1]] <- list(column = cl, value_this = val_this,
                                                                   comparison = "in", value_comp = smallest_in)
            } else if (n_in  > 0 & n_eq  > 0) {
                replacing_cond[[NROW(replacing_cond) + 1]] <- list(column = cl, value_this = val_this,
                                                                   comparison = "=", value_comp = val_eq)
            } else if (n_in  > 0 & n_neq > 0) {
                replacing_cond[[NROW(replacing_cond) + 1]] <- list(column = cl, value_this = val_this,
                                                                   comparison = "!=", value_comp = val_neq)
            }
            
        }
        conditions <- append(conditions[sapply(conditions, function(x) !(x$column %in% repeated_cols))],
                             replacing_cond)
    }
    
    conditions <- lapply(conditions, function(x)
                                        if ((x$comparison == "in") && (NROW(x$value_comp) == 1)) {
                                            x$comparison <- "="; return(x);
                                        } else { return(x) })
    return(conditions[rev(1:NROW(conditions))])
}


outliers.to.list <- function(df, outliers_info) {
    outliers_data <- lapply(1:NROW(df),
                            function(row, lst) list(
                                suspicous_value  = lst$suspicous_value[[row]],
                                group_statistics = lst$group_statistics[[row]],
                                conditions       = lst$conditions[[row]],
                                tree_depth       = lst$tree_depth[[row]],
                                uses_NA_branch   = lst$uses_NA_branch[[row]],
                                outlier_score    = lst$outlier_score[[row]]
                                ),
                            outliers_info)
    names(outliers_data) <- row.names(df)
    class(outliers_data) <- c("outlieroutputs", class(outliers_data))
    return(outliers_data)
}


list.to.outliers <- function(outliers_data) {
    return(list(
        suspicous_value  = sapply(outliers_data, function(x) x$suspicous_value, simplify = FALSE),
        group_statistics = sapply(outliers_data, function(x) x$group_statistics, simplify = FALSE),
        conditions       = sapply(outliers_data, function(x) x$conditions, simplify = FALSE),
        tree_depth       = sapply(outliers_data, function(x) x$tree_depth, simplify = TRUE),
        uses_NA_branch   = sapply(outliers_data, function(x) x$uses_NA_branch, simplify = TRUE),
        outlier_score    = sapply(outliers_data, function(x) x$outlier_score, simplify = TRUE)
    ))
}

produce.empty.outliers <- function(row_names) {
    empty_lst        <- rep_len(list(list(suspicous_value = list(), group_statistics= list(), conditions = list(),
                                          tree_depth = NA, uses_NA_branch = NA, outlier_score = NA)),
                                NROW(row_names))
    names(empty_lst) <- row_names
    class(empty_lst) <- c("outlieroutputs", class(empty_lst))
    return(empty_lst)
}
