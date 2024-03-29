% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/obj_methods.R
\name{summary.outlieroutputs}
\alias{summary.outlieroutputs}
\title{Print outliers in human-readable format}
\usage{
\method{summary}{outlieroutputs}(object, outliers_print = 15, ...)
}
\arguments{
\item{object}{Outliers as returned by predict method on an object from `outlier.tree`.}

\item{outliers_print}{Maximum number of outliers to print.}

\item{...}{Not used.}
}
\value{
The same `object` input, returned invisibly.
}
\description{
Pretty-prints outliers as output by the `predict` function from an Outlier Tree
model (as generated by function `outlier.tree`), or by `extract.training.outliers`.
Same as function `print` (see the documentation of `print` for more information about
the parameters).
}
\seealso{
\link{print.outlieroutputs}
}
