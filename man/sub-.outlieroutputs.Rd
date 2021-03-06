% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/outliertree.R
\name{[.outlieroutputs}
\alias{[.outlieroutputs}
\title{Slice or sub-set outliers}
\usage{
\method{[}{outlieroutputs}(x, i)
}
\arguments{
\item{x}{An object of class `outlieroutputs`, as returned by e.g. \link{predict.outliertree}.}

\item{i}{Rows to select. Can pass numbers or strings. Works the same as when selecting
elements from a list.}
}
\value{
An object of class `outlieroutputs` containing only the selected rows.
}
\description{
Select given rows from outlier results.
}
