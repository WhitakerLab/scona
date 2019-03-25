#!/usr/bin/env python
"""
Given a set of participants split into groups group_1, group_2, ... of sizes
x_1, x_2,... randomly redivide the set in to sizes of x_1, x_2, ... to build
a null distribution for the network structure.
"""
# Essential package imports
import os
import numpy as np
from scona.make_corr_matrices import corrmat_from_regionalmeasures

def corr_matrices_from_divided_data(
        regional_measures,
        names,
        "group"_index,
        covars=None,
        method='pearson'):
    '''
    Calculate the correlation of `names` columns over the rows of
    `regional_measures` after correcting for covariance with the columns in
    `covars`

    Parameters
    ----------
    regional_measures : :class:`pandas.DataFrame`
        a pandas data frame with subjects as rows, and columns including
        brain regions, covariates and a column describing the "group". Should be numeric for the columns in
        names and covars_list
    names : list
        a list of the brain regions you wish to correlate
    "group"_index : str
        the index of the column describing "group" participation
    covars: list
        covars is a list of covariates (as df column headings)
        to correct for before correlating the regions.
    methods : string
        the method of correlation passed to :func:`pandas.DataFrame.corr`

    Returns
    -------
    :class:`pandas.DataFrame`
        A correlation matrix with rows and columns keyed by `names`
    '''
    corr_dict = {}
    for key, group in regional_measures.groupby("group"_index):
        corr_dict[key] = corrmat_from_regionalmeasures(
            group, names, covars=covars, method=method)
    return corr_dict

    def shuffle_"group"():
