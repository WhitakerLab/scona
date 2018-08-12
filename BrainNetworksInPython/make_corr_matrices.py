#!/usr/bin/env python
"""
Tools to create a correlation matrix from regional measures
"""
# Essential package imports
import os
import numpy as np
import pandas as pd
from BrainNetworksInPython.stats_functions import residuals


def get_non_numeric_cols(df):
    '''
    FILL
    '''
    numeric = np.fromiter((np.issubdtype(y, np.number) for y in df.dtypes),
                          bool)
    non_numeric_cols = np.array(df.columns)[~numeric]
    return non_numeric_cols


def create_residuals_df(df, names, covars_list):
    '''
    Calculate residuals of columns specified by names, correcting for the
    columns in covars_list.
    * df is a pandas data frame with subjects as rows and columns including
        brain regions and covariates
    * names is a list of the brain regions you wish to correlate.
    * covars_list is a list of covariates (as df column headings)
        to correct for before correlating the regions.

    df should be numeric for the columns in names and covars_list
    '''
    # Raise TypeError if any of the relevant columns are nonnumeric
    non_numeric_cols = get_non_numeric_cols(df[names+covars_list])
    if non_numeric_cols:
        raise TypeError('DataFrame columns {} are non numeric'
                        .format(', '.join(non_numeric_cols)))

    # Make a new data frame that will contain
    # the residuals for each column after correcting for
    # the covariates in covars
    df_res = df[names+covars_list].copy()

    # Create your covariates array
    if len(covars_list) > 1:
        x = np.vstack([df[covars_list]])
    elif len(covars_list) == 1:
        x = df[covars_list]
    else:
        x = np.ones_like(df.iloc[:, 0])

    # Calculate the residuals
    for name in names:
        df_res.loc[:, name] = residuals(x.T, df.loc[:, name])

    # Return the residuals data frame
    return df_res


def create_corrmat(df_residuals, names=None, method='pearson'):
    '''
    Returns a correlation matrix
    * df_res is a pandas data frame with participants as rows.
    * names is a list of the brain regions you wish to correlate.
    * method is the method of correlation passed to pandas.DataFram.corr
    '''
    if names is None:
        names = df_residuals.columns
    # Raise TypeError if any of the relevant columns are nonnumeric
    non_numeric_cols = get_non_numeric_cols(df_residuals)
    if non_numeric_cols:
        raise TypeError('DataFrame columns {} are non numeric'
                        .format(', '.join(non_numeric_cols)))

    return df_residuals.loc[:, names].astype(float).corr(method=method)


def corrmat_from_regionalmeasures(
        regional_measures,
        names,
        covars=None,
        method='pearson'):
    '''
    Return residuals of names columns correcting for the columns in covars_list
    * df is a pandas data frame with subjects as rows and columns including
        brain regions and covariates
    * names is a list of the brain regions you wish to correlate.
    * covars_list is a list of covariates (as df column headings)
        to correct for before correlating the regions.
    * method is the method of correlation passed to pandas.DataFram.corr

    df should be numeric for the columns in names and covars_list
    '''
    # Correct for your covariates
    df_res = create_residuals_df(regional_measures, names, covars)

    # Make your correlation matrix
    M = create_corrmat(df_res, names=names, method=method).values

    return M


def save_mat(M, M_text_name):
    '''
    Save matrix M as a text file
    '''
    # Check to see if the output directory
    # exists, and make it if it does not
    dirname = os.path.dirname(M_text_name)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    # Save the matrix as a text file
    np.savetxt(M_text_name,
               M,
               fmt='%.5f',
               delimiter='\t',
               newline='\n')
