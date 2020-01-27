#!/usr/bin/env python
"""
Tools to create a correlation matrix from regional measures
"""
# Essential package imports
import os
import numpy as np
from scona.stats_functions import residuals


def get_non_numeric_cols(df):
    '''
    returns the columns of df whose dtype is not numeric (e.g not a subtype of
    :class:`numpy.number`)

    Parameters
    ----------
    df : :class:`pandas.DataFrame`

    Returns
    -------
    list
        non numeric columns of df
    '''
    numeric = np.fromiter((np.issubdtype(y, np.number) for y in df.dtypes),
                          bool)
    non_numeric_cols = np.array(df.columns)[~numeric]
    return non_numeric_cols


def generate_windows(df, window_var, window_size, shuffle=False, seed=None):
    '''
 Parameters
    ----------
    df : :class:`pandas.DataFrame`
        a pandas DataFrame with individual brain scans as rows, and 
        columns including brain regions and covariates. The columns in
        names and covars_list should be numeric.
    names : list
        a list of the brain regions whose correlation you want to assess
    window_var : str
        the name of the column in df from which to
        construct sliding windows.
    window_size : int or float
        the size (number of subjects) of each sliding window. 
        A decimal value between 0 and 1 will be interpreted as a 
        proportion of the whole cohort. E.g if window_size is 0.1, and
        the cohort is 100 subjects each window will contain 10 subjects.
    window_overlap : int or float
        the number of subjects in the overlap between two consecutive
        windows. If window_overlap is a decimal between 0 and
        1(not inclusive) then the intersection of two consecutive
        windows will be window_overlap*(size of first window).
    odd_sized_bin : "last" or "first", optional
        If it is not possible to construct equally sized windows,
        choose either the last or the first window to have a 
        different size to the others. Default "last".
    '''
    if window_var not in get_non_numeric_columns(df):
        raise TypeError("`window_var` must index a numeric column")
    # calculate window sizes and overlaps
    if window_size <= 1:
        window_size = window_size*len(df)
    if window_overlap < 1:
        window_overlap = window_overlap*window_size
    
    if shuffle:
        sorted_df = df.sample(frac=1, random_state=seed)
    else:
        sorted_df = df.sort_values(by=[window_var])
    moving_window_df = {}
    for t in range(df.shape[0] - window_size + 1):
        moving_window_df[t] = df.truncate(before=t, after=window_size+t-1)
    return moving_window_df
    

def split_groups(df, group_var, shuffle=False, seed=None):
    '''
    Separate a dataframe into different participant groups.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
    group_var : str
        A string indexing a column of `df` which contains the group coding
        of each participant
    shuffle : bool, optional
        If True is passed split_groups will randomly assign each participant
        to a group from the original group_var column, preserving the size
        of the original groups. 
        This is achieved by drawing values from the group_var column without
        replacement. This does not modify the dataframe `df`.

    Returns
    -------
    dict
        A dictionary mapping values of the group_var column to a
        :class:`pandas.DataFrame` of correspondingly coded participants.
    '''
    if group_var not in df.columns:
        raise ValueError(
            "The group_var argument '{}' does not index a column in this dataframe.")
    split_dict = {}

    if shuffle is False:
        for value in set(df.loc[:, group_var].values):
            split_dict[value] = df.loc[df[group_var] == value, :]

    elif shuffle is True:
        # if shuffle is true, create a new dataframe, with a new column,
        # identical to the group_var column, only randomly permuted.
        if seed is not None:
            np.random.seed(seed)
        df = df.copy()
        group_rand = "rand_{}".format(group_var)
        df[group_rand] = np.random.permutation(df.loc[:, group_var].values)
        for value in set(df.loc[:, group_rand].values):
            split_dict[value] = df.loc[df[group_rand] == value, :]
        # and clean up by deleting the new column
        del df[group_rand]
        
    return split_dict


def create_residuals_df(df, names, covars=[]):
    '''
    Calculate residuals of columns specified by names, correcting for the
    columns in covars_list.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A pandas data frame with subjects as rows and columns including
        brain regions and covariates. Should be numeric for the columns in
        `names` and `covars`.
    names : list
        A list of the brain regions you wish to correlate.
    covars: list, optional
        A list of covariates to correct for before correlating
        the regional measures. Each element should correspond to a
        column heading in `df`.
        Default is an empty list.

    Returns
    -------
    :class:`pandas.DataFrame`
        Residuals of columns `names` of `df`, correcting for `covars`

    Raises
    ------
    TypeError
        if there are non numeric entries in the columns specified by `names` or
        `covars`
    '''
    # Raise TypeError if any of the relevant columns are nonnumeric
    non_numeric_cols = get_non_numeric_cols(df[names+covars])
    if non_numeric_cols.size > 0:
        raise TypeError('DataFrame columns {} are non numeric'
                        .format(', '.join(non_numeric_cols)))

    # Make a new data frame that will contain
    # the residuals for each column after correcting for
    # the covariates in covars
    df_res = df[names].copy()

    if len(covars) > 1:
        x = np.vstack([df[covars]])
    elif len(covars) == 1:
        x = df[covars]
    else:
        x = np.ones_like(df.iloc[:, 0])

    # Calculate the residuals
    for name in names:
        df_res.loc[:, name] = residuals(x.T, df.loc[:, name])

    # Return the residuals data frame
    return df_res


def create_corrmat(df_res, names=None, method='pearson'):
    '''
    Correlate over the rows of `df_res`

    Parameters
    ----------
    df_res : :class:`pandas.DataFrame`
        `df_res` contains structural data about regions of the brain with
        subjects as rows after correction for any covariates of no interest.
    names : list, optional
        The brain regions you wish to correlate over. These will become nodes
        in your graph. If `names` is None then all columns are included.
        Default is `None`.
    methods : string, optional
        The method of correlation passed to :func:`pandas.DataFrame.corr`.
        Default is pearsons correlation (`pearson`).

    Returns
    -------
    :class:`pandas.DataFrame`
        A correlation matrix.

    Raises
    ------
    TypeError
        If there are non numeric entries in the columns in `df_res` specified
        by `names`.
    '''
    if names is None:
        names = df_res.columns

    # Raise TypeError if any of the relevant columns are nonnumeric
    non_numeric_cols = get_non_numeric_cols(df_res)
    if non_numeric_cols.size > 0:
        raise TypeError('DataFrame columns {} are non numeric'
                        .format(', '.join(non_numeric_cols)))

    return df_res.loc[:, names].astype(float).corr(method=method)


def corrmat_from_regionalmeasures(
        regional_measures,
        names,
        covars=None,
        method='pearson'):
    '''
    Calculate the correlation of `names` columns over the rows of
    `regional_measures` after correcting for covariance with the columns in
    `covars`

    Parameters
    ----------
    regional_measures : :class:`pandas.DataFrame`
        a pandas DataFrame with individual brain scans as rows, and 
        columns including brain regions and covariates. The columns in
        names and covars_list should be numeric.
    names : list
        a list of the brain regions whose correlation you want to measure
    covars: list, optional
        covars is a list of covariates (as DataFrame column headings)
        to correct for before correlating brain regions.
    method : string, optional
        the method of correlation passed to :func:`pandas.DataFramecorr`

    Returns
    -------
    :class:`pandas.DataFrame`
        A correlation matrix with rows and columns keyed by `names`
    '''
    # Correct for your covariates
    df_res = create_residuals_df(regional_measures, names, covars)

    # Make your correlation matrix
    M = create_corrmat(df_res, names=names, method=method).values

    return M

def corrmat_by_group(
        regional_measures,
        names,
        group_var,
        covars=None,
        method='pearson',
        shuffle=False,
        seed=None):
    '''
    Separate `regional_measures` rows by their `group_var` value. 
    Create a dictionary mapping each value of the `group_var` column
    to a correlation matrix.

    Parameters
    ----------
    regional_measures : :class:`pandas.DataFrame`
        a pandas DataFrame with subjects as rows, and columns representing
        brain regions, covariates and group codings. Should be numeric for
        the columns in names and covars_list.
    names : list
        a list of the brain regions you wish to correlate
    group_var : str
        a string indexing a column in regional_measure containing the
        group coding data.
    covars: list, optional
        covars is a list of covariates (as DataFrame column headings)
        to correct for before correlating the regions.
    methods : string, optional
        the method of correlation passed to :func:`pandas.DataFrame.corr`
    shuffle : bool, optional
        if True, a random permutation of the group_var column will be
        used to assign group codings.

    Returns
    -------
    :class:`pandas.DataFrame`
        A correlation matrix with rows and columns keyed by `names`
    '''
    # split dataframe by group coding
    df_by_group = split_groups(
        regional_measures, group_var, shuffle=shuffle, seed=seed)
    
    matrix_by_group=dict()
    # iterate over groups to create correlation matrices
    for group_code, group_df in df_by_group:
        M = mcm.corrmat_from_regionalmeasures(
            group_df, names, covars=covars_list, method=method)
        matrix_by_group[group_code] = M

    return matrix_by_group

def corrmat_by_window(
        regional_measures,
        names,
        window_var,
        window_size,
        covars=None,
        method='pearson',
        shuffle=False,
        seed=None):
    '''
    Bin `regional_measures` rows by their value in `window_var` column. 
    Return 
    Create a correlation matrix of the rows selected by...

    Parameters
    ----------
    regional_measures : :class:`pandas.DataFrame`
        a pandas DataFrame with subjects as rows, and columns representing
        brain regions, covariates and group codings. Should be numeric for
        the columns in names and covars_list.
    names : list
        a list of the brain regions you wish to correlate
    window_var : str
        a string indexing a column in regional_measures by which to
        bin rows
    window_size : int
        the number of rows to include in each window
    covars: list, optional
        covars is a list of covariates (as DataFrame column headings)
        to correct for before correlating the regions.
    methods : string, optional
        the method of correlation passed to :func:`pandas.DataFrame.corr`
    shuffle : bool, optional
        if True, a random permutation of the group_var column will be
        used to assign group codings.

    Returns
    -------
    :class:`pandas.DataFrame`
        A correlation matrix with rows and columns keyed by `names`
    '''
    # create moving window of dataframe
    df_by_window = generate_windows(
        regional_measures, window_var, window_size, shuffle=shuffle, seed=seed)

    # iterate over windows to create correlation matrices
    matrix_by_window = {}
    for t, window in moving_window_df:
        M = mcm.corrmat_from_regionalmeasures(
            window, names, covars=covars_list, method=method)
        matrix_by_window[t] = M

    return matrix_by_window

def save_mat(M, name):
    '''
    Save matrix M as a text file

    Parameters
    ----------
    M : array
    name : str
        name of the output directory
    '''
    # Check to see if the output directory
    # exists, and make it if it does not
    dirname = os.path.dirname(name)

    if not os.path.isdir(dirname) and dirname != "":
        os.makedirs(dirname)

    # Save the matrix as a text file
    np.savetxt(name,
               M,
               fmt='%.5f',
               delimiter='\t',
               newline='\n')
