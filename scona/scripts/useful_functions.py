#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

def list_from_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

def read_in_data(
        data,
        names_file,
        covars_file=None,
        centroids_file=None,
        data_as_df=True):
    '''
    Read in data from file paths

    Parameters
    ----------
    data : str
        path to a csv file.
        Read in as a :class:`pandas.DataFrame` unless ``data_as_df=False``
    names_file : str
        path to a text file containing names of brain regions. Read in as list.
    covars_file : str, optional
        a text file containing a list of covariates to correct for. Read in as
        list.
    centroids_file : str, optional
        a text file containing cartesian coordinates of
        brain regions. Should be aligned with names_file so that the ith
        line of centroids_file is the coordinates of the brain region named
        in the ith line of names_file. Read in as list.
    data_as_df : bool, optional
        If False, returns data uses :func:`numpy.loadtext` to import data as
        :class:`numpy.ndarray`

    Returns
    -------
    :class:`pandas.DataFrame`, list, list or None, list or None
        `data, names, covars, centroids`
    '''

    names = list_from_file(names_file)


    # Load covariates
    if covars_file is not None:
        covars_list = list_from_file(covars_file)
    else:
        covars_list = []

    if centroids_file is not None:
        centroids = list(np.loadtxt(centroids_file))
    else:
        centroids = None

    # Load data
    if data_as_df:
        df = pd.read_csv(data)
    else:
        df = np.loadtxt(data)

    return df, names, covars_list, centroids


def write_out_measures(df, output_dir, name, first_columns=[]):
    '''
    Write out a DataFrame as a csv

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A dataframe of measures to write out
    output_dir, name : str
        The output and filename to write out to. Creates output_dir if it does
        not exist
    first_columns : list, optional
        There may be columns you want to be saved on the left hand side of the
        csv for readability. Columns will go left to right in the order
        specified by first_columns, followed by any columns not in
        first_columns.
    '''
    # Make the output directory if it doesn't exist already
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_f_name = os.path.join(output_dir, name)

    # Write the data frame out (with the degree column first)
    new_col_list = first_columns.extend(
                    [col_name
                     for col_name in df.columns
                     if col_name not in first_columns])

    df.to_csv(output_f_name, columns=new_col_list)
