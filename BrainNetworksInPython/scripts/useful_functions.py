#!/usr/bin/env python

import pandas as pd
import numpy as np
import os


def read_in_data(
        data,
        names_file,
        covars_file=None,
        centroids_file=None,
        names_308_style=False,
        data_as_df=True):
    '''
    Read in the data from the three input files:
        * data : a csv file
        * names_file : a text file containing names of brain regions. One name
            per line of text.
        * covars_file : a text file containing a list of covariates to correct
            for. One covariate per line of text.
        * centroids_file : a text file containing cartesian coordinates of
            brain regions. Should be aligned with names_file such that the ith
            line of centroids_file is the coordinates of the brain region named
            in the ith line of names_file.
        * names_308_style : If the names are in 308 style then drop the first
            41 entries from the names file.
    '''
    # Load names
    with open(names_file) as f:
        names = [line.strip() for line in f]
    # If you have your names in names_308_style you need to strip the
    # first 41 items
    if names_308_style:
        names = names[41:]

    # Load covariates
    if covars_file is not None:
        with open(covars_file) as f:
            covars_list = [line.strip() for line in f]
    else:
        covars_list = []

    if centroids_file is not None:
        centroids = np.loadtxt(centroids_file)
        # If you have your names in names_308_style you need to strip the
        # first 41 items
        if names_308_style:
            names = names[41:]
            centroids = centroids[41:, :]

    # Load data
    if data_as_df:
        df = pd.read_csv(data)
        # You may also have to strip the words "thickness" from the
        # end of the names in the data frame
        if names_308_style:
            df.columns = [col.rsplit('_thickness', 1)[0] for col in df.columns]
    else:
        df = np.loadtxt(data)

    return df, names, covars_list, centroids, names_308_style


def write_out_measures(df, output_dir, name, first_columns=[]):
    '''
    Write out a DataFrame as a csv
    * df is the DataFrame to write
    * output_dir is the output directory to write to
    * name is the filename to write to
    * first_columns is a list of columns to put at the left hand side of the
        csv file. e.g subject names
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
