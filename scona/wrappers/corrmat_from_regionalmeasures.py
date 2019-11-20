#!/usr/bin/env python

# ============================================================================
# Created by Kirstie Whitaker
# at Hot Numbers coffee shop on Trumpington Road in Cambridge, September 2016
# Contact: kw401@cam.ac.uk
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================

import scona.make_corr_matrices as mcm
from scona.scripts.useful_functions import read_in_data
from scona.wrappers.parsers import corrmat_from_regionalmeasures_parser

def corrmat_from_regionalmeasures(regional_measures_file,
                                  output_name=None,
                                  covariates=None,
                                  names_list=None,
                                  names_file=None,
                                  covars_file=None,
                                  method='pearson',
                                  group_var=None):
    '''
    Read in regional measures, names and covariates files to compute
    and return a correlation matrix and write it to output_name.

    Parameters
    ----------
        regional_measures_file : str
            A csv file of regional measures data containing any required
            brain regions and covariates as columns. 
            The first row of regional_measures should be column headings,
            with subsequent rows corresponding to individual subjects.
        names_file : str, optional
            a text file containing names of brain regions. One name
            per line of text. These names will specify the columns
            in the regional_measures_file to correlate through. 
            Failing to pass an argument either to both names_file and
            names_list will cause an exception to be raised.
        names_list : list, optional
            a list of names of brain regions. These names will specify
            the columns in the regional_measures_file to correlate
            through. 
            Failing to pass an argument to both names_file and
            names_list will cause an exception to be raised.
        covars_file : str, optional
            a text file containing a list of covariates to account
            for. One covariate per line of text. These covariates 
            should index columns in regional_measure_file.
        covariates : list, optional
            a list of covariates to account for. These covariates
            should index columns in regional_measures_file.    
        output_name : str, optional
            a file name to save output matrix to. 
            If the output directory does not yet exist it will be
            created.
            If an argument is passed to group_var the group correlation
            matrices will be written to a file with the corresponding
            group coding appended.
            If no name is passed the output matrix (or matrices) will
            be returned but not saved.
        group_var : str, optional
            This variable can be used to specify a column in 
            regional_measures_file containing the group coding. A
            correlation matrix will be produced for each patient group.
            

    Returns
    -------
    :class:`pandas.DataFrame or dict
        A correlation matrix, or a dictionary of group codings mapping to
        correlation matrices. 
    '''
    if names is None and names_file is None:
        raise Exception(
            "You must pass the names of brain regions you want to examine.\n"
            + "Use either the `names_list` or the `names_file` argument.")
    # Read in the data
    df, names, covars_list, *a = read_in_data(
        regional_measures_file,
        names_file,
        covars_file=covars_file)

    if covariates is not None:
        covars_list=covariates

    if group_var is None:
        # create correlation matrix
        M = mcm.corrmat_from_regionalmeasures(
            df, names, covars=covars_list, method=method)
        if output_name is not None:
            # Save the matrix
            mcm.save_mat(M, output_name)
        return M

    else:
        # split dataframe by group coding
        df_by_group = split_groups(df, group_var)
        matrix_by_group=dict()
        # iterate over groups
        for group_code, group_df in df_by_group:
            # create correlation matrix
            M = mcm.corrmat_from_regionalmeasures(
            group_df, names, covars=covars_list, method=method)
            if output_name is not None:
                # Save the matrix
                mcm.save_mat(M, output_name+str(group_code))
            matrix_by_group[group_code] = M
        return matrix_by_group
            

def main():
    # Read in the command line arguments
    arg = corrmat_from_regionalmeasures_parser.parse_args()

    # Now run the main function :)
    corrmat_from_regionalmeasures(
        arg.regional_measures_file,
        names_file=arg.names_file,
        output_name=arg.output_name,
        covariates=arg.covariates,
        covars_file=arg.covars_file,
        method=arg.method,
        group_var=arg.group_var)

    
if __name__ == "__main__":
    main()
