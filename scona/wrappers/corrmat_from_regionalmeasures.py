#!/usr/bin/env python

# ============================================================================
# Created by Kirstie Whitaker
# at Hot Numbers coffee shop on Trumpington Road in Cambridge, September 2016
# Contact: kw401@cam.ac.uk
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import textwrap

import scona.make_corr_matrices as mcm
from scona.scripts.useful_functions import read_in_data


def setup_argparser():
    # Build a basic parser.
    help_text = (('Generate a structural correlation \
    matrix from an input csv file,\n') + ('a list of \
    region names and (optional) covariates.'))

    sign_off = 'Author: Kirstie Whitaker <kw401@cam.ac.uk>'

    parser = argparse.ArgumentParser(
        description=help_text,
        epilog=sign_off,
        formatter_class=argparse.RawTextHelpFormatter)

    # Now add the arguments
    parser.add_argument(
        dest='regional_measures_file',
        type=str,
        metavar='regional_measures_file',
        help=textwrap.dedent(
            ('A CSV file containing regional values for each participant.\
\n') +
            ('Column labels should include the region names or covariate \
variable\n') +
            ('names. All participants in the file will be correlated over\n')))

    parser.add_argument(
        dest='names_file',
        type=str,
        metavar='names_file',
        help=textwrap.dedent(('Text file that contains the names of each \
region to be included\n') + ('in the correlation matrix. One region name \
on each line.')))

    parser.add_argument(
        dest='output_name',
        type=str,
        metavar='output_name',
        help=textwrap.dedent(
            ('File name of the output correlation matrix.\n') +
            ('If the output directory does not yet exist it will be \
created.\n') +
            ('If an argument is passed to `--group_by`, output names \
will have the appropriate group coding appended to them'
        ))

    parser.add_argument(
        'covariates',
        type=list,
        metavar='covariates',
        help=textwrap.dedent(
            ('List of variables that should be covaried for before \
the creation\n') +
            ('of the correlation matrix. Overrides `--covars_file` \
argument below\n') +
            ('Default: None')),
        default=None)

    parser.add_argument(
        '--covars_file',
        type=str,
        metavar='covars_file',
        help=textwrap.dedent(
            ('Text file that contains the names of variables that \
should be\n') +
            ('covaried for each regional measure before the creation \
of the\n') +
            ('correlation matrix. One variable name on each line.\n') +
            ('  Default: None')),
        default=None)

    parser.add_argument(
        '--corr_method',
        type=str,
        metavar='method',
        help=textwrap.dedent(
            ('Flag submitted to pandas.DataFrame.corr().\n') +
            ('options are "pearson", "spearman", "kendall"')),
        default='pearson')

    parser.add_argument(
        '--group_by',
        type=str,
        metavar='group_var',
        help=textwrap.dedent(
            ('This variable can be used to specify a column in \
`regional_measures_file`\n') +
            ('containing the group coding. A correlation matrix will be produced for each patient group.\n') +
            ('  Default: None')),
        default=None)
    
    arguments = parser.parse_args()

    return arguments, parser


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
            


if __name__ == "__main__":

    # Read in the command line arguments
    arg, parser = setup_argparser()

    # Now run the main function :)
    corrmat_from_regionalmeasures(
        arg.regional_measures_file,
        names_file=arg.names_file,
        output_name=arg.output_name,
        covariates=arg.covariates,
        covars_file=arg.covars_file,
        method=arg.method,
        group_var=arg.group_var)

# ============================================================================
# Wooo! All done :)
# ============================================================================
