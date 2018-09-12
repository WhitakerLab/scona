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
            ('CSV file that contains regional values for each participant.\
\n') +
            ('Column labels should be the region names or covariate \
variable\n') +
            ('names. All participants in the file will be included in the\n') +
            ('correlation matrix.')))

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
created.')))

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
        '--method',
        type=str,
        metavar='method',
        help=textwrap.dedent(
            ('Flag submitted to pandas.DataFrame.corr().\n') +
            ('options are "pearson", "spearman", "kendall"')),
        default='pearson')

    arguments = parser.parse_args()

    return arguments, parser


def corrmat_from_regionalmeasures(regional_measures_file,
                                  names_file,
                                  output_name,
                                  covars_file=None,
                                  method='pearson'):
    '''
    Read in regional measures, names and covariates files to compute
    correlation matrix and write it to output_name.

    Parameters:
        * regional_measures_file : a csv containing data for some regional
            measures with brain regions and covariates as columns and subjects
            as rows. The first row of regional_measures should be column
            headings.
        * names_file : a text file containing names of brain regions. One name
            per line of text. These names key columns in df to correlate over.
        * covars_file : a text file containing a list of covariates to account
            for. One covariate per line of text. These names key columns in df.
        * output_name : file name to save output matrix to.
    '''
    # Read in the data
    df, names, covars_list, *a = read_in_data(
        regional_measures_file,
        names_file,
        covars_file=covars_file)

    M = mcm.corrmat_from_regionalmeasures(
        df, names, covars=covars_list, method=method)

    # Save the matrix
    mcm.save_mat(M, output_name)


if __name__ == "__main__":

    # Read in the command line arguments
    arg, parser = setup_argparser()

    # Now run the main function :)
    corrmat_from_regionalmeasures(
        arg.regional_measures_file,
        arg.names_file,
        arg.output_name,
        covars_file=arg.covars_file,
        method=arg.method)

# ============================================================================
# Wooo! All done :)
# ============================================================================
