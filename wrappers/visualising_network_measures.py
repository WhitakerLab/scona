#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# on a day when she really should have been replying to emails
# but wanted to make some pretty network pictures instead, April 2017
# Contact: kw401@cam.ac.uk
#=============================================================================

#=============================================================================
# IMPORTS
#=============================================================================
import os
import sys
import argparse
import textwrap

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../SCRIPTS/'))
import make_graphs as mkg
from corrmat_from_regionalmeasures import corrmat_from_regionalmeasures


#=============================================================================
# FUNCTIONS
#=============================================================================
def setup_argparser():
    '''
    Code to read in arguments from the command line
    Also allows you to change some settings
    '''
    # Build a basic parser.
    help_text = (('I AM SOME TEXT\n')+
                 ('I AM SOME MORE TEXT.'))

    sign_off = 'Author: Kirstie Whitaker <kw401@cam.ac.uk>'

    parser = argparse.ArgumentParser(description=help_text,
                                     epilog=sign_off,
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Now add the arguments
    parser.add_argument(dest='regional_measures_file_A',
                            type=str,
                            metavar='regional_measures_file_A',
                            help=textwrap.dedent(('CSV file that contains regional values for each participant in group A.\n')+
                                                 ('Column labels should be the region names or covariate variable\n')+
                                                 ('names. All participants in the file will be included in the\n')+
                                                 ('correlation matrix.')))

    parser.add_argument(dest='regional_measures_file_B',
                            type=str,
                            metavar='regional_measures_file_B',
                            help=textwrap.dedent(('CSV file that contains regional values for each participant in group B.\n')+
                                                 ('Column labels should be the region names or covariate variable\n')+
                                                 ('names. All participants in the file will be included in the\n')+
                                                 ('correlation matrix.')))

    parser.add_argument(dest='names_file',
                            type=str,
                            metavar='names_file',
                            help=textwrap.dedent(('Text file that contains the names of each region, in the same\n')+
                                                 ('order as the two correlation matrices. One region name on each line.')))

    parser.add_argument(dest='centroids_file',
                            type=str,
                            metavar='centroids_file',
                            help=textwrap.dedent(('Text file that contains the x, y, z coordinates of each region,\n')+
                                                 ('in the same order as the two correlation matrices. One set of three\n')+
                                                 ('coordinates, tab or space delimited, on each line.')))

    parser.add_argument(dest='output_dir',
                            type=str,
                            metavar='output_dir',
                            help=textwrap.dedent(('Location in which to save global and nodal measures for the two groups.')))

    parser.add_argument('--nameA',
                            type=str,
                            metavar='nameA',
                            help=textwrap.dedent(('Name of group A')),
                            default='GroupA')

    parser.add_argument('--nameB',
                            type=str,
                            metavar='nameA',
                            help=textwrap.dedent(('Name of group B')),
                            default='GroupB')

    parser.add_argument('-c', '--cost',
                            type=float,
                            metavar='cost',
                            help=textwrap.dedent(('Cost at which to threshold the matrix.\n')+
                                                 ('  Default: 10.0')),
                            default=10.0)

    parser.add_argument('-n', '--n_perm',
                            type=int,
                            metavar='n_perm',
                            help=textwrap.dedent(('Number of permutations of the data to compare with the real groupings.\n')+
                                                 ('  Default: 1000')),
                            default=1000)

    parser.add_argument('--names_308_style',
                            action='store_true',
                            help=textwrap.dedent(('Include this flag if your names are in the NSPN 308\n')+
                                                 ('parcellation style (which means you have 41 subcortical regions)\n')+
                                                 ('that are still in the names and centroids files and that\n')+
                                                 ('the names are in <hemi>_<DK-region>_<part> format.\n')+
                                                 ('  Default: False')),
                            default=False)

    arguments = parser.parse_args()

    return arguments, parser


if __name__ == "__main__":

    # Read in the command line arguments
    arg, parser = setup_argparser()

    # View the correlation matrix
    create_real_corrmats(arg.regional_measures_file_A,
                                arg.nameA,
                                arg.regional_measures_file_B,
                                arg.nameB,
                                arg.names_file,
                                arg.covars_file,
                                arg.output_dir,
                                arg.names_308_style)

    network_analysis_from_corrmat(arg.corr_mat_file,
                                      arg.names_file,
                                      arg.centroids_file,
                                      arg.output_dir,
                                      cost=arg.cost,
                                      n_rand=arg.n_rand,
                                      names_308_style=arg.names_308_style)

#=============================================================================
# Wooo! All done :)
#=============================================================================
