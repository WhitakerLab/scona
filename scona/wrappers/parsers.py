import os
import argparse
import textwrap
from scona.wrappers.scona import standard_analysis, groupwise_analysis, movingwindow_analysis, corrmat_from_regionalmeasures, network_analysis_from_corrmat


# Set up parent arg parsers

corrmat_parser = argparse.ArgumentParser(add_help=False)
network_analysis_parser = argparse.ArgumentParser(add_help=False)
general_parser = argparse.ArgumentParser(add_help=False)

# Fill parent parsers

corrmat_parser.add_argument(
    dest='regional_measures_file',
    type=str,
    metavar='regional_measures_file',
    help=textwrap.dedent('''
    Path (relative) to .csv file reporting regional measures at each 
    brain region for each participant. Column labels should include
    the region names and covariate variables. All subjects (rows) in 
    regional_measures_file will be correlated over'''))


corrmat_parser.add_argument(
    '--output_name',
    type=str,
    metavar='output_name',
    help=textwrap.dedent('''
        Pass a (relative) file name to save the output correlation matrix.
        If the output directory does not yet exist it will be created.'''),
    default=None)

corrmat_parser.add_argument(
    '--covars_file',
    type=str,
    metavar='covars_file',
    help=textwrap.dedent('''
        Text file containing a list of covariates (as column headings
        from regional_measures_file) to be accounted for when calculating
        correlation. One variable name on each line. Relative path.
        Default: None'''),
    default=None)

corrmat_parser.add_argument(
    '--method',
    type=str,
    metavar='method',
    help=textwrap.dedent('''
        Flag submitted to pandas.DataFrame.corr().
        options are "pearson", "spearman", "kendall"'''),
    default='pearson')

network_analysis_parser.add_argument(
    dest='centroids_file',
    type=str,
    metavar='centroids_file',
    help=textwrap.dedent('''
    Relative path to text file that contains the x, y, z 
    coordinates of each region, in the same order as the 
    names in names_file. One set of three coordinates, 
    tab or space delimited, on each line.'''))

network_analysis_parser.add_argument(
    '-c', '--cost',
    type=float,
    metavar='cost',
    help=textwrap.dedent('Cost at which to threshold the matrix.\n' +
    'Default: 10.0'),
    default=10.0)

network_analysis_parser.add_argument(
    '-n', '--n_rand',
    type=int,
    metavar='n_rand',
    help=textwrap.dedent('''
    Number of random graphs to generate to compare 
    with real network.\n  Default: 1000'''),
    default=1000)

network_analysis_parser.add_argument(
    '-s', '--seed',
    type=int,
    metavar='seed',
    help=textwrap.dedent('''
    Set a random seed to pass to the random graph creator. 
    Default: None'''),
    default=None)

general_parser.add_argument(
    dest='names_file',
    type=str,
    metavar='names_file',
    help=textwrap.dedent('''
    Text file listing the names of relevant brain regions. One region 
    name on each line.'''))

general_parser.add_argument(
    '--output_dir',
    type=str,
    metavar='output_dir',
    help=textwrap.dedent('''
    Relative path to a directory in which to save output 
    measures.'''),
    default=None)

# Build specific parsers

scona_parser = argparse.ArgumentParser(
    description="generate network analysis from regional measures.",
    formatter_class=argparse.RawTextHelpFormatter)

subparsers = scona_parser.add_subparsers()

# ============================================================================
# subparser to generate correlation matrix
#
# calls scona.wrappers.corrmat_from_regionalmeasures.corrmat_from_regionalmeas
# ures
# ============================================================================
corrmat_only_parser = subparsers.add_parser(
    'corrmat',
    help="create a correlation matrix from regional measures",
    description=(textwrap.dedent(
    '''
    Read in regional measures, names and covariates files to compute
    and return a structural covariance matrix, or write it to 
    output_name.
    The structural covariance matrix is the pairwise correlation of 
    the columns given in names_file over the rows of regional_measures,
    after correcting for covariance with the columns in covars_file.
    ''')),
    parents=[corrmat_parser, general_parser])

corrmat_only_parser.set_defaults(func=corrmat_from_regionalmeasures)

# ===================================================================
# subparser to do network analysis from corrmat
#
# calls scona.wrappers.network_analysis_from_corrmat.network_analysis
# _from_corrmat
# ===================================================================

nafc_parser = subparsers.add_parser(
    'from_corrmat',
    help='perform standard scona analysis on an existing correlation matrix',
    description=textwrap.dedent('''
    Run the standard scona network analysis on an existing corrmat_file,
    interpreted as a weighted graph. 
    This analysis thresholds corrmat at the desired cost to create a 
    binary network and calculates network measures (for more details
    on network measures see...).
    For the purposes of comparison this analysis also generates a number
    of random graphs via edge swapping (see :func:`networkx.double_edge_swap`).

    Writes
    ------
    * A dataframe reporting the nodal measures for the
    nodes of corrmat
    * A dataframe reporting the global measures of corrmat and
    all random graphs
    * A dataframe reporting the rich club, at every
    degree, of corrmat and all random graphs'''),
    parents=[network_analysis_parser, general_parser])

nafc_parser.add_argument(
    dest='corrmat_file',
    type=str,
    metavar='corrmat_file',
    help=textwrap.dedent('''
    Relative path to text file (tab or space delimited) that
    contains the unthresholded correlation matrix with no 
    column or row labels.'''))

nafc_parser.set_defaults(func=network_analysis_from_corrmat)

# =======================================================================
# subparser to do full analysis
# =======================================================================
simple_parser = subparsers.add_parser(
    'standard_analysis',
    help="perform standard scona analysis from regional_measures_file",
    description=textwrap.dedent('''
    Create a structural covariance analysis network from 
    regional_measures_file and run the standard scona network analysis
    on it.

    To create the structural covariance network from regional_measures_file,
    scona calculates the pairwise correlations of the columns listed in
    names_file over the rows of regional_measures_file, correcting for
    covariance with the columns listed in covars_file.
 
    scona thresholds the resulting matrix at the desired cost to create a 
    binary network and calculates network measures, described...

    For the purposes of comparison this analysis also generates a number
    of random graphs via edge swapping (see :func:`networkx.double_edge_swap`)
    and reports global measures and rich club measures on these.

    Writes
    ------
    * A dataframe reporting the nodal measures for the
    nodes of the structural covariance network.
    * A dataframe reporting the global measures of the
    structural covariance network and random graphs.
    * A dataframe reporting the rich club, at every
    degree, of each network.
    '''),
    parents=[corrmat_parser,
             network_analysis_parser,
             general_parser])

simple_parser.set_defaults(func=standard_analysis)
# =======================================================================
# subparser to do groupwise analysis
# ======================================================================

groupwise_parser = subparsers.add_parser(
    'groupwise',
    help='Perform a groupwise analysis on regional_measures_file',
    parents=[corrmat_parser, network_analysis_parser, general_parser])

groupwise_parser.add_argument(
    dest='group_var',
    type=str,
    metavar='group_var',
    help=textwrap.dedent(
        ("Networks will be constructed per participant group, as\
        indexed by the group_var column in the regional_measures_file")))

groupwise_parser.add_argument(
    '--n_shuffle',
    type=int,
    metavar='n_shuffle',
    help=textwrap.dedent('''
    number of comparison networks to create
    by shuffling group_var column and repeating analysis'''),
    default=1000)

groupwise_parser.set_defaults(func=groupwise_analysis)

movingwindow_parser = subparsers.add_parser(
    'movingwindow',
    help='Perform a moving window analysis on regional_measures_file',
    parents=[corrmat_parser, network_analysis_parser, general_parser])

movingwindow_parser.add_argument(
    dest='window_by',
    type=str,
    metavar='window_by',
    help=textwrap.dedent('''
    a variable by which to window the participants.
    Must index a column in regional_measures_file.'''))

movingwindow_parser.set_defaults(func=movingwindow_analysis)

        
def main():
    args = scona_parser.parse_args()
    args.func(args)
    
