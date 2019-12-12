import os
import argparse
import textwrap

from scona.wrappers.corrmat_from_regionalmeasures import corrmat_from_regionalmeasures
from scona.wrappers.network_analysis_from_corrmat import network_analysis_from_corrmat
from scona.wrappers.scona import standard_analysis, groupwise_analysis, movingwindow_analysis


# Set up parent arg parsers

corrmat_parser = argparse.ArgumentParser(add_help=False)
network_analysis_parser = argparse.ArgumentParser(add_help=False)
name_parser = argparse.ArgumentParser(add_help=False)

# Fill parent parsers

corrmat_parser.add_argument(
    dest='regional_measures_file',
    type=str,
    metavar='regional_measures_file',
    help=textwrap.dedent(
        ('Relative path to CSV file reporting regional values for each participant.\
        \n') +
        ('Column labels should be the region names or covariate \
        variable\n') +
        ('names. All participants in the file will be included in the\n') +
        ('correlation matrix.')))



corrmat_parser.add_argument(
    '--output_name',
    type=str,
    metavar='output_name',
    help=textwrap.dedent(
        ('Pass a (relative) file name to save the output correlation matrix.\n') +
        ('If the output directory does not yet exist it will be \
        created.')),
    default=None)

corrmat_parser.add_argument(
    '--covars_file',
    type=str,
    metavar='covars_file',
    help=textwrap.dedent(
        ('Text file that contains the names of variables that \
        should be\n') +
        ('covaried for each regional measure before the creation \
        of the\n') +
        ('correlation matrix. One variable name on each line. \
        Relative path.\n') +
        ('  Default: None')),
    default=None)

corrmat_parser.add_argument(
    '--method',
    type=str,
    metavar='method',
    help=textwrap.dedent(
        ('Flag submitted to pandas.DataFrame.corr().\n') +
        ('options are "pearson", "spearman", "kendall"')),
    default='pearson')

network_analysis_parser.add_argument(
    dest='centroids_file',
    type=str,
    metavar='centroids_file',
    help=textwrap.dedent(('Relative path to text file that contains the x, y, z \
    coordinates of each region,\n') + ('in the same order as the correlation \
    matrix. One set of three\n') + ('coordinates, tab or space delimited, on each \
    line.')))

network_analysis_parser.add_argument(
    dest='output_dir',
    type=str,
    metavar='output_dir',
    help=textwrap.dedent(('Relative path to a directory in which to save global and nodal \
    measures.')))

network_analysis_parser.add_argument(
    '-c', '--cost',
    type=float,
    metavar='cost',
    help=textwrap.dedent(('Cost at which to threshold the matrix.\n') +
    ('  Default: 10.0')),
    default=10.0)

network_analysis_parser.add_argument(
    '-n', '--n_rand',
    type=int,
    metavar='n_rand',
    help=textwrap.dedent(('Number of random graphs to generate to compare \
    with real network.\n') + ('  Default: 1000')),
    default=1000)

network_analysis_parser.add_argument(
    '-s', '--seed',
    type=int,
    metavar='seed',
    help=textwrap.dedent(('Set a random seed to pass to the random graph \
    creator.\n') + ('  Default: None')),
    default=None)

name_parser.add_argument(
    dest='names_file',
    type=str,
    metavar='names_file',
    help=textwrap.dedent(('Text file that contains the names of each \
    region, in the same\n') + ('order as the correlation matrix. One region \
    name on each line.')))

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
    help=('Generate a structural correlation \
    matrix from an input csv file, a list of \
    region names and (optional) covariates.'),
    parents=[corrmat_parser, name_parser])

corrmat_only_parser.add_argument(
    '--group_var',
    type=str,
    metavar='group_var',
    help=textwrap.dedent(
        ("If a group_var is passed correlation matrices will be constructed per \
        participant group, as indexed by the group_var column in the \
regional measures file")),
    default=None)

corrmat_only_parser.set_defaults(func=corrmat_from_regionalmeasures)

# ===================================================================
# subparser to do network analysis from corrmat
#
# calls scona.wrappers.network_analysis_from_corrmat.network_analysis
# _from_corrmat
# ===================================================================

nafc_parser = subparsers.add_parser(
    'from_corrmat',
    help='Generate a graph as a fixed cost from an \
    existing correlation matrix and return global \
    and nodal measures.',
    parents=[network_analysis_parser, name_parser])

nafc_parser.add_argument(
    dest='corrmat_file',
    type=str,
    metavar='corrmat_file',
    help=textwrap.dedent(('Relative path to text file (tab or space delimited) that \
    contains the unthresholded\n') + ('correlation matrix with no column or row labels.')))

nafc_parser.set_defaults(func=network_analysis_from_corrmat)

# =======================================================================
# subparser to do full analysis
# =======================================================================
simple_parser = subparsers.add_parser(
    'standard_analysis',
    help='Standard analysis ...',
    parents=[corrmat_parser,
             network_analysis_parser,
             name_parser])

simple_parser.set_defaults(func=standard_analysis)
# =======================================================================
# subparser to do groupwise analysis
# ======================================================================

groupwise_parser = subparsers.add_parser(
    'groupwise',
    help='Perform a groupwise analysis on regional_measures_file',
    parents=[corrmat_parser, network_analysis_parser, name_parser])

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
    help=textwrap.dedent(("number of comparison networks to create\
    by shuffling group_var column and repeating analysis")),
    default=1000)

groupwise_parser.set_defaults(func=groupwise_analysis)

movingwindow_parser = subparsers.add_parser(
    'movingwindow',
    help='Perform a moving window analysis on regional_measures_file',
    parents=[corrmat_parser, network_analysis_parser, name_parser])

movingwindow_parser.add_argument(
    dest='window_by',
    type=str,
    metavar='window_by',
    help=textwrap.dedent(("a variable by which to window the participants.\
    Must index a column in regional_measures_file.")))

movingwindow_parser.set_defaults(func=movingwindow_analysis)

        
def main():
    scona_parser.parse_args()
