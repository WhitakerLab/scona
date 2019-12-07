import os
import argparse
import textwrap

# Set up parent arg parsers

corrmat_parser = argparse.ArgumentParser(add_help=False)

corrmat_parser.add_argument(
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

corrmat_parser.add_argument(
    dest='names_file',
    type=str,
    metavar='names_file',
    help=textwrap.dedent(('Text file that contains the names of each \
region to be included\n') + ('in the correlation matrix. One region name \
on each line.')))

corrmat_parser.add_argument(
    '--output_name',
    type=str,
    metavar='output_name',
    help=textwrap.dedent(
        ('Pass a file name to save the output correlation matrix.\n') +
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
        ('correlation matrix. One variable name on each line.\n') +
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

networkanalysis_or_scona_parser = argparse.ArgumentParser(add_help=False)
networkanalysis_only_parser = argparse.ArgumentParser(add_help=False)

networkanalysis_only_parser.add_argument(
    dest='corr_mat_file',
    type=str,
    metavar='corr_mat_file',
    help=textwrap.dedent(('Text file (tab or space delimited) that \
    contains the unthresholded\n') + ('matrix with no column or row labels.')))

networkanalysis_only_parser.add_argument(
    dest='names_file',
    type=str,
    metavar='names_file',
    help=textwrap.dedent(('Text file that contains the names of each \
    region, in the same\n') + ('order as the correlation matrix. One region \
    name on each line.')))

networkanalysis_or_scona_parser.add_argument(
    dest='centroids_file',
    type=str,
    metavar='centroids_file',
    help=textwrap.dedent(('Text file that contains the x, y, z \
    coordinates of each region,\n') + ('in the same order as the correlation \
    matrix. One set of three\n') + ('coordinates, tab or space delimited, on each \
    line.')))

networkanalysis_or_scona_parser.add_argument(
    dest='output_dir',
    type=str,
    metavar='output_dir',
    help=textwrap.dedent(('Location in which to save global and nodal \
    measures.')))

networkanalysis_or_scona_parser.add_argument(
    '-c', '--cost',
    type=float,
    metavar='cost',
    help=textwrap.dedent(('Cost at which to threshold the matrix.\n') +
    ('  Default: 10.0')),
    default=10.0)

networkanalysis_or_scona_parser.add_argument(
    '-n', '--n_rand',
    type=int,
    metavar='n_rand',
    help=textwrap.dedent(('Number of random graphs to generate to compare \
    with real network.\n') + ('  Default: 1000')),
    default=1000)

networkanalysis_or_scona_parser.add_argument(
    '-s', '--seed', '--random_seed',
    type=int,
    metavar='seed',
    help=textwrap.dedent(('Set a random seed to pass to the random graph \
    creator.\n') + ('  Default: None')),
    default=None)

# Build specific parsers

corrmat_from_regionalmeasures_parser=argparse.ArgumentParser(
    parents=[corrmat_parser],
    description=(('Generate a structural correlation \
    matrix from an input csv file,\n') + ('a list of \
    region names and (optional) covariates.')),
    formatter_class=argparse.RawTextHelpFormatter)

network_analysis_from_corrmat_parser=argparse.ArgumentParser(
    parents=[networkanalysis_or_scona_parser, networkanalysis_only_parser],
    description=(('Generate a graph as a fixed cost from a non-thresholded\n')
                 + ('matrix and return global and nodal measures.')),
    formatter_class=argparse.RawTextHelpFormatter)

scona_parser = argparse.ArgumentParser(
    parents=[corrmat_parser, networkanalysis_or_scona_parser],
    description="generate network analysis from regional measures.",
    formatter_class=argparse.RawTextHelpFormatter)
