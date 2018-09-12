#!/usr/bin/env python

# =============================================================================
# Created by Kirstie Whitaker
# at Neurohackweek 2016 in Seattle, September 2016
# Contact: kw401@cam.ac.uk
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import os
import argparse
import textwrap

import scona as scn
from scona.scripts.useful_functions import read_in_data, \
    write_out_measures

# =============================================================================
# FUNCTIONS
# =============================================================================


def setup_argparser():
    '''
    Code to read in arguments from the command line
    Also allows you to change some settings
    '''
    # Build a basic parser.
    help_text = (('Generate a graph as a fixed cost from a non-thresholded\n')
                 + ('matrix and return global and nodal measures.'))

    sign_off = 'Author: Kirstie Whitaker <kw401@cam.ac.uk>'

    parser = argparse.ArgumentParser(
        description=help_text,
        epilog=sign_off,
        formatter_class=argparse.RawTextHelpFormatter)

    # Now add the arguments
    parser.add_argument(
        dest='corr_mat_file',
        type=str,
        metavar='corr_mat_file',
        help=textwrap.dedent(('Text file (tab or space delimited) that \
contains the unthresholded\n') + ('matrix with no column or row labels.')))

    parser.add_argument(
        dest='names_file',
        type=str,
        metavar='names_file',
        help=textwrap.dedent(('Text file that contains the names of each \
region, in the same\n') + ('order as the correlation matrix. One region \
name on each line.')))

    parser.add_argument(
        dest='centroids_file',
        type=str,
        metavar='centroids_file',
        help=textwrap.dedent(('Text file that contains the x, y, z \
coordinates of each region,\n') + ('in the same order as the correlation \
matrix. One set of three\n') + ('coordinates, tab or space delimited, on each \
line.')))

    parser.add_argument(
        dest='output_dir',
        type=str,
        metavar='output_dir',
        help=textwrap.dedent(('Location in which to save global and nodal \
measures.')))

    parser.add_argument(
        '-c', '--cost',
        type=float,
        metavar='cost',
        help=textwrap.dedent(('Cost at which to threshold the matrix.\n') +
                             ('  Default: 10.0')),
        default=10.0)

    parser.add_argument(
        '-n', '--n_rand',
        type=int,
        metavar='n_rand',
        help=textwrap.dedent(('Number of random graphs to generate to compare \
with real network.\n') + ('  Default: 1000')),
        default=1000)


    arguments = parser.parse_args()

    return arguments, parser


def network_analysis_from_corrmat(corr_mat_file,
                                  names_file,
                                  centroids_file,
                                  output_dir,
                                  cost=10,
                                  n_rand=1000):
    '''
    This is the big function!
    It reads in the correlation matrix, thresholds it at the given cost
    (incorporating a minimum spanning tree), creates a networkx graph,
    calculates global and nodal measures (including random comparisons
    for the global measures) and writes them out to csv files.
    '''
    # Read in the data
    M, names, a, centroids = read_in_data(
                                corr_mat_file,
                                names_file,
                                centroids_file=centroids_file,
                                data_as_df=False)

    corrmat = os.path.basename(corr_mat_file).strip('.txt')

    # Initialise graph
    B = scn.BrainNetwork(
        network=M,
        parcellation=names,
        centroids=centroids)
    # Threshold graph
    G = B.threshold(cost)
    # Calculate the modules
    G.partition()
    # Calculate distance and hemispheric attributes
    G.calculate_spatial_measures()
    # Get the nodal measures
    # (note that this takes a bit of time because the participation coefficient
    # takes a while)
    G.calculate_nodal_measures()
    nodal_df = G.report_nodal_measures()
    nodal_name = 'NodalMeasures_{}_cost{:03.0f}.csv'.format(corrmat, cost)
    # FILL possibly wise to remove certain cols here (centroids)
    # Save your nodal measures
    write_out_measures(
        nodal_df, output_dir, nodal_name, first_columns=['name'])

    # Create setup for comparing real_graph against random graphs
    # name your graph G after the corrmat it was created from
    bundle = scn.GraphBundle([G], [corrmat])
    # Get the global measures
    # (note that this takes a bit of time because you're generating random
    # graphs)
    bundle.create_random_graphs(corrmat, n_rand)
    # Add the small world coefficient to global measures
    small_world = bundle.report_small_world(corrmat)
    for gname, G in bundle.items():
        G.graph['global_measures'].update(
            {"sw coeff against " + corrmat: small_world[gname]})
    global_df = bundle.report_global_measures()
    global_name = 'GlobalMeasures_{}_cost{:03.0f}.csv'.format(corrmat, cost)
    # Write out the global measures
    write_out_measures(
        global_df, output_dir, global_name, first_columns=[corrmat])

    # Get the rich club coefficients
    rc_df = bundle.report_rich_club()
    rc_name = 'rich_club_{}_cost{:03.0f}.csv'.format(corrmat, cost)
    # Write out the rich club coefficients
    write_out_measures(
        rc_df, output_dir, rc_name, first_columns=['degree', corrmat])


if __name__ == "__main__":

    # Read in the command line arguments
    arg, parser = setup_argparser()

    # Now run the main function :)
    network_analysis_from_corrmat(
                                arg.corr_mat_file,
                                arg.names_file,
                                arg.centroids_file,
                                arg.output_dir,
                                cost=arg.cost,
                                n_rand=arg.n_rand)

# =============================================================================
# Wooo! All done :)
# =============================================================================
