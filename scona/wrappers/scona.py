#!/usr/bin/env python

from scona.wrappers.corrmat_from_regionalmeasures import corrmat_from_regionalmeasures
from scona.wrappers.network_analysis_from_corrmat import network_analysis_from_corrmat
from scona.wrappers.parsers import scona_parser

if __name__ == "__main__":

    arg = parsers.scona_parser.parse_args()
    
    corrmat_from_regionalmeasures(
        arg.regional_measures_file,
        arg.names_file,
        arg.output_name,
        covars_file=arg.covars_file,
        method=arg.method)

    network_analysis_from_corrmat(
        arg.corr_mat_file,
        arg.names_file,
        arg.centroids_file,
        arg.output_dir,
        cost=arg.cost,
        n_rand=arg.n_rand,
        edge_swap_seed=arg.seed)
