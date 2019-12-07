#!/usr/bin/env python

from scona.wrappers.corrmat_from_regionalmeasures import corrmat_from_regionalmeasures
from scona.wrappers.network_analysis_from_corrmat import network_analysis_from_corrmat
from scona.wrappers.parsers import scona_parser

def main():
    arg = scona_parser.parse_args()
    
    M = corrmat_from_regionalmeasures(
        arg.regional_measures_file,
        arg.names_file,
        covars_file=arg.covars_file,
        output_name=arg.output_name,
        method=arg.method)

    network_analysis_from_corrmat(
        arg.names_file,
        arg.centroids_file,
        arg.output_dir,
        corrmat=M,
        corrmat_file=arg.output_name,
        cost=arg.cost,
        n_rand=arg.n_rand,
        edge_swap_seed=arg.seed)

if __name__ == "__main__":
    main()
