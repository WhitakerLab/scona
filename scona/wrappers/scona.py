import scona
import numpy as np
import os
import scona.make_corr_matrices as mcm
from scona.scripts.useful_functions import read_in_data, \
    write_out_measures, list_from_file


def corrmat_from_regionalmeasures(args):
    '''
    Read in regional measures, names and covariates files to compute
    and return a structural covariance matrix, or write it to 
    output_name.
    The structural covariance matrix is the pairwise correlation of 
    the columns given by names_file over the rows of regional_measures,
    after correcting for covariance with the columns in covars_file.

    Returns
    -------
    :class:`pandas.DataFrame
        A correlation matrix
    '''
    # Read in the data
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)
    # create correlation matrix
    M = mcm.corrmat_from_regionalmeasures(
        df, names, covars=covars_list, method=args.method)
    if args.output_name is not None:
        mfile = os.path.join(args.output_dir, args.output_name)
        print("saving correlation matrix to {}".format(mfile))
        # Save the matrix
        mcm.save_mat(M, mfile)
    return M


def network_analysis_from_corrmat(args, corrmat=None):
    '''
    Run the standard scona network analysis on corrmat, interpreted as
    a weighted graph. 

    This analysis thresholds corrmat at the desired cost to create a 
    binary network and calculates network measures, described further down.

    For the purposes of comparison this analysis also generates a number
    of random graphs via edge swapping (see :func:`networkx.double_edge_swap`)
    and reports global measures and rich club measures

    Writes
    ------
    * A dataframe reporting the nodal measures for the
    nodes of corrmat
    * A dataframe reporting the global measures of corrmat and
    all random graphs
    * A dataframe reporting the rich club, at every
    degree, of corrmat and all random graphs

    Network Measures
    ================
    Nodal Measures
    --------------
    * "degree" : int
        the number of incident edges
    * "betweenness" : float
        the betweenness centrality of each node, see :func:`networkx.betweenness_centrality`
    * "closeness" : float
        the closeness centrality of each node, see :func:`networkx.closeness_centrality`
    * "clustering" : float
        the clustering coefficient of each node, see :func:`networks.clustering`
    * "module" : int
        each node is assigned an integer-named module by the louvain
        method of community detection, see https://python-louvain.readthedocs.io
    * "participation_coefficient" : float
        the participation coefficient of nodes of G with partition
        defined by "module".
    * "shortest_path_length" : float
      the average shortest path length for each node in G.
      "length" in this case means the number of edges, and does
      not consider euclidean distance.
    * "total_dist" : float
        the total length of the incident edges
    * "average_dist" : float
        the average length of the incident edges
    * "hemisphere" : str
        L or R, as determined by the sign of the x coordinate
        and assuming MNI space. The x coordinates are negative
        in the left hemisphere and positive in the right.
    * "interhem" : int
        the number of adjacent interhemispheric edges
    * "interhem_proportion" : float
        the proportion of adjacent edges that are interhemispheric

    Global Measures
    ---------------
    * "average_clustering" : float
      see :func:`networkx.average_clustering`
    * "average_shortest_path_length" : float
      see :func:`networkx.average_shortest_path_length`
    * "assortativity" : float
      see :func:`networkx.degree_assortativity_coefficient`
    * "modularity" : float
      modularity of network under partition defined by "module"
    * "efficiency" : float
      see :func:`networkx.global_efficiency`
    * "small..."
      small world coefficient of networks relative to the
      network derived from M.
      See :func:`graph_measures.small_world_coefficient`

    Rich Club
    ---------
    For some integer k, the rich club coefficient of degree k
    measures the completeness of the subgraph on nodes with
    degree >= k.
    See :func:`networkx.rich_club_coefficient`
    '''
    # Read in the data
    if corrmat is None:
        M, names, a, centroids = read_in_data(
            args.corrmat_file,
            args.names_file,
            centroids_file=args.centroids_file,
            data_as_df=False)
        
    else:
        M = corrmat
        names = list_from_file(args.names_file)
        if args.centroids_file is not None:
            centroids = list(np.loadtxt(args.centroids_file))
        else:
            centroids = None

    # if possible, name network after corrmat_file
    network_name = ""
    if hasattr(args, 'corrmat_file'):
        if args.corrmat_file is not None:
            network_name = args.corrmat_file

    # run standard analysis
    bundle, nodal_df, global_df, rc_df = scona.network_analysis_from_matrix(
        M, args.cost, args.n_rand, name=network_name, seed=args.seed, parcellation=names, centroids=centroids)

    # write out each of the outputs
    nodal_name = 'NodalMeasures_{}_cost{:03.0f}.csv'.format(
        network_name, args.cost)
    write_out_measures(
        nodal_df, args.output_dir, nodal_name, first_columns=['name'])

    global_name = 'GlobalMeasures_{}_cost{:03.0f}.csv'.format(
        network_name, args.cost)
    write_out_measures(
        global_df, args.output_dir, global_name, first_columns=[network_name])
    
    rc_name = 'rich_club_{}_cost{:03.0f}.csv'.format(
        network_name, args.cost)
    write_out_measures(
        rc_df, args.output_dir, rc_name, first_columns=['degree', network_name])

def standard_analysis(args):

    '''
    Create a structural covariance analysis network from 
    regional_measures_file and run the standard scona network analysis
    on it.

    To create the structural covariance network from regional_measures_file,
    scona calculates the pairwise correlations of the columns listed in
    names_file over the rows of regional_measures_file, correcting for
    covariance with the columns listed in covars_file.
 
    scona thresholds the resulting matrix at the desired cost to create a 
    binary network and calculates network measures, described further down.

    For the purposes of comparison this analysis also generates a number
    of random graphs via edge swapping (see :func:`networkx.double_edge_swap`)
    and reports global measures and rich club measures on these.

    Writes
    ------
    * A dataframe reporting the nodal measures for the
    nodes of the structural covariance network.
    * A dataframe reporting the global measures of the
    structural covariance network and random graphs,
    * A dataframe reporting the rich club, at every
    degree, of each network.
    
    Network Measures
    ================
    Nodal Measures
    --------------
    * "degree" : int
        the number of incident edges
    * "betweenness" : float
        the betweenness centrality of each node, see :func:`networkx.betweenness_centrality`
    * "closeness" : float
        the closeness centrality of each node, see :func:`networkx.closeness_centrality`
    * "clustering" : float
        the clustering coefficient of each node, see :func:`networks.clustering`
    * "module" : int
        each node is assigned an integer-named module by the louvain
        method of community detection, see https://python-louvain.readthedocs.io
    * "participation_coefficient" : float
        the participation coefficient of nodes of G with partition
        defined by "module".
    * "shortest_path_length" : float
      the average shortest path length for each node in G.
      "length" in this case means the number of edges, and does
      not consider euclidean distance.
    * "total_dist" : float
        the total length of the incident edges
    * "average_dist" : float
        the average length of the incident edges
    * "hemisphere" : str
        L or R, as determined by the sign of the x coordinate
        and assuming MNI space. The x coordinates are negative
        in the left hemisphere and positive in the right.
    * "interhem" : int
        the number of adjacent interhemispheric edges
    * "interhem_proportion" : float
        the proportion of adjacent edges that are interhemispheric

    Global Measures
    ---------------
    * "average_clustering" : float
      see :func:`networkx.average_clustering`
    * "average_shortest_path_length" : float
      see :func:`networkx.average_shortest_path_length`
    * "assortativity" : float
      see :func:`networkx.degree_assortativity_coefficient`
    * "modularity" : float
      modularity of network under partition defined by "module"
    * "efficiency" : float
      see :func:`networkx.global_efficiency`
    * "small..."
      small world coefficient of networks relative to the
      network derived from M.
      See :func:`graph_measures.small_world_coefficient`

    Rich Club
    ---------
    For some integer k, the rich club coefficient of degree k
    measures the completeness of the subgraph on nodes with
    degree >= k.
    See :func:`networkx.rich_club_coefficient`
    '''

    M = corrmat_from_regionalmeasures(args)
    
    network_analysis_from_corrmat(args, corrmat=M)

def groupwise_analysis(args):
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)
    
    scona.analyses.groupwise_analysis(
        df,
        names,
        args.cost,
        args.group_var,
        covars=covars_list,
        method=args.method,
        seed=args.seed)
        
    

def movingwindow_analysis(args):
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)

    scona.analyses.moving_window_analysis(
        df,
        names,
        args.cost,
        args.window_var,
        args.window_size,
        covars=covars_list,
        method=args.method,
        seed=args.seed)

    

