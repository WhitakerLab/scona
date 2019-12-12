from scona.classes import BrainNetwork, GraphBundle
from scona.make_corr_matrices import corrmat_from_regionalmeasures

def network_analysis_from_matrix(
        M,
        cost,
        n_rand,
        name="critical_network",
        seed=None,
        parcellation=None,
        centroids=None):
    '''
    Run the standard scona network analysis on an array M, interpreted as
    a weighted graph. 

    This network analysis thresholds M at the desired cost to create a 
    binary network and calculates the network measures listed lower down.

    For the purposes of comparison this analysis also generates a number
    of random graphs via edge swapping (see :func:`networkx.double_edge_swap`)
    and reports global measures and rich club measures

    Parameters
    ----------
    M : :class:`numpy.array` or :class:`pandas.DataFrame`
        M will be treated as a weighted graph, where the
        M[i][j] represents the weight of the edge connecting
        node i to node j
    cost : float
        We construct a binary graph from M by restricting
        to the ``cost*n/100`` highest weighted edges, where
        ``n`` is the number of edges in M.
    n_rand : int
        The analysis requires the generation of random graphs
        to create a distribution to test M against. Use n_rand
        to sepcify how many graphs you wish to create.
    name : str, optional
        This is an optional label for the graph M, to distinguish
        it from the random graphs.
    seed : int, optional
    parcellation : list, optional
        Anatomical names to assign to the nodes of your graph
    centroids : list, optional
        Anatomical locations to assign to the nodes of your
        graph.

    Returns
    -------
    :class:`scona.GraphBundle`, :class:`pandas.DataFrame`, :class:`pandas.DataFrame`, :class:`pandas.DataFrame`
        * A dictionary of networks created during this analysis,
        with M indexed by `name` 
        * A dataframe reporting the nodal measures for the
        nodes of M
        * A dataframe reporting the global measures of M and
        all random graphs
        * A dataframe reporting the rich club at every
        degree of M and all random graphs

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
    
    # Initialise graph
    weighted_network = BrainNetwork(
        network=M,
        parcellation=parcellation,
        centroids=centroids)

    # Threshold graph
    binary_network = weighted_network.threshold(cost)

    # Calculate the modules, distance and hemispheric attributes
    # and the nodal measures
    binary_network.partition()
    binary_network.calculate_spatial_measures()
    binary_network.calculate_nodal_measures()
    

    # Create setup for comparing binary_network against random graphs
    # (note that this takes a bit of time because you're generating random
    # graphs)
    bundle = GraphBundle(graph_dict={name: binary_network})
    bundle.create_random_graphs(name, n_rand, seed=seed)
    
    # Add the small world coefficient to global measures
    small_world = bundle.report_small_world(network_name)

    for gname, network in bundle.items():
        network.graph['global_measures'].update(
            {"sw coeff against " + network_name: small_world[gname]})
 
    return bundle, binary_network.report_nodal_measures(), bundle.report_global_measures(), bundle.report_rich_club


def standard_analysis(
        df,
        names,
        cost,
        covars=None,
        centroids=None,
        method='pearson',
        name="critical_network",
        seed=None):
    '''
    Create a structural covariance analysis network from `df` and run
    the standard scona network analysis on it.

    To create the structural covariance network from `df`, scona
    calculates the pairwise correlations of the columns in `names`
    over the rows of `df`, correcting for covariance with the columns
    of `covars`. 
    scona thresholds the resulting matrix at the desired cost to create
    a binary network and calculates and returns a selection of network 
    measures; see :func:`network_analysis_from_matrix`.

    For the purposes of comparison this analysis also generates a number
    of random graphs via edge swapping (see :func:`networkx.double_edge_swap`)
    and reports global measures and rich club measures

    Parameters
    ----------
    regional_measures : :class:`pandas.DataFrame`
        a pandas DataFrame with individual brain scans as rows, and 
        columns including brain regions and covariates. The columns in
        names and covars_list should be numeric.
    names : list
        a list of the brain regions whose correlation you want to assess
    covars: list, optional
        covars is a list of covariates (as DataFrame column headings)
        to correct for before correlating brain regions.
    method : string, optional
        the method of correlation passed to :func:`pandas.DataFramecorr`
    cost : float
        We construct a binary graph from the correlation matrix by 
        restricting to the ``cost*n/100`` highest weighted edges, where
        ``n`` is the number of edges.
    n_rand : int
        The analysis requires the generation of random graphs
        to create a distribution to test the  against. Use n_rand
        to specify how many graphs you wish to create.
    name : str, optional
        This is an optional label for the initial structural covariance
        network, to distinguish it from the random graphs.
    seed : int, optional
    centroids : list, optional
        Anatomical locations to assign to the nodes of your graph.

    Returns
    -------
    :class:`scona.GraphBundle`, :class:`pandas.DataFrame`, :class:`pandas.DataFrame`, :class:`pandas.DataFrame`
        * A dictionary of networks created during this analysis,
        with the initial structural covariance network indexed by
        `name` 
        * A dataframe reporting the nodal measures for the nodes of
        the structural covariance network
        * A dataframe reporting the global measures of all networks
        * A dataframe reporting the rich club, at every degree, of
        each network.
    '''

    M = corrmat_from_regionalmeasures(
        df,
        names,
        covars=covars,
        method=method)

    return network_analysis_from_matrix(
        M,
        cost,
        n_rand,
        name=name,
        seed=seed,
        parcellation=names,
        centroids=centroids)

def groupwise_analysis(
        df,
        names,
        group_var,
        cost,
        covars=None,
        method='pearson',
        seed=None):
    # run first for true group assignments
    groupwise_bundle = GraphBundle.from_regional_measures(
        df,
        names,
        groupby=group_var,
        covars=covars,
        method=method)
    groupwise_bundle.threshold(cost)

    for i in range(args.n_shuffle):
        gb = GraphBundle.from_regional_measures(
            df,
            names,
            groupby=group_var,
            covars=covars,
            method=method,
            shuffle=True,
            seed=seed)
        gb.threshold(cost)
        gb.report_global_measures()

def moving_window_analysis(
        df,
        names,
        cost,
        window_var,
        window_size,
        covars=None,
        method='pearson',
        seed=None):

    moving_window_bundle = GraphBundle.from_regional_measures(
        df, names, covars=covars, method=method,
        windowby=window_var, window_size=window_size)
    moving_window_bundle.threshold(cost)
    

    shuffle_list = []
    for i in range(args.n_shuffle):
            gb = GraphBundle.from_regional_measures(
                df, names, covars=covars, method=method,
                windowby=window_var, window_size=window_size,
                shuffle=True, seed=seed)
            gb.threshold(cost)
