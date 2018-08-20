import numpy as np
import networkx as nx
# ==================== Nodal methods =======================


def calc_nodal_partition(G):
    '''
    Calculate a nodal partition of G using the louvain algorithm as
    iBrainNetworkommunity.best_partition`

    Note that this is a time intensive process and it is also
    non-deterministic, so for consistency and speed it's best
    to hold on to your partition.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        A binary graph

    Returns
    -------
    (dict, dict)
        Two dictionaries represent the resulting nodal partition of G. The
        first maps nodes to modules and the second maps modules to nodes.
    '''
    import community
    # Make sure the edges are binarized
    for u, v, d in G.edges(data=True):
        if d.get('weight', 1) != 1:
            raise ValueError("G should be a binary graph")
    # Now calculate the best partition
    nodal_partition = community.best_partition(G)

    # Reverse the dictionary to record a list of nodes per module, rather than
    # module per node
    module_partition = {}
    for n, m in nodal_partition.items():
        try:
            module_partition[m].append(n)
        except KeyError:
            module_partition[m] = [n]

    return nodal_partition, module_partition


def participation_coefficient(G, module_partition):
    '''
    Computes the participation coefficient of nodes of G with partition
    defined by module_partition.
    (Guimera et al. 2005).

    Parameters
    ----------
    G : :class:`networkx.Graph`
    module_partition : dict
        a dictionary mapping each community name to a list of nodes in G

    Returns
    -------
    dict
        a dictionary mapping the nodes of G to their participation coefficient
        under the participation specified by module_partition.
    '''
    # Initialise dictionary for the participation coefficients
    pc_dict = {}

    # Print a little note to the screen because it can take a long
    # time to run this code
    print('        Calculating participation coefficient -\
           may take a little while')

    # Loop over modules to calculate participation coefficient for each node
    for m in module_partition.keys():
        # Create module subgraph
        M = G.subgraph(set(module_partition[m]))
        for v in M.nodes:
            # Calculate the degree for v
            degree = nx.degree(G=G, nbunch=v)

            # Calculate the number of intramodule edges
            wm_edges = float(nx.degree(G=M, nbunch=v))

            # The participation coeficient is 1 - the square of
            # the ratio of the within module degree and the total degree
            pc = 1 - ((float(wm_edges) / float(degree))**2)

            pc_dict[v] = pc

    return pc_dict


def z_score(G, module_partition):
    '''
    Calculate the z-score of the nodes of G under partition module_partition.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    module_partition : dict
        a dictionary mapping each community name to a lists of nodes in G

    Returns
    -------
    dict
        a dictionary mapping the nodes of G to their z-score under
        module_partition.
    '''
    # Initialise dictionary for the z-scores
    z_score = {}

    # Loop over modules to calculate z-score for each node
    for m in module_partition.keys():
        # Create module subgraph
        M = G.subgraph(set(module_partition[m]))
        # Calculate relevant module statistics
        M_degrees = list(dict(M.degree()).values())
        M_degree = np.mean(M_degrees)
        M_std = np.std(M_degrees)
        for v in M.nodes:
            # Calculate the number of intramodule edges
            wm_edges = float(nx.degree(G=M, nbunch=v))
            # Calculate z score as the intramodule degree of v
            # minus the mean intramodule degree, all divided by
            # the standard deviation of intramodule degree
            if M_std != 0:
                zs = (wm_edges - M_degree)/M_std
            else:
                # If M_std is 0, then all M_degrees must be equal.
                # It follows that the intramodule degree of v must equal
                # the mean intramodule degree.
                # It is therefore valid to assign a 0 value to the z-score
                zs = 0
            z_score[v] = zs

    return z_score


def shortest_path(G):
    '''
    Calculate average shortest path length for each node in G.
    "length" in this case means the number of edges, and does not consider
    euclidean distance.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        a connected graph

    Returns
    -------
    dict
        a dictionary mapping a node v to the average length of the shortest
        from v to other nodes in G.
    '''
    shortestpl_dict = {}
    for node in G.nodes():
        shortestpl_dict[node] = np.average(
            list(nx.shortest_path_length(G, source=node).values()))
    return shortestpl_dict


# =============== anatomical measures ========================


def assign_nodal_distance(G):
    '''
    Assigns nodal and edge attributes of G. Modifies G in place.

    Edge attributes

    "euclidean" : float
        the euclidean length, derived from the "centroids" values of nodes

    Node attributes

    "total_dist" : float
        the total length of the incident edges
    "average_dist" : float
        the average length of the incident edges

    Parameters
    ----------
    G : :class:`networkx.Graph`
        a graph with nodal attribute 'centroids' defined for each
        node. The value of 'centroids' should be the cartesian coordinates of
        each node.

    Returns
    -------
    :class:`networkx.Graph`
        G
    '''
    from scipy.spatial import distance
    for i, node in enumerate(G.nodes()):
        # Loop through the edges connecting to this node
        # Note that "node1" is equal to "node"
        for node1, node2 in G.edges(nbunch=[node]):

            # Calculate the euclidean distance for this edge
            cent1 = G.node[node1]['centroids']
            cent2 = G.node[node2]['centroids']

            dist = distance.euclidean(cent1, cent2)

            # And assign this value to the edge
            G.adj[node1][node2]['euclidean'] = dist

        # Create two nodal attributes (average distance and
        # total distance) by summarizing the euclidean distance
        # for all edges which connect to the node
        euc_list = [G.adj[m][n]['euclidean'] for m, n in G.edges(nbunch=node)]

        G.node[node]['average_dist'] = np.mean(euc_list)
        G.node[node]['total_dist'] = np.sum(euc_list)
    return G


def assign_interhem(G):
    '''
    Assigns nodal and edge attributes of G. Modifies G in place.
    An edge is considered interhemispheric if the x coordinates of its
    nodes have different signs.

    Edge attributes:

    "interhem" : int
        1 if the edge is interhemispheric, 0 otherwise

    Node attributes

    "hemisphere" : int
        1 or -1, represents the sign of the x coordinate.
    "interhem" : int
        the number of adjacent interhemispheric edges
    "interhem_proportion" : float
        the proportion of adjacent edges that are interhemispheric

    Parameters
    ----------
    G : :class:`networkx.Graph`
        a graph with nodal attribute 'centroids' or 'x' defined for each
        node. The value of 'centroids' should be the cartesian coordinates
        of each node.

    Returns
    -------
    :class:`networkx.Graph`
        G
    '''
    for i, node in enumerate(G.nodes()):
        for node1, node2 in G.edges(nbunch=[node]):
            # Determine whether this edge is interhemispheric
            # by multiplying the x values.
            try:
                x1 = G.node[node1]['x']
                x2 = G.node[node2]['x']
            except KeyError:
                x1 = G.node[node1]['centroids'][0]
                x2 = G.node[node2]['centroids'][0]

            # Determine whether this edge is interhemispheric
            # by multiplying the x values.
            if x1*x2 > 0:
                G.adj[node1][node2]['interhem'] = 0
            else:
                G.adj[node1][node2]['interhem'] = 1

        # Create an interhem nodal attribute by getting the average
        # of the interhem values for all edges which connect to the node
        interhem_list = [G.adj[m][n]['interhem']
                         for m, n in G.edges(nbunch=node)]
        G.node[node]['hemisphere'] = np.sign(x1)
        G.node[node]['interhem'] = sum(interhem_list)
        G.node[node]['interhem_proportion'] = np.mean(interhem_list)
    return G


# ============== Nodal Measure ===============

def calculate_nodal_measures(
        G,
        measure_list=None,
        additional_measures=None,
        force=True):
    '''
    Calculate and store nodal measures as nodal attributes.

    By default calculates:

    * "nodal_partition" : int or str
    * "degree" : int
    * "closeness" : float
    * "betweenness" : float
    * "shortest_path_length" : int
    * "clustering" : float
    * "participation_coefficient" : float

    Use `measure_list` to specify which of the default nodal attributes to
    calculate.
    Use `additional_measures` to describe and calculate new measure definitions.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    measure_list : list of str, optional
        pass a subset of of the keys defined above to specify which of the
        default measures to calculate
    additional_measures : dict, optional
        map from names of nodal attributes to functions
        defining how they should be calculated. Such a function should take a
        graph as an argument and return a dictionary mapping nodes to attribute
        values.
    force : bool, optional
        pass True to recalculate any measures that already
        exist in the nodal attributes.

    See Also
    --------
    :func:`BrainNetwork.calculate_nodal_measures`
    :func:`calc_nodal_partition`

    Example
    -------

    '''
    # ==== DESCRIBE MEASURES =====================
    nodal_measure_dict = {
        "degree": (lambda x: dict(nx.degree(x))),
        "closeness": nx.closeness_centrality,
        "betweenness": nx.betweenness_centrality,
        "shortest_path_length": nx.betweenness_centrality,
        "clustering": nx.clustering,
        "participation_coefficient": (lambda x: participation_coefficient(
                                        x,
                                        x.graph['partition']))
        }

    if measure_list is not None:
        nodal_measure_dict = {key: value
                              for key, value in nodal_measure_dict.items()
                              if key in measure_list}
    if additional_measures is not None:
        nodal_measure_dict.update(additional_measures)

    # ==== CALCULATE MEASURES ====================

    def calc_nodal_measure(G, measure, method, force=False):
        if (not nx.get_node_attributes(G, name=measure)) or force:
            nx.set_node_attributes(G,
                                   name=measure,
                                   values=method(G))

    for measure, method in nodal_measure_dict.items():
        calc_nodal_measure(G, measure, method, force=force)


# ============= Global measures =============


def calc_modularity(G, nodal_partition):
    '''
    Calculate the modularity of G under partition nodal_partition.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    nodal_partition : dict
        a dictionary nodes to communities

    Returns
    -------
    float
        the modularity of G
    '''
    import community
    return community.modularity(nodal_partition, G)


def rich_club(G):
    '''
    Calculate the rich club coefficient of G for each degree between 0 and
    ``max([degree(v) for v in G.nodes])``.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        a binary graph

    Returns
    -------
    dict
        a dictionary mapping integer ``x`` to the rich club coefficient of G
        for degree ``x``

    See Also
    --------
    :func:`BrainNetwork.rich_club`
    '''
    return nx.rich_club_coefficient(G, normalized=False)


# ================= Small World methods ============================


def small_world_sigma(tupleG, tupleR):
    '''
    Compute small world sigma from tuples

    Parameters
    ----------
    tupleG, tupleR : tuple of floats

    Returns
    -------
    float
    '''
    Cg, Lg = tupleG
    Cr, Lr = tupleR
    return ((Cg/Cr)/(Lg/Lr))


def small_coefficient(G, R):
    '''
    Calculate the small coefficient of G relative to R.

    Parameters
    ----------
    G, R : :class:`networkx.Graph`
        A binary graph

    Returns
    -------
    float
        The small coefficient of G relative to R
    '''
    return small_world_sigma((nx.average_clustering(G),
                              nx.average_shortest_path_length(G)),
                             (nx.average_clustering(R),
                              nx.average_shortest_path_length(R)))


# ============ Calculate Global Measures En Masse ================


def calculate_global_measures(G,
                              partition=None,
                              existing_global_measures=None):
    '''
    Calculate global measures `average_clustering`,
    `average_shortest_path_length`, `assortativity`, `modularity`, and
    `efficiency` of G.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        A binary graph
    partition : dict, optional
        A nodal partition of G. A dictionary mapping nodes of G to modules.
        Pass a partition in order to calculate the modularity of G.
    existing_global_measures : dict, optional
        An existing dictionary of global measures of G can be passed.
        :func:`calculate_global_measures` will not recalculate any measures
        already indexed in G

    Returns
    -------
    dict
        a dictionary of global network measures of G

    See Also
    --------
    :func:`BrainNetworksInPython.BrainNetwork.calculate_global_measures`
    '''
    # ==== MEASURES ====================
    if existing_global_measures is not None:
        global_measures = existing_global_measures
    else:
        global_measures = {}

    # ---- Clustering coefficient ------
    if 'average_clustering' not in global_measures:
        global_measures['average_clustering'] = (
            nx.average_clustering(G))

    # ---- Shortest path length --------
    if 'average_shortest_path_length' not in global_measures:
        global_measures['average_shortest_path_length'] = (
            nx.average_shortest_path_length(G))

    # ---- Assortativity ---------------
    if 'assortativity' not in global_measures:
        global_measures['assortativity'] = (
            np.mean(nx.degree_assortativity_coefficient(G)))

    # ---- Modularity ------------------
    if partition is not None and 'modularity' not in global_measures:
        global_measures['modularity'] = (
            calc_modularity(G, partition))

    #  ---- Efficiency ------------------
    if 'efficiency' not in global_measures:
        global_measures['efficiency'] = (
            nx.global_efficiency(G))

    return global_measures
