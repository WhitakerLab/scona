#!/usr/bin/env python

# A Global import to make code python 2 and 3 compatible
from __future__ import print_function
import numpy as np
import networkx as nx
import pandas as pd


# ==================== Anatomical Functions =================


def anatomical_node_attributes():
    '''
    default anatomical nodal attributes for scona

    Returns
    -------
    list
        nodal attributes considered "anatomical" by
        scona
    '''
    return ["name", "name_34", "name_68", "hemi", "centroids", "x", "y", "z"]


def anatomical_graph_attributes():
    '''
    default anatomical graph attributes for scona

    Returns
    -------
    list
        graph attributes considered "anatomical" by
        scona
    '''
    return ['parcellation', 'centroids']


def assign_node_names(G, parcellation):
    """
    Modify nodal attribute "name" for nodes of G, inplace.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    parcellation : list
        ``parcellation[i]`` is the name of node ``i`` in ``G``

    Returns
    -------
    :class:`networkx.Graph`
        graph with nodal attributes modified
    """
    # Assign anatomical names to the nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[i]['name'] = parcellation[i]
    #
    G.graph['parcellation'] = True
    return G


def assign_node_centroids(G, centroids):
    """
    Modify the nodal attributes "centroids", "x", "y", and "z" of nodes
    of G, inplace. "centroids" will be set according to the scheme
    ``G[i]["centroids"] = centroids[i]``
    for nodes ``i`` in ``G``. "x", "y" and "z" are assigned as the first,
    second and third coordinate of ``centroids[i]`` respectively.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    centroids : list
        ``centroids[i]`` is a tuple representing the cartesian coordinates of
        node ``i`` in ``G``

    Returns
    -------
    :class:`networkx.Graph`
        graph with nodal attributes modified
    """
    # Assign cartesian coordinates to the nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[i]['x'] = centroids[i][0]
        G.nodes[i]['y'] = centroids[i][1]
        G.nodes[i]['z'] = centroids[i][2]
        G.nodes[i]['centroids'] = centroids[i]
    #
    G.graph['centroids'] = True
    return G


def copy_anatomical_data(R, G,
                         nodal_keys=anatomical_node_attributes(),
                         graph_keys=anatomical_graph_attributes()):
    '''
    Copies nodal and graph attributes data from ``G`` to ``R`` for keys
    included in ``nodal_keys`` and ``graph_keys`` respectively.

    ``R`` and ``G`` are both graphs. If they have non equal vertex sets
    node data will only be copied from ``G`` to ``R`` for nodes in the
    intersection.

    Parameters
    ----------
    R : :class:`networkx.Graph`
    G : :class:`networkx.Graph`
        ``G`` has anatomical data to copy to ``R``
    nodal_keys : list, optional
        The list of keys to treat as anatomical nodal data.
        Default is set to ``anatomical_node_attributes()``, e.g
        ``["name", "name_34", "name_68", "hemi", "centroids", "x", "y", "z"]``
    graph_keys : list, optional
        The list of keys rto treat as anatomical graph data.
        Set to ``anatomical_graph_attributes()`` by default. E.g
        ``["centroids", "parcellation"]``

    Returns
    -------
    :class:`networkx.Graph`
        graph ``R`` with the anatomical data of graph ``G``

    See Also
    --------
    :func:`copy_anatomical_data`
    '''
    for key in [x for x in R._node.keys() if x in G._node.keys()]:
        R._node[key].update({k: v
                            for k, v in G._node[key].items()
                            if k in nodal_keys})

    R.graph.update({key: G.graph.get(key)
                   for key in graph_keys})
    return R


def anatomical_copy(G,
                    nodal_keys=anatomical_node_attributes(),
                    graph_keys=anatomical_graph_attributes()):
    '''
    Create a new graph from ``G`` preserving:
    * nodes
    * edges
    * any nodal attributes specified in nodal_keys
    * any graph attributes specified in graph_keys

    Parameters
    ----------
    G : :class:`networkx.Graph`
        ``G`` has anatomical data to copy to ``R``
    nodal_keys : list, optional
        The list of keys to treat as anatomical nodal data.
        Default is set to ``anatomical_node_attributes()``, e.g
        ``["name", "name_34", "name_68", "hemi", "centroids", "x", "y", "z"]``
    graph_keys : list, optional
        The list of keys rto treat as anatomical graph data.
        Set to ``anatomical_graph_attributes()`` by default. E.g
        ``["centroids", "parcellation"]``

    Returns
    -------
    :class:`networkx.Graph`
        A new graph with the same nodes and edges as ``G`` and identical
        anatomical data.

    See Also
    --------
    :func:`BrainNetwork.anatomical_copy`
    :func:`copy_anatomical_data`
    '''
    # Create new empty graph
    R = type(G)()
    # Copy nodes
    R.add_nodes_from(G.nodes())
    # Copy anatomical data
    copy_anatomical_data(R, G, nodal_keys=nodal_keys, graph_keys=graph_keys)
    # Preserve edges and edge weights
    R.add_edges_from(G.edges)
    nx.set_edge_attributes(R,
                           name="weight",
                           values=nx.get_edge_attributes(G, name="weight"))
    return R


def is_nodal_match(G, H, keys=None):
    '''
    Check that G and H have equal vertex sets.
    If keys is passed, also check that the nodal dictionaries of G and H (e.g
    the nodal attributes) are equal over the attributes in keys.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    H : :class:`networkx.Graph`
    keys : list, optional
        a list of attributes on which the nodal dictionaries of G and H should
        agree.

    Returns
    -------
    bool
        ``True`` if `G` and `H` have equal vertex sets, or if `keys` is
        specified: ``True`` if vertex sets are equal AND the graphs'
        nodal dictionaries agree on all attributes in `keys`.
        ``False`` otherwise
    '''
    if set(G.nodes()) != set(H.nodes()):
        return False
    elif keys is None:
        return True
    elif False in [(H._node.get(i).get(att) == G._node.get(i).get(att))
                   for att in keys
                   for i in G.nodes()]:
        return False
    else:
        return True


def is_anatomical_match(G,
                        H,
                        nodal_keys=anatomical_node_attributes(),
                        graph_keys=anatomical_graph_attributes()):
    '''
    Check that G and H have identical anatomical data (including vertex sets).

    Parameters
    ----------
    G : :class:`networkx.Graph`
    H : :class:`networkx.Graph`
    nodal_keys : list, optional
        The list of keys to treat as anatomical nodal data.
        Default is set to ``anatomical_node_attributes()``, e.g
        ``["name", "name_34", "name_68", "hemi", "centroids", "x", "y", "z"]``
    graph_keys : list, optional
        The list of keys to treat as anatomical graph data.
        Set to ``anatomical_graph_attributes()`` by default. E.g
        ``["centroids", "parcellation"]``

    Returns
    -------
    bool
        ``True`` if `G` and `H` have the same anatomical data;
        ``False`` otherwise
    '''
    # check nodes match
    if not is_nodal_match(G, H, keys=nodal_keys):
        return False
    # check graph attributes match
    elif ({k: v for k, v in G.graph.items() if k in graph_keys}
            != {k: v for k, v in H.graph.items() if k in graph_keys}):
        return False
    else:
        return True


# ================= Graph construction =====================


def weighted_graph_from_matrix(M, create_using=None):
    '''
    Create a weighted graph from a correlation matrix.

    Parameters
    ----------
    M : :class:`numpy.array`
        a correlation matrix
    create_using : :class:`networkx.Graph`, optional
        Use specified graph for result. The default is Graph()

    Returns
    -------
    :class:`networkx.Graph`
        A weighted graph with edge weights equivalent to matrix entries
    '''
    # Make a copy of the matrix
    thr_M = np.copy(M)

    # Set all diagonal values to 0
    thr_M[np.diag_indices_from(thr_M)] = 0

    # Read this full matrix into a graph G
    G = nx.from_numpy_matrix(thr_M, create_using=create_using)

    return G


def weighted_graph_from_df(df, create_using=None):
    '''
    Create a weighted graph from a correlation matrix.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        a correlation matrix
    create_using : :class:`networkx.Graph`
        Use specified graph for result. The default is Graph()

    Returns
    -------
    :class:`networkx.Graph`
        A weighted graph with edge weights equivalent to DataFrame entries
    '''
    return weighted_graph_from_matrix(df.values, create_using=create_using)


def scale_weights(G, scalar=-1, name='weight'):
    '''
    Multiply edge weights of `G` by `scalar`.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    scalar : float, optional
        scalar value to multiply edge weights by. Default is -1
    name : str, optional
        string that indexes edge weights. Default is "weight"

    Returns
    -------
    :class:`networkx.Graph`

    '''
    edges = nx.get_edge_attributes(G, name=name)
    new_edges = {key: value*scalar for key, value in edges.items()}
    nx.set_edge_attributes(G, name=name, values=new_edges)
    return G


def threshold_graph(G, cost, mst=True):
    '''
    Create a binary graph by thresholding weighted graph G.

    First creates the minimum spanning tree for the graph, and then adds
    in edges according to their connection strength up to cost.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        A complete weighted graph
    cost : float
        A number between 0 and 100. The resulting graph will have the
        ``cost*n/100`` highest weighted edges from G, where ``n`` is the number
        of edges in G.
    mst : bool, optional
        If ``False``, skip creation of minimum spanning tree. This may cause
        output graph to be disconnected

    Returns
    -------
    :class:`networkx.Graph`
        A binary graph

    Raises
    ------
    Exception
        If it is impossible to create a minimum_spanning_tree at the given cost

    See Also
    --------
    :func:`scona.BrainNetwork.threshold`
    '''
    # Weights scaled by -1 as minimum_spanning_tree minimises weight
    H = scale_weights(anatomical_copy(G))
    # Make a list of all the sorted edges in the full matrix
    H_edges_sorted = sorted(H.edges(data=True),
                            key=lambda edge_info: edge_info[2]['weight'])
    # Create an empty graph with the same nodes as H
    germ = anatomical_copy(G)
    germ.remove_edges_from(G.edges)

    if mst:
        # Calculate minimum spanning trees
        germ.add_edges_from(nx.minimum_spanning_edges(H))

    # Make a list of the germ graph's edges
    germ_edges = germ.edges(data=True)

    # Create a list of sorted edges that are *not* in the germ
    # (because you don't want to add them in twice!)
    H_edges_sorted_not_germ = [edge for edge in H_edges_sorted
                               if edge not in germ_edges]
    # Calculate how many edges need to be added to reach the specified cost
    # and round to the nearest integer.
    n_edges = (cost/100.0) * len(H)*(len(H)-1)*0.5
    n_edges = np.int(np.around(n_edges))
    n_edges = n_edges - len(germ_edges)

    # If your cost is so small that your minimum spanning tree already covers
    # it then you can't do any better than the MST and you'll just have to
    # return it with an accompanying error message
    # A tree has n-1 edges and a complete graph has n(n − 1)/2 edges, so we
    # need cost/100 > 2/n, where n is the number of vertices
    if n_edges < 0:
        raise Exception('Unable to calculate matrix at this cost -\
                         minimum spanning tree is too large')
        print('cost must be >= {}'.format(2/len(H)))
    # Otherwise, add in the appropriate number of edges (n_edges)
    # from your sorted list (H_edges_sorted_not_germ)
    else:
        germ.add_edges_from(H_edges_sorted_not_germ[:n_edges])
    # binarise edge weights
    nx.set_edge_attributes(germ, name='weight', values=1)
    # And return the updated germ as your graph
    return germ


def graph_at_cost(M, cost, mst=True):
    '''
    Create a binary graph by thresholding weighted matrix M.

    First creates the minimum spanning tree for the graph, and then adds
    in edges according to their connection strength up to cost.

    Parameters
    ----------
    M : :class:`numpy.array` or :class:`pandas.DataFrame`
        A correlation matrix.
    cost : float
        A number between 0 and 100. The resulting graph will have the
        ``cost*n/100`` highest weighted edges available, where ``n`` is the
        number of edges in G
    mst : bool, optional
        If ``False``, skip creation of minimum spanning tree. This may cause
        output graph to be disconnected

    Returns
    -------
    :class:`networkx.Graph`
        A binary graph

    Raises
    ------
    Exception
        if M is not a :class:`numpy.array` or :class:`pandas.DataFrame`
    '''
    # If dataframe, convert to array
    if isinstance(M, pd.DataFrame):
        array = M.values
    elif isinstance(M, np.ndarray):
        array = M
    else:
        raise TypeError(
              "M should be a numpy array or pandas dataframe")

    # Read this array into a graph G
    G = weighted_graph_from_matrix(array)
    return threshold_graph(G, cost, mst=mst)


# ===================== Random Graphs ======================


def random_graph(G, Q=10, seed=None):
    '''
    Return a connected random graph that preserves degree distribution
    by swapping pairs of edges, using :func:`networkx.double_edge_swap`.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    Q : int, optional
        constant that specifies how many swaps to conduct for each edge in G
    seed : int, random_state or None (default)
        Indicator of random state to pass to networkx

    Returns
    -------
    :class:`networkx.Graph`

    Raises
    ------
    Exception
        if it is not possible in 15 attempts to create a connected random
        graph.
    '''
    R = anatomical_copy(G)

    # Calculate the number of edges and set a constant
    # as suggested in the nx documentation
    E = R.number_of_edges()

    # Start the counter for randomisation attempts and set connected to False
    attempt = 0
    connected = False

    # Keep making random graphs until they are connected
    while not connected and attempt < 15:
        # Now swap some edges in order to preserve the degree distribution
        nx.double_edge_swap(R, Q*E, max_tries=Q*E*10, seed=seed)

        # Check that this graph is connected! If not, start again
        connected = nx.is_connected(R)
        if not connected:
            attempt += 1

    if attempt == 15:
        raise Exception("** Failed to randomise graph in first 15 tries -\
                             Attempt aborted. Network is likely too sparse **")
    return R


def get_random_graphs(G, n=10, Q=10, seed=None):
    '''
    Create n random graphs through edgeswapping.

    Parameters
    ----------
    G : :class:`networkx.Graph`
    n : int, optional
    Q : int, optional
        constant to specify how many swaps to conduct for each edge in G
    seed : int, random_state or None (default)
        Indicator of random state to pass to networkx

    Returns
    -------
    list of :class:`networkx.Graph`
        A list of length n of randomisations of G created using
        :func:`scona.make_graphs.random_graph`
    '''
    graph_list = []

    print('        Creating {} random graphs - may take a little while'
          .format(n))

    for i in range(n):
        if len(graph_list) <= i:
            graph_list += [random_graph(G, Q=Q, seed=seed)]

    return graph_list
