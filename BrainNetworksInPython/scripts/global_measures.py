import numpy as np
import networkx as nx
import make_graphs as mkg
import graph_measures as gm


def get_random_graphs(G, n=10):
    '''
    Creates n random graphs through edgeswapping.

    Returns a tuple (a,b) where
    - a is a list of edgeswap randomisations of G
    - b is a list of the corresponding nodal partitions

    G should be a graph and n an integer.
    '''
    R_list = []
    R_nodal_partition_list = []

    print('        Creating {} random graphs - may take a little while'
          .format(n))

    for i in range(n):
        if len(R_list) <= i:
            R_list += [mkg.random_graph(G)]

            R_nodal_partition_list += [gm.calc_nodal_partition(R_list[i])]

    return (R_list, R_nodal_partition_list)


def rich_club(G, R_list=None, n=10):
    '''
    This calculates the rich club coefficient for each degree
    value in the graph (G).

    Inputs:
        G      - networkx graph
        R_list - list of random graphs with matched degree distribution
                   if R_list is None then a random graph is calculated
                   within the code
                   if len(R_list) is less than n then the remaining random
                   graphs are calculated within the code
                 Default R_list = None
        n ------ number of random graphs for which to calculate rich club
                   coefficients
                 Default n = 10

    Returns:
        rc ------ dictionary of rich club coefficients for the real graph
        rc_rand - array of rich club coefficients for the n random graphs
    '''
    # First, calculate the rich club coefficient for the regular graph
    rc_dict = nx.rich_club_coefficient(G, normalized=False)

    # Save the degrees as a numpy array
    deg = list(rc_dict.keys())

    # Save the rich club coefficients as a numpy array
    rc = list(rc_dict.values())

    # Calculate n different random graphs and their
    # rich club coefficients

    # Start by creating an empty array that will hold
    # the n random graphs' rich club coefficients
    rc_rand = np.ones([len(rc), n])

    for i in range(n):
        # If you haven't already calculated random graphs
        # or you haven't given this function as many random
        # graphs as it is expecting then calculate a random
        # graph here
        if not R_list or len(R_list) <= i:
            R = mkg.random_graph(G)
        # Otherwise just use the one you already made
        else:
            R = R_list[i]

        # Calculate the rich club coefficient
        rc_rand_dict = nx.rich_club_coefficient(R, normalized=False)

        # And save the values to the numpy array you created earlier
        rc_rand[:, i] = list(rc_rand_dict.values())

    return deg, rc, rc_rand


def calculate_global_measures(G,
                              R_list=None,
                              n_rand=10,
                              nodal_partition=None,
                              R_nodal_partition_list=None):
    '''
    A wrapper function that calls a bunch of useful functions
    and reports a plethora of network measures for the real graph
    G, and for n random graphs that are matched on degree distribution
    (unless otherwise stated)
    '''
    import networkx as nx
    import numpy as np

    # ==== SET UP ======================
    # If you haven't already calculated random graphs
    # or you haven't given this function as many random
    # graphs as it is expecting then calculate a random
    # graph here
    if R_list is None:
        R_list, R_nodal_partition_list = get_random_graphs(G, n_rand)
    else:
        n = len(R_list)

    # If you haven't passed the nodal partition
    # then calculate it here
    if not nodal_partition:
        nodal_partition = gm.calc_nodal_partition(G)

    # ==== MEASURES ====================
    global_measures_dict = {}

    # ---- Clustering coefficient ------
    global_measures_dict['C'] = nx.average_clustering(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = nx.average_clustering(R_list[i])
    global_measures_dict['C_rand'] = rand_array

    # ---- Shortest path length --------
    global_measures_dict['L'] = nx.average_shortest_path_length(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = nx.average_shortest_path_length(R_list[i])
    global_measures_dict['L_rand'] = rand_array

    # ---- Assortativity ---------------
    global_measures_dict['a'] = np.mean(nx.degree_assortativity_coefficient(G))
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = np.mean(nx.degree_assortativity_coefficient(R_list[i]))
    global_measures_dict['a_rand'] = rand_array

    # ---- Modularity ------------------
    global_measures_dict['M'] = gm.calc_modularity(G, nodal_partition)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = gm.calc_modularity(R_list[i], R_nodal_partition_list[i])
    global_measures_dict['M_rand'] = rand_array

    #  ---- Efficiency ------------------
    global_measures_dict['E'] = gm.calc_efficiency(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = gm.calc_efficiency(R_list[i])
    global_measures_dict['E_rand'] = rand_array

    # ---- Small world -----------------
    sigma_array = np.ones(n)
    for i in range(n):
        sigma_array[i] = ((global_measures_dict['C']
                           / global_measures_dict['C_rand'][i])
                          / (global_measures_dict['L']
                             / global_measures_dict['L_rand'][i]))
    global_measures_dict['sigma'] = sigma_array
    global_measures_dict['sigma_rand'] = 1.0

    return global_measures_dict
