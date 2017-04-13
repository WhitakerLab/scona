#!/usr/bin/env python

def make_graphs(graph_dir, mat_dict, centroids, aparc_names, n_rand=1000):
    '''
    A function that makes all the required graphs from the correlation
    matrices in mat_dict. These include the full graph with all
    connections including weights, and binarized graphs at 30 different
    costs betwen 1% to 30%. These graphs are fully connected because the
    minimum spanning tree is used before the strongest edges are added
    up to the required density.

    If the graphs do not already exist they are saved as gpickle files in
    graph_dir. If they do exist then they're read in from those files.

    In addition, files with values for the nodal topological measures and
    global topological measures are created and saved or loaded as
    appropriate.

    The function requires the centroids and aparc_names values in order
    to calculate the nodal measures. The value n_rand is the number of
    random graphs to calculate for the global and nodal measure
    calculations.

    The function returns a dictionary of graphs, nodal measures and
    global measures
    '''
    #==========================================================================
    # IMPORTS
    #==========================================================================
    import os
    import networkx as nx
    import numpy as np
    import pickle

    #==========================================================================
    # Print to screen what you're up to
    #==========================================================================
    print "--------------------------------------------------"
    print "Making or loading graphs"

    #==========================================================================
    # Create an empty dictionary
    #==========================================================================
    graph_dict = {}

    #==========================================================================
    # Loop through all the matrices in mat_dict
    #==========================================================================
    for k in mat_dict.keys():

        print '    {}'.format(k)

        # Read in the matrix
        M = mat_dict[k]

        # Get the covars name
        mat_name, covars_name = k.split('_COVARS_')

        #-------------------------------------------------------------------------
        # Make the full graph first
        #-------------------------------------------------------------------------
        # Define the graph's file name and its dictionary key
        g_filename = os.path.join(graph_dir,
                                    'COVARS_{}'.format(covars_name),
                                    'Graph_{}_COST_100.gpickle'.format(mat_name))

        g_key = '{}_COST_100'.format(k)
        print '      Loading COST: 100',

        # If it already exists just read it in from the pickled file
        if os.path.isfile(g_filename):
            graph_dict[g_key] = nx.read_gpickle(g_filename)

        # Otherwise you'll have to create it using the graph_at_cost function above
        else:
            graph_dict[g_key] = full_graph(M)

            # Save it as a gpickle file so you don't have to do this next time!
            dirname = os.path.dirname(g_filename)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            nx.write_gpickle(graph_dict[g_key], g_filename)

        #-------------------------------------------------------------------------
        # Then for all the different costs between 1% and 30%
        #-------------------------------------------------------------------------
        for cost in [2] + range(5,21,5):

            #-------------------------------------------------------------------------
            # Define the graph's file name along with those of the the associated
            # global and nodal dictionaries
            #-------------------------------------------------------------------------
            g_filename = os.path.join(graph_dir,
                                            'COVARS_{}'.format(covars_name),
                                            'Graph_{}_COST_{:02.0f}.gpickle'.format(mat_name, cost))
            global_dict_filename = os.path.join(graph_dir,
                                            'COVARS_{}'.format(covars_name),
                                            'GlobalDict_{}_COST_{:02.0f}.p'.format(mat_name, cost))
            nodal_dict_filename = os.path.join(graph_dir,
                                            'COVARS_{}'.format(covars_name),
                                            'NodalDict_{}_COST_{:02.0f}.p'.format(mat_name, cost))
            rich_club_filename = os.path.join(graph_dir,
                                            'COVARS_{}'.format(covars_name),
                                            'RichClub_{}_COST_{:02.0f}.p'.format(mat_name, cost))

            g_key = '{}_COST_{:02.0f}'.format(k, cost)

            #-------------------------------------------------------------------------
            # Make or load the graph
            #-------------------------------------------------------------------------
            # If the graph already exists just read it in from the pickled file
            if os.path.isfile(g_filename):
                graph_dict[g_key] = nx.read_gpickle(g_filename)

            # Otherwise you'll have to create it using the graph_at_cost function
            else:
                graph_dict[g_key] = graph_at_cost(M, cost)

                # Save it as a gpickle file so you don't have to do this next time!
                nx.write_gpickle(graph_dict[g_key], g_filename)

            #-------------------------------------------------------------------------
            # Make or load the global and nodal measures dictionaries
            #-------------------------------------------------------------------------
            # If the rich_club measures files already exists just read it
            # and the nodal and global measures files in
            if os.path.isfile(rich_club_filename):

                # Print to screen so you know where you're up to
                if cost == 20:
                    print '- {:02.0f}'.format(cost)
                else:
                    print '- {:02.0f}'.format(cost),

                graph_dict['{}_GlobalMeasures'.format(g_key)] = pickle.load(open(global_dict_filename))
                graph_dict['{}_NodalMeasures'.format(g_key)] = pickle.load(open(nodal_dict_filename))
                graph_dict['{}_RichClub'.format(g_key)] = pickle.load(open(rich_club_filename))

            # Otherwise you'll have to create them using the calculate_global_measures
            # and calculate_nodal_measures functions
            else:
                G = graph_dict[g_key]

                print '\n      Calculating COST: {:02.0f}'.format(cost)

                # You need to calculate the same nodal partition for the global
                # and nodal measures
                nodal_partition = calc_nodal_partition(G)

                # And you'll also want the same list of random graphs
                R_list, R_nodal_partition_list = make_random_list(G, n_rand=n_rand)

                graph_dict['{}_GlobalMeasures'.format(g_key)] = calculate_global_measures(G,
                                                                                           R_list=R_list,
                                                                                           nodal_partition=nodal_partition,
                                                                                           R_nodal_partition_list=R_nodal_partition_list)
                (graph_dict[g_key],
                    graph_dict['{}_NodalMeasures'.format(g_key)]) = calculate_nodal_measures(G,
                                                                                              centroids,
                                                                                              aparc_names,
                                                                                              nodal_partition=nodal_partition)
                graph_dict['{}_RichClub'.format(g_key)] = rich_club(G, R_list=R_list)

                # Save them as pickle files so you don't have to do this next time!
                pickle.dump(graph_dict['{}_GlobalMeasures'.format(g_key)], open(global_dict_filename, "wb"))
                pickle.dump(graph_dict['{}_NodalMeasures'.format(g_key)], open(nodal_dict_filename, "wb"))
                pickle.dump(graph_dict['{}_RichClub'.format(g_key)], open(rich_club_filename, "wb"))
                nx.write_gpickle(graph_dict[g_key], g_filename)

    # Return the full graph dictionary
    return graph_dict


def full_graph(M):
    '''
    Very easy, set the diagonals to 0
    and then save the graph
    '''
    import numpy as np
    import networkx as nx

    # Make a copy of the matrix
    thr_M = np.copy(M)

    # Set all diagonal values to 0
    thr_M[np.diag_indices_from(thr_M)] = 0

    # Read this full matrix into a graph G
    G = nx.from_numpy_matrix(thr_M)

    return G


def graph_at_cost(M, cost):
    '''
    A function that first creates the minimum spanning tree
    for the graph, and then adds in edges according to their
    connection strength up to a particular cost
    '''
    import numpy as np
    import networkx as nx

    # Make a copy of the matrix
    thr_M = np.copy(M)

    # Set all diagonal values to 0
    thr_M[np.diag_indices_from(thr_M)] = 0

    # Multiply all values by -1 because the minimum spanning tree
    # looks for the smallest distance - not the largest correlation!
    thr_M = thr_M*-1

    # Read this full matrix into a graph G
    G = nx.from_numpy_matrix(thr_M)

    # Make a list of all the sorted edges in the full matrix
    G_edges_sorted = [ edge for edge in sorted(G.edges(data = True), key = lambda (a, b, dct): dct['weight']) ]

    # Calculate minimum spanning tree and make a list of the mst_edges
    mst = nx.minimum_spanning_tree(G)
    mst_edges = mst.edges(data = True)

    # Create a list of edges that are *not* in the mst
    # (because you don't want to add them in twice!)
    G_edges_sorted_notmst = [ edge for edge in G_edges_sorted if not edge in mst_edges ]

    # Figure out the number of edges you want to keep for this
    # particular cost. You have to round this number because it
    # won't necessarily be an integer, and you have to subtract
    # the number of edges in the minimum spanning tree because we're
    # going to ADD this number of edges to it
    n_edges = (cost/100.0) * len(G_edges_sorted)
    n_edges = np.int(np.around(n_edges))
    n_edges = n_edges - len(mst.edges())

    # If your cost is so small that your minimum spanning tree already covers it
    # then you can't do any better than the MST and you'll just have to return
    # it with an accompanying error message
    if n_edges < 0:
        print 'Unable to calculate matrix at this cost - minimum spanning tree is too large'

    # Otherwise, add in the appropriate number of edges (n_edges)
    # from your sorted list (G_edges_sorted_notmst)
    else:
        mst.add_edges_from(G_edges_sorted_notmst[:n_edges])

    # And return the *updated* minimum spanning tree
    # as your graph
    return mst


def make_random_list(G, n_rand=10):
    '''
    A little (but useful) function to wrap
    around random_graph and return a list of
    random graphs (matched for degree distribution)
    that can be passed to multiple calculations so
    you don't have to do it multiple times
    '''
    R_list = []
    R_nodal_partition_list = []

    print '        Creating {} random graphs - may take a little while'.format(n_rand)

    for i in range(n_rand):
        if len(R_list) <= i:
            R_list += [ random_graph(G) ]

            R_nodal_partition_list += [ calc_nodal_partition(R_list[i]) ]

    return R_list, R_nodal_partition_list


def calculate_global_measures(G, R_list=None, n_rand=10, nodal_partition=None, R_nodal_partition_list=None):
    '''
    A wrapper function that calls a bunch of useful functions
    and reports a plethora of network measures for the real graph
    G, and for n random graphs that are matched on degree distribution
    (unless otherwise stated)
    '''
    import networkx as nx
    import numpy as np

    #==== SET UP ======================
    # If you haven't already calculated random graphs
    # or you haven't given this function as many random
    # graphs as it is expecting then calculate a random
    # graph here
    if R_list is None:
        R_list, R_nodal_partition_list = make_random_list(n_rand)
    else:
        n = len(R_list)

    # If you haven't passed the nodal partition
    # then calculate it here
    if not nodal_partition:
        nodal_partition = calc_nodal_partition(G)

    #==== MEASURES ====================
    global_measures_dict = {}

    #---- Clustering coefficient ------
    global_measures_dict['C'] = nx.average_clustering(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = nx.average_clustering(R_list[i])
    global_measures_dict['C_rand'] = rand_array

    #---- Shortest path length --------
    global_measures_dict['L'] = nx.average_shortest_path_length(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = nx.average_shortest_path_length(R_list[i])
    global_measures_dict['L_rand'] = rand_array

    #---- Assortativity ---------------
    global_measures_dict['a'] = np.mean(nx.degree_assortativity_coefficient(G))
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = np.mean(nx.degree_assortativity_coefficient(R_list[i]))
    global_measures_dict['a_rand'] = rand_array

    #---- Modularity ------------------
    global_measures_dict['M'] = calc_modularity(G, nodal_partition)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = calc_modularity(R_list[i], R_nodal_partition_list[i])
    global_measures_dict['M_rand'] = rand_array

    #---- Efficiency ------------------
    global_measures_dict['E'] = calc_efficiency(G)
    rand_array = np.ones(n)
    for i in range(n):
        rand_array[i] = calc_efficiency(R_list[i])
    global_measures_dict['E_rand'] = rand_array

    #---- Small world -----------------
    sigma_array = np.ones(n)
    for i in range(n):
        sigma_array[i] = ( ( global_measures_dict['C'] / global_measures_dict['C_rand'][i] )
                            / ( global_measures_dict['L'] / global_measures_dict['L_rand'][i] ) )
    global_measures_dict['sigma'] = sigma_array
    global_measures_dict['sigma_rand'] = 1.0

    return global_measures_dict


def calculate_nodal_measures(G, centroids, aparc_names, nodal_partition=None, names_308_style=True):
    '''
    A function which returns a dictionary of numpy arrays for a graph's
        * degree
        * participation coefficient
        * average distance
        * total distance
        * clustering
        * closeness
        * interhemispheric proportion
        * name
    If you have names in 308 style (as described in Whitaker, Vertes et al 2016)
    then you can also add in
        * hemisphere
        * 34_name (Desikan Killiany atlas region)
        * 68_name (Desikan Killiany atlas region with hemisphere)
    '''
    import numpy as np
    import networkx as nx

    #==== SET UP ======================
    # If you haven't passed the nodal partition
    # then calculate it here
    if not nodal_partition:
        nodal_partition = calc_nodal_partition(G)


    #==== MEASURES ====================
    nodal_dict = {}

    #---- Degree ----------------------
    deg = G.degree().values()
    nodal_dict['degree'] = np.array(deg)

    #---- Closeness -------------------
    closeness = nx.closeness_centrality(G).values()
    nodal_dict['closeness'] = np.array(closeness)

    #---- Betweenness -----------------
    betweenness = nx.betweenness_centrality(G).values()
    nodal_dict['betweenness'] = np.array(betweenness)

    #---- Shortest path length --------
    L = shortest_path(G).values()
    nodal_dict['shortest_path'] = np.array(L)

    #---- Clustering ------------------
    clustering = nx.clustering(G).values()
    nodal_dict['clustering'] = np.array(clustering)

    #---- Participation coefficent ----
    #---- and module assignment -------
    partition, pc_dict = participation_coefficient(G, nodal_partition)
    nodal_dict['module'] = np.array(partition.values())
    nodal_dict['pc'] = np.array(pc_dict.values())

    #---- Euclidean distance and ------
    #---- interhem proporition --------
    G = assign_nodal_distance(G, centroids)
    average_dist = nx.get_node_attributes(G, 'average_dist').values()
    total_dist = nx.get_node_attributes(G, 'total_dist').values()
    interhem_prop = nx.get_node_attributes(G, 'interhem_proportion').values()

    nodal_dict['average_dist'] = np.array(average_dist)
    nodal_dict['total_dist'] = np.array(total_dist)
    nodal_dict['interhem_prop'] = np.array(interhem_prop)

    #---- Names -----------------------
    G = assign_node_names(G, aparc_names, names_308_style=names_308_style)
    name = nx.get_node_attributes(G, 'name').values()
    nodal_dict['name'] = np.array(name)
    if names_308_style:
        name_34 = nx.get_node_attributes(G, 'name_34').values()
        name_68 = nx.get_node_attributes(G, 'name_68').values()
        hemi = nx.get_node_attributes(G, 'hemi').values()
        nodal_dict['name_34'] = np.array(name_34)
        nodal_dict['name_68'] = np.array(name_68)
        nodal_dict['hemi'] = np.array(hemi)

    return G, nodal_dict


def random_graph(G, Q=10):
    '''
    Create a random graph that preserves degree distribution
    by swapping pairs of edges (double edge swap).

    Inputs:
        G: networkx graph
        Q: constant that determines how many swaps to conduct
           for every edge in the graph
           Default Q =10

    Returns:
        R: networkx graph

    CAVEAT: If it is not possible in 15 attempts to create a
    connected random graph then this code will just return the
    original graph (G). This means that if you come to look at
    the values that are an output of calculate_global_measures
    and see that the values are the same for the random graph
    as for the main graph it is not necessarily the case that
    the graph is random, it may be that the graph was so low cost
    (density) that this code couldn't create an appropriate random
    graph!

    This should only happen for ridiculously low cost graphs that
    wouldn't make all that much sense to investigate anyway...
    so if you think carefully it shouldn't be a problem.... I hope!
    '''

    import networkx as nx

    # Copy the graph
    R = G.copy()

    # Calculate the number of edges and set a constant
    # as suggested in the nx documentation
    E = R.number_of_edges()

    # Start with assuming that the random graph is not connected
    # (because it might not be after the first permuatation!)
    connected=False
    attempt=0

    # Keep making random graphs until they are connected!
    while not connected and attempt < 15:
        # Now swap some edges in order to preserve the degree distribution
        nx.double_edge_swap(R,Q*E,max_tries=Q*E*10)

        # Check that this graph is connected! If not, start again
        connected = nx.is_connected(R)
        if not connected:
            attempt +=1

    if attempt == 15:
        print '          ** Attempt aborted - can not randomise graph **'
        R = G.copy()

    return R


def calc_modularity(G, nodal_partition):
    '''
    A function that calculates modularity from the best partition
    of a graph using the louvain method
    '''
    import community

    modularity = community.modularity(nodal_partition, G)

    return modularity


def calc_nodal_partition(G):
    '''
    You only need to create the nodal partition using the
    community module once. It takes a while and can be
    different every time you try so it's best to save a
    partition and use that for any subsequent calculations
    '''
    import community

    # Make sure the edges are binarized
    for u,v,d in G.edges(data=True):
        d['weight']=1

    # Now calculate the best partition
    nodal_partition = community.best_partition(G)

    return nodal_partition


def calc_efficiency(G):
    '''
    A little wrapper to calculate global efficiency
    '''
    import networkx as nx

    E=0.0
    for node in G:
        path_length=nx.single_source_shortest_path_length(G, node)
        E += 1.0/sum(path_length.values())

    return E


def participation_coefficient(G, nodal_partition):
    '''
    Computes the participation coefficient for each node (Guimera et al. 2005).

    Returns dictionary of the participation coefficient for each node.
    '''
    # Import the modules you'll need
    import networkx as nx
    import numpy as np

    # Reverse the dictionary because the output of Louvain is "backwards"
    # meaning it saves the module per node, rather than the nodes in each
    # module
    module_partition = {}
    for m,n in zip(nodal_partition.values(),nodal_partition.keys()):
        try:
            module_partition[m].append(n)
        except KeyError:
            module_partition[m] = [n]

    # Create an empty dictionary for the participation
    # coefficients
    pc_dict = {}
    all_nodes = set(G.nodes())

    # Print a little note to the screen because it can take a long
    # time to run this code
    print '        Calculating participation coefficient - may take a little while'

    # Loop through modules
    for m in module_partition.keys():

        # Get the set of nodes in this module
        mod_list = set(module_partition[m])

        # Loop through each node (source node) in this module
        for source in mod_list:

            # Calculate the degree for the source node
            degree = float(nx.degree(G=G, nbunch=source))

            # Calculate the number of these connections
            # that are to nodes in *other* modules
            count = 0

            for target in mod_list:

                # If the edge is in there then increase the counter by 1
                if (source, target) in G.edges():
                    count += 1

            # This gives you the within module degree
            wm_degree = float(count)

            # The participation coeficient is 1 - the square of
            # the ratio of the within module degree and the total degree
            pc = 1 - ((float(wm_degree) / float(degree))**2)

            # Save the participation coefficient to the dictionary
            pc_dict[source] = pc

    return nodal_partition, pc_dict


def assign_nodal_distance(G, centroids):
    '''
    Give each node in the graph their
    x, y, z coordinates and then calculate the eucledian
    distance for every edge that connects to each node

    Also calculate the number of interhemispheric edges
    (defined as edges which different signs for the x
    coordinate

    Returns the graph
    '''
    import networkx as nx
    import numpy as np
    from scipy.spatial import distance

    # First assign the x, y, z values to each node
    for i, node in enumerate(G.nodes()):
        G.node[node]['x'] = centroids[i, 0]
        G.node[node]['y'] = centroids[i, 1]
        G.node[node]['z'] = centroids[i, 2]
        G.node[node]['centroids'] = centroids[i, :]

    # Loop through every node in turn
    for i, node in enumerate(G.nodes()):
        # Loop through the edges connecting to this node
        # Note that "node1" should always be exactly the same
        # as "node", I've just used another name to keep
        # the code clear (which I may not have achieved given
        # that I thought this comment was necesary...)
        for node1, node2 in G.edges(nbunch=[node]):

            # Calculate the eulidean distance for this edge
            cent1 = G.node[node1]['centroids']
            cent2 = G.node[node2]['centroids']

            dist = distance.euclidean(cent1, cent2)

            # And assign this value to the edge
            G.edge[node1][node2]['euclidean'] = dist

            # Also figure out whether this edge is interhemispheric
            # by multiplying the x values. If x1 * x2 is negative
            # then the nodes are in different hemispheres.
            x1 = G.node[node1]['x']
            x2 = G.node[node2]['x']

            if x1*x2 > 0:
                G.edge[node1][node2]['interhem'] = 0
            else:
                G.edge[node1][node2]['interhem'] = 1

        # Create two nodal attributes (average distance and
        # total distance) by summarizing the euclidean distance
        # for all edges which connect to the node
        euc_list = [ G.edge[m][n]['euclidean'] for m, n in G.edges(nbunch=node) ]

        G.node[node]['average_dist'] = np.mean(euc_list)
        G.node[node]['total_dist'] = np.sum(euc_list)

        # Create an interhem nodal attribute by getting the average
        # of the interhem values for all edges which connect to the node

        interhem_list = [ G.edge[m][n]['interhem'] for m, n in G.edges(nbunch=node) ]

        G.node[node]['interhem_proportion'] = np.mean(interhem_list)

    return G

def shortest_path(G):
    import networkx as nx
    import numpy as np

    shortestpl_dict_dict = nx.shortest_path_length(G)

    shortestpl_dict = {}

    for node in G.nodes():
        shortestpl_dict[node] = np.average(shortestpl_dict_dict[node].values())

    return shortestpl_dict


def assign_node_names(G, aparc_names, names_308_style=True):

    # Assign names to the nodes
    for i, node in enumerate(G.nodes()):
        G.node[node]['name'] = aparc_names[i]
        if names_308_style:
            G.node[node]['name_34'] = aparc_names[i].split('_')[1]
            G.node[node]['name_68'] = aparc_names[i].rsplit('_',1)[0]
            G.node[node]['hemi'] = aparc_names[i].split('_',1)[0]

    return G


def rich_club(G, R_list=None, n=10):
    '''
    This calculates the rich club coefficient for each degree
    value in the graph (G).

    Inputs:
        G ------ networkx graph
        R_list - list of random graphs with matched degree distribution
                   if R_list is None then a random graph is calculated
                   within the code
                   if len(R_list) is less than n then the remaining random graphs
                   are calculated within the code
                 Default R_list = None
        n ------ number of random graphs for which to calculate rich club
                   coefficients
                 Default n = 10

    Returns:
        rc ------ dictionary of rich club coefficients for the real graph
        rc_rand - array of rich club coefficients for the n random graphs
    '''
    # Import the modules you'll need
    import networkx as nx
    import numpy as np

    # First, calculate the rich club coefficient for the regular graph
    rc_dict = nx.rich_club_coefficient(G, normalized=False)

    # Save the degrees as a numpy array
    deg = np.array(rc_dict.keys())

    # Save the rich club coefficients as a numpy array
    rc = np.array(rc_dict.values())

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
            R = random_graph(G)
        # Otherwise just use the one you already made
        else:
            R = R_list[i]

        # Calculate the rich club coefficient
        rc_rand_dict = nx.rich_club_coefficient(R, normalized=False)

        # And save the values to the numpy array you created earlier
        rc_rand[:, i] = rc_rand_dict.values()

    return deg, rc, rc_rand
