import numpy as np
import networkx as nx
import pandas as pd
from scona.make_graphs import assign_node_names, \
    assign_node_centroids, anatomical_copy, threshold_graph, \
    weighted_graph_from_matrix, anatomical_node_attributes, \
    anatomical_graph_attributes, get_random_graphs, is_nodal_match, \
    is_anatomical_match
from scona.graph_measures import assign_interhem, \
    calculate_nodal_measures, assign_nodal_distance, \
    calc_nodal_partition, calculate_global_measures, small_coefficient


class BrainNetwork(nx.classes.graph.Graph):
    '''
    A lightweight subclass of :class:`networkx.Graph`.

    Parameters
    ----------
    network : :class:`networkx.Graph` or :class:`numpy.ndarray` or :class:`pandas.DataFrame`, optional
        `network` is used to define graph features of G.
        If an array is passed, create a weighted graph from this array.
        If a graph is passed, G will be a :class:`BrainNetwork` otherwise
        identical to this graph.
    parcellation : list of str, optional
        A list of node names, passed to :func:`BrainNetwork.set_parcellation`
    centroids : list of tuple, optional
        A list of node centroids, passed to :func:`BrainNetwork.set_centroids`


    Attributes
    ----------
    anatomical_node_attributes : list of str
        a list of node attribute keys to treat as anatomical.
    anatomical_graph_attributes : list of str
        a list of graph attribute keys to treat as anatomical.


    See Also
    --------
    :class:`networkx.Graph`
    '''
    def __init__(self,
                 network=None,
                 parcellation=None,
                 centroids=None,):
        # ============== Create graph ================
        nx.classes.graph.Graph.__init__(self)
        if network is not None:
            if isinstance(network, nx.classes.graph.Graph):
                # Copy graph
                self.__dict__.update(network.__dict__)

            else:
                # Create weighted graph from a dataframe or
                # numpy array
                if isinstance(network, pd.DataFrame):
                    M = network.values
                elif isinstance(network, np.ndarray):
                    M = network
                network = weighted_graph_from_matrix(M, create_using=self)

        # ===== Give anatomical labels to nodes ======
        if parcellation is not None:
            # assign parcellation names to nodes
            self.set_parcellation(parcellation)
        if centroids is not None:
            # assign centroids to nodes
            self.set_centroids(centroids)
        # Tell BrainNetwork class which attributes to consider "anatomical"
        # and therefore preserve when copying or creating new graphs
        self.anatomical_node_attributes = anatomical_node_attributes()
        self.anatomical_graph_attributes = anatomical_graph_attributes()
        # intialise global measures as an empty dict
        self.graph['global_measures'] = {}

    def threshold(self, cost, mst=True):
        '''
        Create a binary graph by thresholding weighted BrainNetwork G

        First creates the minimum spanning tree for the graph, and then adds
        in edges according to their connection strength up to cost.

        Parameters
        ----------
        cost : float
            A number between 0 and 100. The resulting graph will have the
            ``cost*n/100`` highest weighted edges from G, where
            ``n`` is the number of edges in G.
        mst : bool, optional
            If ``False``, skip creation of minimum spanning tree. This may
            cause output graph to be disconnected

        Returns
        -------
        :class:`BrainNetwork`
            A binary graph

        Raises
        ------
        Exception
            If it is impossible to create a minimum_spanning_tree at the given
            cost

        See Also
        --------
        :func:`threshold_graph`
        '''
        return threshold_graph(self, cost, mst=True)

    def partition(self, force=False):
        '''
        Calculate the best nodal partition of G using the louvain
        algorithm as implemented in :func:`community.best_partition`.
        If desired, the node-wise partition can be accessed and manipulated by
        the nodal attribute "module" and the module-wise partition as
        ``G.graph["partition"]``

        Parameters
        ----------
        force : bool
            pass True to recalculate when a partition exists

        Returns
        -------
        (dict, dict)
            Two dictionaries represent the resulting nodal partition of G. The
            first maps nodes to modules and the second maps modules to nodes.

        See Also
        --------
        :func:`BrainNetwork.calc_nodal_partition`
        '''
        if 'partition' not in self.graph:
            nodal_partition, module_partition = calc_nodal_partition(self)
            nx.set_node_attributes(self, nodal_partition, name='module')
            self.graph['partition'] = module_partition
        return nx.get_node_attributes(self, name='module'), self.graph['partition']

    def set_parcellation(self, parcellation):
        '''
        Modify nodal attribute "name" for nodes of G inplace.

        Parameters
        ----------
        G : :class:`networkx.Graph`
        parcellation : list
            ``parcellation[i]`` is the name of node ``i`` in ``G``

        See Also
        --------
        :func:`assign_node_names`
        :func:`BrainNetwork.set_centroids`
        '''
        assign_node_names(self, parcellation)

    def set_centroids(self, centroids):
        '''
        Modify the nodal attributes "centroids", "x", "y", and "z" of nodes
        of G inplace.

        Parameters
        ----------
        G : :class:`networkx.Graph`
        centroids : list
            ``centroids[i]`` is a tuple representing the cartesian coordinates
            of node ``i`` in ``G``. "x", "y" and "z" will be assigned as the
            first, second and third coordinate of ``centroids[i]``
            respectively.

        See Also
        --------
        :func:`BrainNetwork.calculate_spatial_values`
        :func:`BrainNetwork.set_parcellation`
        :func:`assign_node_centroids`
        '''
        assign_node_centroids(self, centroids)

    def calculate_spatial_values(self):
        '''
        Assigns spatial nodal and edge attributes of G. Modifies G in place.

        Edge attributes:

        "euclidean" : float
            the euclidean length, as derived from the nodal attribute
            "centroids"
        "interhem" : int
            1 if the edge is interhemispheric, 0 otherwise

        Node attributes

        "total_dist" : float
            the total length of the incident edges
        "average_dist" : float
            the average length of the incident edges
        "hemisphere" : int
            1 or -1, represents the sign of the x coordinate
        "interhem" : int
            the number of adjacent interhemispheric edges
        "interhem_proportion" : float
            the proportion of adjacent edges that are interhemispheric

        Raises
        ------
        KeyError
            If the graph centroids are not set

        See Also
        --------
        :func:`BrainNetwork.set_centroids`
        :func:`assign_interhem`
        :func:`assign_nodal_distance`
        '''
        if not self.graph.get("centroids"):
            raise KeyError("Cannot calculate spatial values if centroids are \
            not set")
        assign_nodal_distance(self)
        assign_interhem(self)

    def calculate_nodal_measures(
            self,
            force=False,
            measure_list=None,
            additional_measures=None):
        '''
        Calculate and store nodal measures as nodal attributes
        which can be accessed directly, or using
        :func:`BrainNetwork.report_nodal_measures()``

        By default calculates:

        * "nodal_partition" : int or str
        * "degree" : int
        * "closeness" : float
        * "betweenness" : float
        * "shortest_path_length" : float
        * "clustering" : float
        * "participation_coefficient" : float

        Use `measure_list` to specify which of the default nodal attributes to
        calculate.
        Use `additional_measures` to describe and calculate new measure
        definitions.

        Parameters
        ----------
        measure_list : list of str, optional
            pass a subset of of the keys defined above to specify which of the
            default measures to calculate
        additional_measures : dict, optional
            map from names of nodal attributes to functions
            defining how they should be calculated. Such a function should take
            a graph as an argument and return a dictionary mapping nodes to
            attribute values.
        force : bool, optional
            pass True to recalculate any measures that already
            exist in the nodal attributes.

        See Also
        --------
        :func:`BrainNetwork.report_nodal_measures`
        :func:`BrainNetwork.partition`
        :func:`calculate_nodal_measures`

        Example
        -------

        '''
        # ==== SET UP ================================
        # Ensure nodal partition exists
        a, partition = self.partition()
        # ==== calculate nodal measures ==============
        calculate_nodal_measures(
            self,
            partition=partition,
            force=force,
            measure_list=measure_list,
            additional_measures=additional_measures)

    def report_nodal_measures(self, columns=None, as_dict=False):
        '''
        Report the nodal attributes of G as a pandas dataframe or python
        dictionary

        Parameters
        ----------
        columns : None or list, optional
            pass a list of nodal attributes to columns to specify which should
            be reported
        as_dict : bool, optional
            pass True to return a nested dictionary instead of a dataframe.

        Returns
        -------
        :class:`pandas.DataFrame` or dict
            the node attribute data from G as a pandas dataframe or dict of
            dicts

        See Also
        --------
        :func:`BrainNetwork.calculate_nodal_measures`
        :func:`BrainNetwork.calculate_spatial_values`
        '''
        if columns is not None:
            nodal_dict = {x: {u: v for u, v in y.items() if u in columns}
                          for x, y in self._node.items()}
        else:
            nodal_dict = self._node
        if as_dict:
            return nodal_dict
        df = pd.DataFrame(nodal_dict).transpose()
        return df

    def rich_club(self, force=False):
        '''
        Calculate the rich club coefficient of G for each degree between 0 and
        ``max([degree(v) for v in G.nodes])``. The resulting dictionary of rich
        club coefficients can be accessed and manipulated as
        ``G.graph['rich_club']``

        Parameters
        ----------
        force : bool
            pass True to recalculate when a dictionary of rich club
            coefficients already exists

        Returns
        -------
        dict
            a dictionary mapping integer `x` to the rich club coefficient of
            G for degree `x`

        See Also
        --------
        :func:`rich_club`
        '''
        if ('rich_club' not in self.graph) or force:
            self.graph['rich_club'] = nx.rich_club_coefficient(
                                        self, normalized=False)
        return self.graph['rich_club']

    def calculate_global_measures(
            self, force=False, seed=None, partition=True):

        '''
        Calculate global measures `average_clustering`,
        `average_shortest_path_length`, `assortativity`, `modularity`, and
        `efficiency` of G. The resulting dictionary of global measures can be
        accessed and manipulated as ``G.graph['global_measures']``

        Parameters
        ----------
        force : bool
            pass True to recalculate any global measures that have already
            been calculated
        partition : bool
            The "modularity" measure evaluates a graph partition.
            pass True to calculate the partition of each graph using
            :func:`BrainNetwork.partition`. Note that this won't recalculate
            a partition that exists.
            If False, modularity may not be calculated.


        Returns
        -------
        dict
            a dictionary of global network measures of G

        See Also
        --------
        :func:`calculate_global_measures`
        '''
        if partition:
            self.partition()
            partition = nx.get_node_attributes(self, name='module')
        else:
            partition = None

        if force:
            global_measures = calculate_global_measures(
                self, partition=partition)
            self.graph['global_measures'] = global_measures
        else:
            global_measures = calculate_global_measures(
                self,
                partition=partition,
                existing_global_measures=self.graph.get('global_measures'))
            self.graph['global_measures'].update(global_measures)
        return self.graph['global_measures']

    def anatomical_copy(self):
        '''
        Create a new graph from G preserving:
        * nodes
        * edges
        * any nodal attributes specified in G.anatomical_node_attributes
        * any graph attributes specified in G.anatomical_graph_attributes
        * ``G.anatomical_node_attributes``
        * ``G.anatomical_graph_attributes``

        Returns
        -------
        :class:`networkx.Graph`
            A new graph with the same nodes and edges as G and identical
            anatomical data.

        See Also
        --------
        :func:`BrainNetwork.anatomical_copy`
        :func:`anatomical_data`
        :func:`set_anatomical_node_attributes`
        :func:`set_anatomical_graph_attributes`
        '''
        H = anatomical_copy(self,
                            nodal_keys=self.anatomical_node_attributes,
                            graph_keys=self.anatomical_graph_attributes)
        H.set_anatomical_node_attributes(self.anatomical_node_attributes)
        H.set_anatomical_graph_attributes(self.anatomical_graph_attributes)
        return H

    def set_anatomical_node_attributes(self, names):
        '''
        Define the list of node attribute keys to preserve when using
        :func:`BrainNetwork.anatomical_copy`.
        This list is set using :func:`anatomical_node_attributes` when
        a BrainNetwork is initialised, and can be accessed as
        ``G.anatomical_node_attributes``
        It is also among the object attributes preserved by
        :func:`BrainNetwork.anatomical_copy`

        Parameters
        ----------
        names : list
            a list of node attribute keys to treat as anatomical.

        See Also
        --------
        :func:`BrainNetwork.set_anatomical_graph_attributes`
        '''
        self.anatomical_node_attributes = names

    def set_anatomical_graph_attributes(self, names):
        '''
        Define the list of graph attribute keys to preserve when using
        :func:`BrainNetwork.anatomical_copy`.
        This list is set using :func:`anatomical_graph_attributes` when
        a BrainNetwork is initialised, and can be accessed as
        ``G.anatomical_graph_attributes``
        It is also among the object attributes preserved by
        :func:`BrainNetwork.anatomical_copy`

        Parameters
        ----------
        names : list
            a list of graph attribute keys to treat as anatomical.

        See Also
        --------
        :func:`BrainNetwork.set_anatomical_node_attributes`
        '''
        self.anatomical_graph_attributes = names


class GraphBundle(dict):
    '''
    GraphBundle is a subclass of :class:`dict` containing
    :class:`str`: :class:`BrainNetwork` pairs.

    Parameters
    ----------
    graph_list : list of :class:`networkx.Graph`
    name_list : list of str

    See Also
    --------
    :class:`BrainNetwork`

    Example
    -------
    '''
    def __init__(self, graph_list, name_list):
        dict.__init__(self)
        self.add_graphs(graph_list, name_list)

    def add_graphs(self, graph_list, name_list=None):
        '''
        Update dictionary with `graph_list : names_list` pairs.

        Parameters
        ----------
        graph_list : list of :class:`networkx.Graph`
        name_list : list of str, optional

        See Also
        --------
        :class:`GraphBundle.create_random_graphs`
        '''
        if name_list is None:
            name_list = [len(self) + i for i in range(len(graph_list))]
        elif len(name_list) != len(graph_list):
            raise IndexError("name_list and graph_list must have equal length")
        for graph in graph_list:
            if not isinstance(graph, BrainNetwork):
                graph = BrainNetwork(graph)
        self.update({a: b for a, b in zip(name_list, graph_list)})

    def apply(self, graph_function):
        '''
        FILL
        '''
        global_dict = {}
        for name, graph in self.items():
            global_dict[name] = graph_function(graph)
        return global_dict

    def report_global_measures(self, as_dict=False, partition=True):
        '''
        Calculate global_measures for each BrainNetwork object and report as a
        :class:`pandas.DataFrame` or nested dict.

        Parameters
        ----------
        as_dict : bool
            pass True to return global measures as a nested dictionary;
            pass False to return a :class:`pandas.DataFrame`
        partition: bool
            argument to pass to :func:`BrainNetwork.calculate_global_measures`

        Return
        ------
        :class:`pandas.DataFrame` or dict

        See Also
        --------
        :func:`BrainNetwork.calculate_global_measures`
        '''
        self.apply(lambda x: x.calculate_global_measures())
        global_dict = self.apply(lambda x: x.graph['global_measures'])
        if as_dict:
            return global_dict
        else:
            return pd.DataFrame.from_dict(global_dict).transpose()

    def report_rich_club(self, as_dict=False):
        '''
        Calculate rich_club coefficients for each BrainNetwork object and
        report as a :class:`pandas.DataFrame` or nested dict.

        Parameters
        ----------
        as_dict : bool
            pass True to return rich club coefficients as a nested dictionary;
            pass False to return a :class:`pandas.DataFrame`

        Return
        ------
        :class:`pandas.DataFrame` or dict

        See Also
        --------
        :func:`BrainNetwork.rich_club`
        '''
        rc_dict = self.apply(lambda x: x.rich_club())
        if as_dict:
            return rc_dict
        else:
            return pd.DataFrame.from_dict(rc_dict)

    def create_random_graphs(self, gname, n, Q=10, name_list=None, rname="_R"):
        '''
        Create `n` edge swap randomisations of :class:`BrainNetwork` keyed by
        `gname`. These random graphs are added to GraphBundle.

        Parameters
        ----------
        gname : str
            indexes a graph in GraphBundle
        n : int
            the number of random graphs to create
        Q : int, optional
            constant to specify how many swaps to conduct for each edge in G
        name_list : list of str, optional
            a list of names to use for indexing the new random graphs in
            GraphBundle.
        rname : str, optional
            if ``name_list=None`` the new random graphs will be indexed
            according to the scheme ``gname + rname + r`` where `r` is some
            integer.

        See Also
        --------
        :func:`get_random_graphs`
        :func:`random_graph`
        :func:`BrainNetwork.add_graphs`
        '''
        if name_list is None:
            # Choose r to be the smallest integer that is larger than all
            # integers already naming a random graph in brainnetwork
            r = len(self)
            while (gname + rname + str(r) not in self) and (r >= 0):
                r -= 1
            name_list = [gname + rname + str(i)
                         for i in range(r+1, r+1+n)]
        self.add_graphs(
            get_random_graphs(self[gname], n=n), name_list=name_list)

    def report_small_world(self, gname):
        '''
        Calculate the small coefficient of `gname` relative to each other
        graph in GraphBundle.

        Parameters
        ----------
        gname : str
            indexes a graph in GraphBundle

        Returns
        -------
        dict
            a dictionary, mapping a GraphBundle key "x" to the small
            coefficient of graph "gname" relative to graph "x".

        See Also
        --------
        :func:`small_coefficient`
        '''
        small_world_dict = self.apply(
            lambda x: small_coefficient(self[gname], x))
        return small_world_dict

    def nodal_matches(self):
        '''
        Checks the statement "All graphs in GraphBundle have congruent
        vertex sets"

        Returns
        -------
        bool
            `True` if all graphs have the same node set, `False` otherwise.

        See Also
        --------
        :func:`is_nodal_match`
        :func:`BrainNetwork.anatomical_matches`
        '''
        H = list(self.values())[0]
        return (False not in [is_nodal_match(H, G) for G in self.values()])

    def anatomical_matches(self,
                           nodal_keys=anatomical_node_attributes(),
                           graph_keys=anatomical_graph_attributes()):
        '''
        Checks that all graphs in GraphBundle are pairwise anatomical matches
        as defined in :func:`is_anatomical_match`.

        Parameters
        ----------
        nodal_keys : list of str, optional
        graph_keys : list of str, optional

        Returns
        -------
        bool
            `True` if all graphs are anatomically matched, `False` otherwise.

        See Also
        --------
        :func:`is_anatomical_match`
        :func:`BrainNetwork.is_nodal_match`
        '''
        H = list(self.values())[0]
        return (False not in
                [is_anatomical_match(
                    H,
                    G,
                    nodal_keys=nodal_keys,
                    graph_keys=graph_keys)
                 for G in self.values()])
