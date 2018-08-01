import numpy as np
import networkx as nx
import pandas as pd
import make_graphs as mkg
import graph_measures as gm


def cascader(dict1, dict2, name):
    return {key: value.update({name: dict2[key]})
            for key, value in dict1.items() if key in dict2}


class BrainNetwork(nx.classes.graph.Graph):
    def __init__(self,
                 network=None,
                 parcellation=None,
                 centroids=None,
                 names_308_style=False):
        '''
        Lightweight subclass of networkx.classes.graph.Graph
        '''
        # ============== Create graph ================
        if not network:
            # Create empty graph
            nx.classes.graph.Graph.__init__(self)

        elif isinstance(network, nx.classes.graph.Graph):
            # Copy graph
            nx.classes.graph.Graph.__init__(self)
            self.__dict__.update(network.__dict__)

        else:
            # Create weighted graph from a dataframe or
            # numpy array
            if isinstance(network, pd.DataFrame):
                M = network.values
            elif isinstance(network, np.ndarray):
                M = network
            M[np.diag_indices_from(M)] = 0
            nx.classes.graph.Graph.__init__(self, M)

        # ===== Give anatomical labels to nodes ======
        if parcellation:
            # assign parcellation names to nodes
            mkg.assign_node_names(self,
                                  parcellation,
                                  names_308_style=names_308_style)
        if centroids:
            # assign centroids to nodes
            mkg.assign_node_centroids(self, centroids)

    def partition(self):
        if 'partition' not in self.graph:
            nodal_partition, module_partition = gm.calc_nodal_partition(self)
            cascader(self._node, nodal_partition, 'module')
            self.graph['partition'] = module_partition
        return self.graph['partition']

    def calculate_nodal_measures(self):
        '''
        Calculates
        '''

        # ==== SET UP ======================
        # If you haven't passed the nodal partition
        # then calculate it here
        self.partition()

        # ==== MEASURES ====================
        # ---- Degree ----------------------
        cascader(self._node, dict(self.degree), 'degree')
        # ---- Closeness -------------------
        cascader(self._node, nx.closeness_centrality(self), 'closeness')
        # ---- Betweenness -----------------
        cascader(self._node, nx.betweenness_centrality(self), 'betweenness')
        # ---- Shortest path length --------
        cascader(self._node, gm.shortest_path(self), 'shortest_path')
        # ---- Clustering ------------------
        cascader(self._node, nx.clustering(self), 'clustering')
        # ---- Participation coefficent ----
        cascader(self._node, gm.participation_coefficient(self,
                 self.graph['partition']), 'pc')

        # ---- Euclidean distance and ------
        # ---- interhem proportion --------
        if self.graph.get('centroids'):
            gm.assign_nodal_distance(self)
            gm.assign_interhem(self)

    def export_nodal_measures(self):
        '''
        Returns the node attribute data from G as a pandas dataframe.
        '''
        return pd.DataFrame.from_dict(self._node).transpose()

    def calculate_rich_club(self, force=False):
        if ('rich_club' not in self.graph) or force:
            self.graph['rich_club'] = nx.rich_club_coefficient(
                                        self, normalized=False)
        return self.graph['rich_club']

    def calculate_global_measures(self, force=False):
        if ('global_measures' not in self.graph) or force:
            global_measures = gm.calculate_global_measures(self,
                                                           self.partition())
            self.graph['global_measures'] = global_measures
        return self.graph['global_measures']

    def anatomical_copy(self):
        '''
        '''
        mkg.anatomical_copy(self)

    def update_nodal_attributes(self, name, dictionary):
        '''
        '''
        cascader(self._node, dictionary, name)


class GraphBundle(dict):
    '''
    '''
    def __init__(self, name_list, graph_list):
        '''
        '''
        dict.__init__(self)
        for graph in graph_list:
            if not isinstance(graph, BrainNetwork):
                graph = BrainNetwork(graph)

    def apply(self, graph_function):
        '''
        '''
        global_dict = {}
        for name, graph in self.items():
            global_dict[name] = graph_function(graph)
        return global_dict

    def report_global_measures(self):
        '''
        '''
        global_dict = self.apply(lambda x: x.calculate_global_measures())
        return pd.DataFrame.from_dict(global_dict).transpose()

    def report_rich_club(self):
        '''
        '''
        rc_dict = self.apply(lambda x: x.calculate_rich_club())
        return pd.DataFrame.from_dict(rc_dict).transpose()
