import numpy as np
import networkx as nx
import pandas as pd
import make_graphs as mkg
import graph_measures as gm


def cascader(dict1, dict2, name):
    return {key: value.update({name: dict2[key]})
            for key, value in dict1.items()}


class BrainNetwork(nx.classes.graph.Graph):
    def __init__(self,
                 network,
                 parcellation,
                 centroids,
                 names_308_style=False):
        if isinstance(network, nx.classes.graph.Graph):
            # Copy graph
            nx.classes.graph.Graph.__init__(self)
            self.__dict__.update(network.__dict__)
        else:
            # Create weighted graph
            if isinstance(network, pd.DataFrame):
                M = network.values
            elif isinstance(network, np.ndarray):
                M = network
            M[np.diag_indices_from(M)] = 0
            nx.classes.graph.Graph.__init__(self, M)

        # assign names and centroids to nodes
        mkg.assign_node_names(self,
                              parcellation,
                              names_308_style=names_308_style)
        mkg.assign_node_centroids(self, centroids)


class BinaryBrainNetwork(nx.classes.graph.Graph):
    def __init__(self, brainnetwork, cost, mst=True):

        nx.classes.graph.Graph.__init__(self)
        self.__dict__.update(brainnetwork.__dict__)

        self = mkg.threshold_graph(self, self.cost, mst=mst)
        self.graph['cost'] = cost
        self.graph['mst'] = mst

    def partition(self):
        nodal_partition, module_partition = gm.calc_nodal_partition(self)
        cascader(self._node, nodal_partition, 'module')
        self.graph['partition'] = module_partition

    def calculate_nodal_measures(self):
        '''
        Calculates
        '''

        # ==== SET UP ======================
        # If you haven't passed the nodal partition
        # then calculate it here
        if 'partition' not in self.graph:
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
        # ---- interhem proporition --------
        gm.assign_nodal_distance(self)
        gm.assign_interhem(self)

    def export_nodal_measures(self):
        '''
        Returns the node attribute data from G as a pandas dataframe.
        '''
        return pd.DataFrame.from_dict(self._node).transpose()
