import pytest
import unittest
import pandas as pd
import networkx as nx
import numpy as np
import BrainNetworksInPython.make_graphs as mkg
import BrainNetworksInPython.graph_measures as gm


class Partitioning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.karate = nx.karate_club_graph()

        self.totalpart = {x: [x] for x in list(self.karate.nodes)}
        self.triviallpart = {0: [x for x in list(self.karate.nodes)]}
        self.nonpart = {0: [0]}
        n, m = gm.calc_nodal_partition(self.karate)
        self.bestpart_n = n
        self.bestpart_m = m

        self.nonbinary = nx.karate_club_graph()
        nx.set_edge_attributes(self.nonbinary, '0.5', name='weight')

    def throw_out_nonbinary_graph(self):
        with self.assertRaises(Exception):
            gm.calc_nodal_partition(self.nonbinary)

    def check_n_m_consistency(self):
        return

    def test_total_partition_pc(self):
        pc = gm.participation_coefficient(self.karate, self.totalpart)
        for x in pc.values():
            assert x == 1

    def test_total_partition_zs(self):
        zs = gm.z_score(self.karate, self.totalpart)
        for x in zs.values():
            assert x == 0

    def test_trivial_partition_pc(self):
        pc = gm.participation_coefficient(self.karate, self.triviallpart)
        for x in pc.values():
            assert x == 0

    def test_trivial_partition_zs(self):
        zs = gm.z_score(self.karate, self.triviallpart)
        karate_degrees = list(dict(self.karate.degree()).values())
        karate_degree = np.mean(karate_degrees)
        karate_std = np.std(karate_degrees)
        for node, score in zs.items():
            assert score == (self.karate.degree(node)
                             - karate_degree)/karate_std

    def test_non_partition_pc(self):
        pc = gm.participation_coefficient(self.karate, self.nonpart)
        assert pc == {0: 1}


def shortest_path_test():
    G = nx.complete_graph(6)
    sp = gm.shortest_path(G)
    assert sp == {x: 1 for x in G.nodes}


class AnatomicalMeasures(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.no_centroids = nx.karate_club_graph()
        self.identical_centroids = nx.karate_club_graph()
        mkg.assign_node_centroids(
            self.identical_centroids,
            [(-1, 0, 0) for x in self.no_centroids.nodes])
        self.opposing_centroids = nx.complete_graph(6)
        mkg.assign_node_centroids(
            self.opposing_centroids,
            [((-1)**i, 0, 0) for i in self.no_centroids.nodes])

    def test_no_centroids_assign_distance(self):
        with self.assertRaises(Exception):
            gm.assign_nodal_distance(self.no_centroids)

    def test_no_centroids_assign_interhem(self):
        with self.assertRaises(Exception):
            gm.assign_interhem(self.no_centroids)

    def test_identical_centroids_assign_distance(self):
        gm.assign_nodal_distance(self.identical_centroids)
        assert (nx.get_edge_attributes(self.identical_centroids, 'euclidean')
                == {edge: 0 for edge in self.no_centroids.edges})
        assert (nx.get_node_attributes(self.identical_centroids, 'average_dist')
                == {node: 0 for node in self.no_centroids.nodes})
        assert (nx.get_node_attributes(self.identical_centroids, 'total_dist')
                == {node: 0 for node in self.no_centroids.nodes})

    def test_identical_centroids_assign_interhem(self):
        gm.assign_interhem(self.identical_centroids)
        assert (nx.get_edge_attributes(self.identical_centroids, 'interhem')
                == {edge: 0 for edge in self.no_centroids.edges})
        assert (nx.get_node_attributes(self.identical_centroids, 'interhem')
                == {node: 0 for node in self.no_centroids.nodes})
        assert (nx.get_node_attributes(self.identical_centroids, 'interhem_proportion')
                == {node: 0 for node in self.no_centroids.nodes})

    def test_opposing_centroids_assign_distance(self):
        gm.assign_nodal_distance(self.opposing_centroids)
        assert (nx.get_edge_attributes(self.opposing_centroids, 'euclidean')
                == {edge: (1+(-1)**(sum(edge)+1))
                for edge in self.opposing_centroids.edges})
        assert (nx.get_node_attributes(self.opposing_centroids, 'average_dist')
                == {node: 1.2 for node in self.opposing_centroids.nodes})
        assert (nx.get_node_attributes(self.opposing_centroids, 'total_dist')
                == {node: 6 for node in self.opposing_centroids.nodes})

    def test_opposing_centroids_assign_interhem(self):
        gm.assign_interhem(self.opposing_centroids)
        assert (nx.get_edge_attributes(self.opposing_centroids, 'interhem')
                == {edge: (1+(-1)**(sum(edge)+1))//2
                for edge in self.opposing_centroids.edges})
        assert (nx.get_node_attributes(self.opposing_centroids, 'interhem')
                == {node: 3 for node in self.opposing_centroids.nodes})
        assert (nx.get_node_attributes(self.opposing_centroids, 'interhem_proportion')
                == {node: 0.6 for node in self.opposing_centroids.nodes})


# omit testing of calc_modularity or rich_club since these
# are relabeled networkx measures


def test_small_worlds():
    return


def test_calculate_global_measures():
    return
