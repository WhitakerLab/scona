import pytest
import unittest
import pandas as pd
import networkx as nx
import numpy as np
import BrainNetworksInPython.scripts.make_graphs as mkg


@pytest.fixture
def symmetric_matrix_1():
    return np.array([[1+x, 2+x, 1+x] for x in [1, 4, 1]])


@pytest.fixture
def symmetric_df_1():
    return pd.DataFrame(symmetric_matrix_1(), index=['a', 'b', 'c'])


@pytest.fixture
def simple_weighted_graph():
    G = nx.Graph()
    G.add_path([1, 2], weight=2)
    G.add_path([1, 0], weight=3)
    G.add_path([2, 0], weight=5)
    return G


@pytest.fixture
def em():
    return nx.algorithms.isomorphism.numerical_edge_match('weight', 1)


def test_weighted_graph_from_matrix_isomorphic():
    G1 = mkg.weighted_graph_from_matrix(symmetric_matrix_1())
    G2 = simple_weighted_graph()
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_weighted_graph_from_df_isomorphic():
    G1 = mkg.weighted_graph_from_df(symmetric_df_1())
    G2 = simple_weighted_graph()
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_scale_weights():
    G1 = mkg.scale_weights(simple_weighted_graph())
    G2 = mkg.weighted_graph_from_matrix(symmetric_matrix_1()*-1)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_threshold_graph():
    with pytest.raises(Exception):
        mkg.threshold_graph(simple_weighted_graph(), 30)


def test_threshold_graph_mst_false():
    G1 = mkg.threshold_graph(simple_weighted_graph(), 30, mst=False)
    G2 = simple_weighted_graph()
    G2.remove_edge(1, 2)
    G2.remove_edge(1, 0)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_threshold_graph_mst_true():
    G1 = mkg.threshold_graph(simple_weighted_graph(), 70, mst=True)
    G2 = simple_weighted_graph()
    G2.remove_edge(1, 2)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_graph_at_cost_df():
    G1 = mkg.graph_at_cost(symmetric_df_1(), 70)
    G2 = mkg.threshold_graph(simple_weighted_graph(), 70)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_graph_at_cost_array():
    G1 = mkg.graph_at_cost(symmetric_matrix_1(), 70)
    G2 = mkg.threshold_graph(simple_weighted_graph(), 70)
    assert nx.is_isomorphic(G1, G2, edge_match=em())

# Test random graph generation
# This is non-deterministic, so we will check that the random graphs have
# the properties we would like them to have. Namely, they:
# * should be connected binary graphs
# * they should not be equal to the original graphs
# * The number of edges should be constant
# * the degree distribution should be contant
# * should fail on particular graphs

class RandomGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.karate_club = nx.karate_club_graph()
        cls.karate_club_rand = mkg.random_graph(cls.karate_club)
        cls.lattice = nx.grid_graph(dim=[5, 5, 5], periodic=True)
        cls.lattice_rand = mkg.random_graph(cls.lattice)
        G = nx.Graph()
        G.add_path([0, 1, 2])
        cls.short_path = G

    def test_random_graph_type(self):
        self.assertTrue(isinstance(self.karate_club_rand, nx.Graph))
        self.assertTrue(isinstance(self.lattice_rand, nx.Graph))

    def test_random_graph_connected(self):
        self.assertTrue(nx.is_connected(self.karate_club_rand))
        self.assertTrue(nx.is_connected(self.lattice_rand))

    def test_random_graph_edges(self):
        self.assertEqual(self.karate_club_rand.size(), self.karate_club.size())
        self.assertEqual(self.lattice_rand.size(), self.lattice.size())

    def test_random_graph_degree_distribution(self):
        self.assertEqual(list(self.karate_club_rand.degree()),
                         list(self.karate_club.degree()))
        self.assertEqual(list(self.lattice_rand.degree()),
                         list(self.lattice.degree()))

    def test_random_graph_makes_changes(self):
        self.assertNotEqual(self.karate_club, self.karate_club_rand)
        self.assertNotEqual(self.lattice, self.lattice_rand)

    def test_random_graph_fail_short_path(self):
        with self.assertRaises(Exception):
            mkg.random_graph(self.short_path)
