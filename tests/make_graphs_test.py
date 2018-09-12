import pytest
import unittest
import pandas as pd
import networkx as nx
import numpy as np
import scona.make_graphs as mkg


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
def simple_anatomical_graph():
    G = simple_weighted_graph()
    mkg.assign_node_centroids(G,
                              [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    mkg.assign_node_names(G, ['a', 'b', 'c'])
    return G


@pytest.fixture
def em():
    return nx.algorithms.isomorphism.numerical_edge_match('weight', 1)


@pytest.fixture
def nm(exclude=[]):
    nm = ["name", "name_34", "name_68", "hemi",
          "centroids", "x", "y", "z",
          "hats", "socks"]
    return nx.algorithms.isomorphism.categorical_node_match(
        [x for x in nm if x not in exclude],
        [None for i in range(len([x for x in nm if x not in exclude]))])


class AnatCopying(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = simple_weighted_graph()

        cls.Gname = simple_weighted_graph()
        nx.set_node_attributes(cls.Gname,
                               {0: 'a', 1: 'b', 2: 'c'},
                               name='name')

        cls.Gcentroids = simple_weighted_graph()
        nx.set_node_attributes(cls.Gcentroids,
                               {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)},
                               name='centroids')
        nx.set_node_attributes(cls.Gcentroids,
                               {0: '1', 1: '0', 2: '0'},
                               name='x')
        nx.set_node_attributes(cls.Gcentroids,
                               {0: '0', 1: '1', 2: '0'},
                               name='y')
        nx.set_node_attributes(cls.Gcentroids,
                               {0: '0', 1: '0', 2: '1'},
                               name='z')

        cls.H = simple_anatomical_graph()

        cls.J = nx.Graph()
        cls.J.add_nodes_from(cls.H.nodes)
        mkg.copy_anatomical_data(cls.J, cls.H)
        cls.K = simple_anatomical_graph()
        cls.K.remove_edges_from(cls.H.edges)
        cls.L = mkg.anatomical_copy(cls.H)
        cls.R = mkg.anatomical_copy(cls.H)
        nx.set_node_attributes(cls.R, 'stetson', name='hats')

    def test_assign_name_to_nodes(self):
        G1 = simple_weighted_graph()
        mkg.assign_node_names(G1,
                              ['a', 'b', 'c'])
        assert nx.is_isomorphic(G1, self.Gname,
                                edge_match=em(),
                                node_match=nm(
                                    exclude=['centroids', 'y', 'x', 'z']))

    def test_assign_name_to_nodes_308_style(self):
        return

    def test_assign_node_centroids(self):
        G1 = simple_weighted_graph()
        mkg.assign_node_centroids(G1,
                                  [(1, 0, 0), (0, 1, 0), (0, 0, 1)])

        assert nx.is_isomorphic(
            G1,
            self.Gcentroids,
            edge_match=em(),
            node_match=nm(exclude=['name', "name_34", "name_68", "hemi"]))

    def test_copy_anatomical_data(self):

        G2 = nx.Graph()
        mkg.copy_anatomical_data(G2, self.H)
        assert nx.is_isomorphic(G2, nx.Graph(),
                                edge_match=em(),
                                node_match=nm())

        assert nx.is_isomorphic(self.J, self.K,
                                edge_match=em(),
                                node_match=nm())

        G4 = simple_weighted_graph()
        mkg.copy_anatomical_data(G4, self.H)
        assert nx.is_isomorphic(G4, self.H,
                                edge_match=em(),
                                node_match=nm())

    def no_data_does_nothing(self):
        mkg.copy_anatomical_data(self.H, self.G)
        assert nx.is_isomorphic(self.H, simple_anatomical_graph(),
                                edge_match=em(),
                                node_match=nm())

    def copy_different_anatomical_keys(self):
        G5 = simple_weighted_graph()
        G6 = simple_weighted_graph()
        nx.set_node_attributes(G6, 'bowler', name='hats')
        mkg.copy_anatomical_data(G5, G6)
        assert nx.is_isomorphic(G5, simple_weighted_graph(),
                                edge_match=em(),
                                node_match=nm())

        mkg.copy_anatomical_data(
            G5, G6, nodal_keys=['hats', 'socks'])
        assert nx.is_isomorphic(G5, G6,
                                edge_match=em(),
                                node_match=nm())

    def test_anatomical_copy(self):
        assert nx.is_isomorphic(self.H, self.L,
                                edge_match=em(),
                                node_match=nm())

    def test_anatomical_copy_hats(self):
        # check hats not copied to P
        P = mkg.anatomical_copy(self.R)
        assert nx.is_isomorphic(self.H, P,
                                edge_match=em(),
                                node_match=nm())
        # check otherwise the same
        assert nx.is_isomorphic(self.R, P,
                                edge_match=em(),
                                node_match=nm(exclude=['hats']))
        # check hats copied if specified as an additional key
        new_keys = mkg.anatomical_node_attributes()
        new_keys.append('hats')
        Q = mkg.anatomical_copy(self.R, nodal_keys=new_keys)
        assert nx.is_isomorphic(self.R, Q,
                                edge_match=em(),
                                node_match=nm())

    def test_matchers(self):
        N = nx.Graph()
        N.add_nodes_from({0, 1, 2})
        assert mkg.is_nodal_match(self.G, N)
        assert mkg.is_nodal_match(self.H, N)

    def test_different_vertex_sets(self):
        M = nx.Graph()
        S = nx.karate_club_graph()
        assert not mkg.is_nodal_match(self.G, M)
        assert not mkg.is_nodal_match(self.G, S)
        assert not mkg.is_nodal_match(S, M)

    def test_key_matchings(self):
        assert not mkg.is_nodal_match(self.G, self.H, keys=['x'])
        assert mkg.is_nodal_match(self.R, self.H, keys=['x'])
        assert not mkg.is_nodal_match(self.R, self.H, keys=['x', 'hats'])

    def check_anatomical_matches(self):
        assert mkg.is_anatomical_match(self.L, self.H)
        assert not mkg.is_anatomical_match(self.G, self.H)

    def check_anat_matches_with_hats(self):
        assert mkg.is_anatomical_match(self.H, self.R)


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


def test_threshold_graph_mst_too_large_exception():
    with pytest.raises(Exception):
        mkg.threshold_graph(simple_weighted_graph(), 30)


def test_threshold_graph_mst_false():
    G1 = mkg.threshold_graph(simple_anatomical_graph(), 30, mst=False)
    G2 = nx.Graph()
    G2.add_nodes_from(G1.nodes)
    G2._node = simple_anatomical_graph()._node
    G2.add_edge(0, 2)
    assert nx.is_isomorphic(G1, G2, edge_match=em(), node_match=nm())


def test_threshold_graph_mst_true():
    G1 = mkg.threshold_graph(simple_anatomical_graph(), 70, mst=True)
    G2 = nx.Graph()
    G2.add_nodes_from(G1.nodes)
    G2._node = simple_anatomical_graph()._node
    G2.add_edge(0, 2)
    G2.add_edge(0, 1)
    assert nx.is_isomorphic(G1, G2, edge_match=em(), node_match=nm())


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
        cls.karate_club_graph = nx.karate_club_graph()
        cls.karate_club_graph_rand = mkg.random_graph(cls.karate_club_graph)
        cls.karate_club_graph_list = mkg.get_random_graphs(cls.karate_club_graph, n=5)
        cls.lattice = nx.grid_graph(dim=[5, 5, 5], periodic=True)
        cls.lattice_rand = mkg.random_graph(cls.lattice)
        cls.lattice_list = mkg.get_random_graphs(cls.lattice, n=5)
        G = nx.Graph()
        G.add_path([0, 1, 2])
        cls.short_path = G

    def test_random_graph_type(self):
        self.assertTrue(isinstance(self.karate_club_graph_rand, nx.Graph))
        self.assertTrue(isinstance(self.lattice_rand, nx.Graph))

    def test_random_graph_connected(self):
        self.assertTrue(nx.is_connected(self.karate_club_graph_rand))
        self.assertTrue(nx.is_connected(self.lattice_rand))

    def test_random_graph_edges(self):
        self.assertEqual(self.karate_club_graph_rand.size(), self.karate_club_graph.size())
        self.assertEqual(self.lattice_rand.size(), self.lattice.size())

    def test_random_graph_degree_distribution(self):
        self.assertEqual(list(self.karate_club_graph_rand.degree()),
                         list(self.karate_club_graph.degree()))
        self.assertEqual(list(self.lattice_rand.degree()),
                         list(self.lattice.degree()))

    def test_random_graph_makes_changes(self):
        self.assertNotEqual(self.karate_club_graph, self.karate_club_graph_rand)
        self.assertNotEqual(self.lattice, self.lattice_rand)

    def test_random_graph_fail_short_path(self):
        with self.assertRaises(Exception):
            mkg.random_graph(self.short_path)

    def test_get_random_graphs(self):
        from itertools import combinations
        self.assertEqual(len(self.karate_club_graph_list), 5)
        self.assertEqual(len(self.lattice_list), 5)
        k = self.karate_club_graph_list
        k.append(self.karate_club_graph)
        for a, b in combinations(k, 2):
            self.assertNotEqual(a, b)
        L = self.lattice_list
        L.append(self.lattice)
        for a, b in combinations(L, 2):
            self.assertNotEqual(a, b)
