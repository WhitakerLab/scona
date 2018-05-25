import pytest
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
def graph_1():
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
    G2 = graph_1()
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_weighted_graph_from_df_isomorphic():
    G1 = mkg.weighted_graph_from_df(symmetric_df_1())
    G2 = graph_1()
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_scale_weights():
    G1 = mkg.scale_weights(graph_1())
    G2 = mkg.weighted_graph_from_matrix(symmetric_matrix_1()*-1)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_threshold_graph():
    with pytest.raises(Exception):
        mkg.threshold_graph(graph_1(), 30)


def test_threshold_graph_mst_false():
    G1 = mkg.threshold_graph(graph_1(), 30, mst=False)
    G2 = graph_1()
    G2.remove_edge(1, 2)
    G2.remove_edge(1, 0)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_threshold_graph_mst_true():
    G1 = mkg.threshold_graph(graph_1(), 70, mst=True)
    G2 = graph_1()
    G2.remove_edge(1, 2)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_graph_at_cost_df():
    G1 = mkg.graph_at_cost(symmetric_df_1(), 70)
    G2 = mkg.threshold_graph(graph_1(), 70)
    assert nx.is_isomorphic(G1, G2, edge_match=em())


def test_graph_at_cost_array():
    G1 = mkg.graph_at_cost(symmetric_matrix_1(), 70)
    G2 = mkg.threshold_graph(graph_1(), 70)
    assert nx.is_isomorphic(G1, G2, edge_match=em())
