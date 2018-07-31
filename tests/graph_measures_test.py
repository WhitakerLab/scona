import pytest
import unittest
import pandas as pd
import networkx as nx
import numpy as np
import BrainNetworksInPython.scripts.make_graphs as mkg
import BrainNetworksInPython.scripts.graph_measures as gm

@pytest.fixture



def test_nodal_partition_throw_out_non_binary():
    
