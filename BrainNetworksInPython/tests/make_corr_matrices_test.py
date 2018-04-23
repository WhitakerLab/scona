import os
import sys
sys.path.append(os.path.abspath(os.path.join('wrappers')))
sys.path.append(os.path.abspath(os.path.join('example_data')))
sys.path.append(os.path.abspath(os.path.join('scripts')))

import BrainNetworksInPython.make_corr_matrices as mcm
import BrainNetworksInPython.stats_functions as sf
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def participant_array():
    return np.array([np.arange(x,x+4) for x in [7,4,1]])

@pytest.fixture
def participant_data():
    columns = ['noggin_left', 'noggin_right', 'day of week', 'haircut']
    data = participant_array()
    return pd.DataFrame(data, columns=columns)

@pytest.fixture
def participant_residuals():
    columns = ['noggin_left', 'noggin_right', 'day of week', 'haircut']
    data = np.array([[1,0,-1],[0,-5,5], [1,2,3], [300,200,100]]).T
    return pd.DataFrame(data, columns=columns)

def test_create_residuals_df_covars_plural():
    names, covars = ['noggin_left', 'noggin_right'], ['day of week', 'haircut']
    array_resids = [sf.residuals(participant_array()[:,2:].T,participant_array()[:,i]) for i in [0,1]]
    np.testing.assert_almost_equal(np.array(mcm.create_residuals_df(participant_data(), names, covars)[names]), np.array(array_resids).T)
    
def test_create_residuals_df_covars_singular():
    names, covars = ['noggin_left', 'noggin_right'], ['day of week']
    array_resids = [sf.residuals(participant_array()[:,2:3].T,participant_array()[:,i]) for i in [0,1]]
    np.testing.assert_almost_equal(np.array(mcm.create_residuals_df(participant_data(), names, covars)[names]), np.array(array_resids).T)
    
def test_create_residuals_df_covars_none():
    names, covars = ['noggin_left', 'noggin_right'], []
    array_resids = [sf.residuals(participant_array()[:,2:2].T,participant_array()[:,i]) for i in [0,1]]
    np.testing.assert_almost_equal(np.array(mcm.create_residuals_df(participant_data(), names, covars)[names]), np.array(array_resids).T)
