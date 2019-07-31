from scona.make_corr_matrices import create_residuals_df, \
    get_non_numeric_cols, create_corrmat
from scona.stats_functions import residuals
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def subject_array():
    return np.array([np.arange(x, x+4) for x in [7, 4, 1]])


@pytest.fixture
def subject_data(subject_array):
    columns = ['noggin_left', 'noggin_right', 'day of week', 'haircut']
    data = subject_array
    return pd.DataFrame(data, columns=columns)


@pytest.fixture
def subject_residuals():
    columns = ['noggin_left', 'noggin_right', 'day of week', 'haircut']
    data = np.array([[1, 0, -1], [0, -5, 5], [1, 2, 3], [300, 200, 100]]).T
    return pd.DataFrame(data, columns=columns)


def test_non_numeric_cols(subject_residuals):
    df = subject_residuals
    assert get_non_numeric_cols(df).size == 0
    df['hats'] = 'stetson'
    assert get_non_numeric_cols(df) == np.array(['hats'])


def test_create_residuals_df_covars_plural(subject_array, subject_data):
    names, covars = ['noggin_left', 'noggin_right'], ['day of week', 'haircut']
    array_resids = [residuals(subject_array[:, 2:].T,
                    subject_array[:, i]) for i in [0, 1]]
    np.testing.assert_almost_equal(
        np.array(create_residuals_df(subject_data, names, covars)[names]),
        np.array(array_resids).T)


def test_create_residuals_df_covars_singular(subject_array, subject_data):
    names, covars = ['noggin_left', 'noggin_right'], ['day of week']
    array_resids = [residuals(subject_array[:, 2:3].T,
                    subject_array[:, i]) for i in [0, 1]]
    np.testing.assert_almost_equal(
        np.array(create_residuals_df(subject_data, names, covars)[names]),
        np.array(array_resids).T)


def test_create_residuals_df_covars_none(subject_array, subject_data):
    names, covars = ['noggin_left', 'noggin_right'], []
    array_resids = [residuals(subject_array[:, 2:2].T, subject_array[:, i])
                    for i in [0, 1]]
    np.testing.assert_almost_equal(
        np.array(create_residuals_df(subject_data, names, covars)[names]),
        np.array(array_resids).T)


def test_create_corrmat_pearson(subject_residuals):
    df_res = subject_residuals
    names = ['noggin_left', 'noggin_right']
    np.testing.assert_almost_equal(
        np.array(create_corrmat(df_res, names)),
        np.array([[1, -0.5], [-0.5, 1]]))


def test_create_corrmat_spearman(subject_residuals):
    df_res = subject_residuals
    names = ['noggin_left', 'noggin_right']
    np.testing.assert_almost_equal(
        np.array(create_corrmat(df_res, names, method='spearman')),
        np.array([[1, -0.5], [-0.5, 1]]))
