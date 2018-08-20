#! /usr/bin/env python

from os.path import dirname, abspath
from BrainNetworksInPython.scripts.useful_functions import read_in_data

"""NSPN Whitaker, Vertes et al, 2016"""

TITLE       = """Cortical Thickness Measures"""
SOURCE      = """
Adolescent consolidation of human connectome hubs
Kirstie J. Whitaker, Petra E. Vértes, Rafael Romero-Garcia, František Váša, Michael Moutoussis, Gita Prabhu, Nikolaus Weiskopf, Martina F. Callaghan, Konrad Wagstyl, Timothy Rittman, Roger Tait, Cinly Ooi, John Suckling, Becky Inkster, Peter Fonagy, Raymond J. Dolan, Peter B. Jones, Ian M. Goodyer, the NSPN Consortium, Edward T. Bullmore
Proceedings of the National Academy of Sciences Aug 2016, 113 (32) 9105-9110; DOI: 10.1073/pnas.1601745113
"""
DESCRSHORT  = """Cortical thickness data"""
DESCRLONG   = """  """

filepath = dirname(abspath(__file__))
centroids_file = filepath + "/500.centroids.txt"
names_file = filepath + "/500.names.txt"
regionalmeasures_file = filepath + "/PARC_500aparc_thickness_behavmerge.csv"
covars_file = None


def _data():
    return (regionalmeasures_file,
            names_file,
            covars_file,
            centroids_file)


def _centroids():
    return centroids_file


def _regionalmeasures():
    return regionalmeasures_file


def _names():
    return names_file


def _covars():
    return covars_file


def import_data():
    return read_in_data(
        regionalmeasures_file,
        names_file,
        covars_file=covars_file,
        centroids_file=centroids_file,
        data_as_df=True)
