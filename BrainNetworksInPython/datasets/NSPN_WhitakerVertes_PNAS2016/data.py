#! /usr/bin/env python

"""NSPN Whitaker, Vertes et al, 2016"""

TITLE       = """Cortical Thickness Measures"""
SOURCE      = """
Adolescent consolidation of human connectome hubs
Kirstie J. Whitaker, Petra E. Vértes, Rafael Romero-Garcia, František Váša, Michael Moutoussis, Gita Prabhu, Nikolaus Weiskopf, Martina F. Callaghan, Konrad Wagstyl, Timothy Rittman, Roger Tait, Cinly Ooi, John Suckling, Becky Inkster, Peter Fonagy, Raymond J. Dolan, Peter B. Jones, Ian M. Goodyer, the NSPN Consortium, Edward T. Bullmore
Proceedings of the National Academy of Sciences Aug 2016, 113 (32) 9105-9110; DOI: 10.1073/pnas.1601745113
"""
DESCRSHORT  = """Cortical thickness data"""
DESCRLONG   = """  """


from os.path import dirname, abspath
filepath = dirname(abspath(__file__))
centroids_file = filepath + "/500.centroids.txt"         
names_file = filepath + "/500.names.txt"                   
regionalmeasures_file = filepath + "/PARC_500aparc_thickness_behavmerge.csv"
covars_file=None
names_308_style=True

def _get_data():
    return (centroids_file, regionalmeasures_file, names_file, covars_file, names_308_style)
    
def _get_centroids():
    return centroids_file
    
def _get_regionalmeasures():
    return regionalmeasures_file

def _get_names():
    return names_file
    
def _get_covars():
    return covars_file
    
def _is_names_308_style():
    return names_308_style
    
