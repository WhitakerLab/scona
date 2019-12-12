"""
scona
=====
scona is a Python package for the analysis
of structural covariance brain networks.

Website (including documentation)::
    http://whitakerlab.github.io/scona
Source::
    https://github.com/WhitakerLab/scona
Bug reports::
    https://github.com/WhitakerLab/scona/issues

Simple example
--------------
    FILL

Bugs
----
Please report any bugs that you find in an issue on `our GitHub repository <https://github.com/WhitakerLab/scona/issues>`_.
Or, even better, fork the repository and create a pull request.

License
-------
`scona  is licensed under the MIT License <https://github.com/WhitakerLab/scona/blob/master/LICENSE>`_.
"""

# Release data
__author__ = "Kirstie Whitaker and Isla Staden"
__license__ = "MIT"

__date__ = ""
__version__ = 0.1

__bibtex__ = """ FILL 
"""

from scona.make_corr_matrices import *
from scona.make_graphs import *
from scona.graph_measures import *
from scona.classes import *
from scona.analyses import *

from scona.wrappers import *

from scona.visualisations_helpers import *

import scona.datasets
from scona import *

