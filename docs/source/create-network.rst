Creating a Network
==================

Weighted and Binary Graphs
--------------------------

Minimum Spanning Tree
---------------------
- why a connected graph?



The BrainNetwork Class
----------------------

This is a very lightweight subclass of the :class:`networkx.Graph` class. This means that any methods you can use on a :class:`networkx.Graph` object can also be used on a `BrainNetwork` object, although the reverse is not true. We have **added** various methods that allow us to keep track of measures that have already been calculated. This is particularly useful later on when one is dealing with 1000 random graphs (or more!) and saves a lot of time.

All :class:`scona.BrainNetwork` **methods** have a corresponding ``scona`` **function**. So while the :class:`scona.BrainNetwork` methods can only be applied to :class:`scona.BrainNetwork` objects, you can find the equivalent function in ``scona`` which can be used on a regular :class:`networkx.Graph` object.

A :class:`BrainNetwork` can be initialised from a :class:`networkx.Graph` or from a correlation matrix represented as a :class:`pandas.DataFrame` or :class:`numpy.array`.

Threshold
------------
if ``G`` is a :class:`scona.BrainNetwork` object, you can threshold it by typing ``G.threshold(10)``. If ``nxG`` is a :class:`Networkx.Graph` you can use ``scona.threshold_graph(nxG, 10)`` to perform the same function.

This function creates a binary graph by thresholding weighted graph ``G``.

First creates a spanning forest that is a union of the spanning trees for each connected component of the graph.

Then adds in edges according to their connection strength up to cost.


The GraphBundle Class
----------------------

This is is a subclass of :class:`dict` containing :class:`str`: :class:`BrainNetwork` pairs.


Resources
---------