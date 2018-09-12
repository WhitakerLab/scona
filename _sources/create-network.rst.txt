Creating a Network
==================

Weighted and Binary Graphs
--------------------------

Minimum Spanning Tree
---------------------
- why a connected graph?

Thresholding
------------


The BrainNetwork Class
----------------------

This is a very lightweight subclass of the :class:`networkx.Graph` class. This means that any methods you can use on a `Networkx.Graph` object can also be used on a `BrainNetwork` object, although the reverse is not true. We have **added** various methods that allow us to keep track of measures that have already been calculated. This is particularly useful later on when one is dealing with 1000 random graphs (or more!) and saves a lot of time.

All :class:`BrainNetwork` **methods** have a corresponding ``scona`` **function**. So while the :class:`BrainNetwork` methods can only be applied to :class:`BrainNetwork` objects, you can find the equivalent function in ``scona`` which can be used on a regular :class:`networkx.Graph` object.

For example, if `G` is a `BrainNetwork` object, you can threshold it by typing `G.threshold(10)`. If `nxG` is a `Networkx.Graph` you can use `scn.threshold_graph(nxG, 10)` to perform the same function.

A :class:`BrainNetwork` can be initialised from a :class:`networkx.Graph` or from a correlation matrix represented as a :class:`pandas.DataFrame` or :class:`numpy.array`.


Resources
---------