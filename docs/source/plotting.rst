Plotting Brain Networks
=======================

A good visualisation is worth millions of words. 
That's why our community put an emphasis on ensuring that the produced figures are informative and communicate the results clearly.

The :class:`scona` package has a great visualisation module, which enables you to produce publication-ready plots based on the results of the performed brain network analysis. 
You are also able to export created figures (save as a file) to include in a research paper or in a publication.

As it says the best explanation is through demonstration. That's why we have created `tutorials <https://github.com/WhitakerLab/scona/tree/master/tutorials>`_ which are jupyter notebooks that include visualisation examples of the data. 
In a clear and easy way, these tutorials explain how different visualisation functions can be used to produce different plots and for better data understanding.

1. Global Network Measures visualisation `tutorial <https://github.com/WhitakerLab/scona/blob/master/tutorials/global_measures_viz.ipynb>`_ describes how to use the following functions:
    - plot_degree_dist() - tool for plotting the degree distribution
    - plot_network_measures() - tool for plotting network measures values
    - plot_rich_club() -  tool for plotting the rich club values per degree

With the help of these functions, you can report measures relating to the whole network.

2. Interactive visualisation `tutorial <https://github.com/WhitakerLab/scona/blob/master/tutorials/interactive_viz_tutorial.ipynb>`_ provides examples on how to use functions like:
    - view_nodes_3d() - view the nodes on a 3d plot
    - view_connectome_3d() - view the edges - the connections - of the network on a 3d plot

These tools rely on the excellent `nilearn.plotting library <http://nilearn.github.io/plotting/index.html>`_.

3. Anatomical visualisation `tutorial <https://github.com/WhitakerLab/scona/blob/master/tutorials/anatomical_viz_tutorial.ipynb>`_ shows the usage of the following functions:
    - plot_anatomical_network() - make plots of nodes and edges based on the given anatomical layout
    - plot_connectome () - plot connectome on top of the brain glass schematics
    
These are static visualisions that you could use to report your findings in a published paper.


With proper visualization, a researcher can reveal findings easier, understand complex data relationships and describe obtained insights from analyzed data.

