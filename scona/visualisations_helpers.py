import warnings
import os

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def create_df_sns_barplot(bundleGraphs, original_network):
    """
    In order to plot barplot (with error bars) with the help of seaborn,
    it is needed to pass a argument "data" - dataset for plotting.

    This function restructures a DataFrame obtained from
    `Graph.Bundle.report_global_measures` into an acceptable DataFrame for
    seaborn.barplot.

    Parameters
    ----------
    bundleGraphs : :class:`GraphBundle`
        a python dictionary with BrainNetwork objects as values
        (:class:`str`: :class:`BrainNetwork` pairs), contains real Graph and
        random graphs

    original_network: str, required
        This is the name of the initial Graph in GraphBundle. It should index
        the particular network in `brain_bundle` that you want the figure
        to highlight. A distribution of all the other networks in
        `brain_bundle` will be rendered for comparison.

    Returns
    -------
    :class:`pandas.DataFrame`
        Restructured DataFrame suitable for seaborn.barplot
    """

    # calculate network measures for each graph in brain_bundle
    # if each graph in GraphBundle has already calculated global measures,
    # this step will be skipped
    bundleGraphs_measures = bundleGraphs.report_global_measures()

    # set abbreviations for measures
    abbreviation = {'assortativity': 'a',
                    'average_clustering': 'C',
                    'average_shortest_path_length': 'L',
                    'efficiency': 'E',
                    'modularity': 'M'}

    # set columns for our new DataFrame
    new_columns = ["measure", "value", "TypeNetwork"]

    # get the number of columns from the old DataFrame
    no_columns_old = len(bundleGraphs_measures.columns)

    # get the number of rows from the old DataFrame
    no_rows_old = len(bundleGraphs_measures.index)

    # set number of rows (indexes) in new DataFrame
    total_rows = no_columns_old * no_rows_old

    # set index for our new DataFrame
    index = [i for i in range(1, total_rows + 1)]

    # Build array to contain all data to futher use for creating new DataFrame

    # store values of *Real Graph* in data_array - used to create new DataFrame
    data_array = list()

    for measure in bundleGraphs_measures.columns:
        # check that the param - original_network - is correct,
        # otherwise pass an error
        try:
            # for original_network get value of each measure
            value = bundleGraphs_measures.loc[original_network, measure]
        except KeyError:
            raise KeyError(
                "The name of the initial Graph you passed to the function - \"{}\""              # noqa
                " does not exist in GraphBundle. Please provide a true name of "
                "initial Graph (represented as a key in GraphBundle)".format(original_network))  # noqa

        # get the abbreviation for measure and use this abbreviation
        measure_short = abbreviation[measure]

        type_network = "Observed network"

        # create a temporary array to store measure - value of Real Network
        tmp = [measure_short, value, type_network]

        # add the record (measure - value - Real Graph) to the data_array
        data_array.append(tmp)

    # now store the measure and measure values of *Random Graphs* in data_array

    # delete Real Graph from old DataFrame -
    random_df = bundleGraphs_measures.drop(original_network)

    # for each measure in measures
    for measure in random_df.columns:

        # for each graph in Random Graphs
        for rand_graph in random_df.index:
            # get the value of a measure for a random Graph
            value = random_df[measure][rand_graph]

            # get the abbreviation for measure and use this abbreviation
            measure_short = abbreviation[measure]

            type_network = "Random network"

            # create temporary array to store measure - value of Random Network
            tmp = [measure_short, value, type_network]

            # add record (measure - value - Random Graph) to the global array
            data_array.append(tmp)

    # finally create a new DataFrame
    NewDataFrame = pd.DataFrame(data=data_array, index=index,
                                columns=new_columns)

    # include the small world coefficient into new DataFrame

    # check that random graphs exist in GraphBundle
    if len(bundleGraphs) > 1:
        # get the small_world values for Real Graph
        small_world = bundleGraphs.report_small_world(original_network)

        # delete the comparison of the graph labelled original_network with itself  # noqa
        del small_world[original_network]

        # create list of dictionaries to later append to the new DataFrame
        df_small_world = []
        for i in list(small_world.values()):
            tmp = {'measure': 'sigma',
                   'value': i,
                   'TypeNetwork': 'Observed network'}

            df_small_world.append(tmp)

        # add small_world values of *original_network* to new DataFrame
        NewDataFrame = NewDataFrame.append(df_small_world, ignore_index=True)

        # bar for small_world measure of random graphs should be set exactly to 1   # noqa

        # set constant value of small_world measure for random bar
        rand_small_world = {'measure': 'sigma',
                            'value': 1,
                            'TypeNetwork': 'Random network'}

        # add constant value of small_world measure for random bar to new DataFrame # noqa
        NewDataFrame = NewDataFrame.append(rand_small_world,
                                           ignore_index=True)

    return NewDataFrame


def save_fig(figure, path_name):
    """
    Helper function to save figure at the specified location - path_name

    Parameters
    ----------
    figure : :class:`matplotlib.figure.Figure`
        Figure to save.

    path_name : str
        Location where to save the figure.

    Returns
    -------
        Saves figure to the location - path_name
    """

    # If path_name ends with "/" - do not save, e.g. "/home/user/dir1/dir2/"
    if os.path.basename(path_name):

        # get the directory path (exclude file_name in the end of path_name)
        # For example: "/home/dir1/myfile.png" -> "/home/dir1"
        dir_path = os.path.dirname(path_name)

        # Create the output directory (dir_path) if it does not already exist
        # or if the path_name is not a directory
        if not os.path.exists(dir_path) and os.path.dirname(path_name):
            warnings.warn('The path "{}" does not exist.\n'
                          "We will create this directory for you "
                          "and store the figure there.\n"
                          "This warning is just to make sure that you aren't "
                          "surprised by a new directory "
                          "appearing!".format(dir_path))

            # Make the directory
            dir_create = os.path.dirname(path_name)
            os.makedirs(dir_create)
    else:
        warnings.warn('The file name you gave us "{}" ends with \"/\". '
                      "That is a directory rather than a file name."
                      "Please run the command again with the name of the file,"
                      "e.g. '/home/dir1/myfile.png' or to save the file in the "
                      "current directory you can just pass 'myfile.png'".format(path_name))     # noqa
        return

    # save the figure to the file
    figure.savefig(path_name, bbox_inches=0, dpi=100)


def setup_color_list(df, cmap_name='tab10', sns_palette=None, measure='module',
                     continuous=False, vmin=None, vmax=None):
    """
    Use a colormap to set color for each value in the DataFrame[column].
    Convert data values (floats) from the interval [vmin,vmax] to the
    RGBA colors that the respective Colormap represents.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The DataFrame that contains the required column (measure parameter).

    measure : str
        The name of the column in the df (pandas.DataFrame) that contains
        values intended to be mapped to colors.

    cmap_name : str or Colormap instance, (optional, default="tab10")
        The colormap used to map normalized data values to RGBA colors.

    sns_palette: seaborn palette, (optional, default=None)
        Discrete color palette only for discrete data. List of colors defining
        a color palette (list of RGB tuples from seaborn color palettes).

    continuous : bool,  optional (default=True)
        if *True* return the list of colors for continuous data.

    vmin : scalar or None, optional
        The minimum value used in colormapping *data*. If *None* the minimum
        value in *data* is used.

    vmax : scalar or None, optional
        The maximum value used in colormapping *data*. If *None* the maximum
        value in *data* is used.

    Returns
    -------
    list
        a list of colors for each value in the DataFrame[measure]
    """

    # Store pair (value, color) as a (key,value) in a dict
    colors_dict = {}

    # If vmin or vmax not passed, calculate the min and max
    # values of the column (measure)
    if vmin is None:
        vmin = min(df[measure].values)
    if vmax is None:
        vmax = max(df[measure].values)

    # The number of different colors needed
    num_color = len(set(df[measure]))

    # Continuous colorbar for continuous data
    if continuous:
        # Normalize data into the [0.0, 1.0] interval
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # Use of data normalization before returning RGBA colors from colormap
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_name)

        # Map normalized data values to RGBA colors
        colors_list = [ scalarMap.to_rgba(x) for x in df[measure] ]

    # For discrete data
    else:
        # Option 1: If you've passed a matplotlib color map
        try:
            cmap = mpl.cm.get_cmap(cmap_name)
        except ValueError:
            warnings.warn("ValueError: Colormap {} is not recognized. ".format(cmap_name) +
                            "Default colormap tab10 will be used.")
            cmap = mpl.cm.get_cmap("tab10")

        for i, value in enumerate(sorted(set(df[measure]))):
            colors_dict[value] = cmap((i+0.5)/num_color)

        # Option 2: If you've passed a sns_color_palette - use color_palette
        if sns_palette:
            color_palette = sns.color_palette(sns_palette, num_color)

            for i, value in enumerate(sorted(set(df[measure]))):
                colors_dict[value] = color_palette[i]

        # Make a list of colors for each node in df based on the measure
        colors_list = [ colors_dict[value] for value in df[measure].values ]

    return colors_list


def add_colorbar(fig, grid, cb_name, cmap_name, vmin=0, vmax=1):
    """
    Add a colorbar to the figure in the location defined by grid.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure`
        Figure to which colorbar will be added.

    grid : str
        Grid spec location to add colormap.

    cb_name: str, (optional, default=None)
        The label for the colorbar.
        Name of data values this colorbar represents.

    cmap_name : str or Colormap instance
        Name of the colormap

    vmin : scalar or None, (optional, default=0)
        Minimum value for the colormap

    vmax : scalar or None, (optional, default=1)
        Maximum value for the colormap

    Returns
    -------
    `matplotlib.figure.Figure` object
        figure with recently added colorbar
    """

    # add ax axes to the figure
    ax_cbar = plt.Subplot(fig, grid)
    fig.add_subplot(ax_cbar)

    # normalise the colorbar to have the correct upper and lower limits
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # set ticks (min, median, max) for colorbar
    ticks = [vmin, (vmin + vmax)/2, vmax]

    # put a colorbar in a specified axes,
    # and make colorbar for a given colormap
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap_name,
                                   norm=norm,
                                   ticks=ticks,
                                   format='%.2f',
                                   orientation='horizontal')

    # set the name of the colorbar
    if cb_name:
        cb.set_label(cb_name, size=20)

    # adjust the fontsize of ticks to look better
    ax_cbar.tick_params(labelsize=20)

    return fig


def axial_layout(x, y, z):
    """
    Axial (horizontal)  plane, the plane that is horizontal and parallel to the
    axial plane of the body. It contains the lateral and the medial axes of the
    brain.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    Returns
    -------
    numpy array
        The node coordinates excluding z-axis. `array([x, y])`

    """

    return np.array([x, y])


def sagittal_layout(x, y, z):
    """
    Sagittal plane, a vertical plane that passes from between the cerebral
    hemispheres, dividing the brain into left and right halves.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    Returns
    -------
    numpy array
        The node coordinates excluding y-axis. `array([x, z])`

    """

    return np.array([x, z])


def coronal_layout(x, y, z):
    """
    Coronal (frontal) plane, a vertical plane that passes through both ears,
    and contains the lateral and dorsoventral axes.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    Returns
    -------
    numpy array
        The node coordinates excluding y-axis. `array([y, z])`

    """

    return np.array([y, z])


def anatomical_layout(x, y, z, orientation="sagittal"):
    """
    This function extracts the required coordinates of a node based on the
    given anatomical layout.

    Parameters
    ----------
    x, y, z : float
        Node Coordinates

    orientation: str, (optional, default="sagittal)
        The name of the plane: `sagittal` or `axial` or `coronal`.

    Returns
    -------
    numpy array
        The node coordinates for the given anatomical layout.
    """

    if orientation == "sagittal":
        return sagittal_layout(x, y, z)
    if orientation == "axial":
        return axial_layout(x, y, z)
    if orientation == "coronal":
        return coronal_layout(x, y, z)
    else:
        raise ValueError(
            "{} is not recognised as an anatomical layout. orientation values "
            "should be one of 'sagittal', 'axial' or "
            "'coronal'.".format(orientation))


def graph_to_nilearn_array(
        G,
        edge_attribute="weight"):
    """
    Derive from G (BrainNetwork Graph) the necessary inputs for the `nilearn`
    graph plotting functions.

    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"

    edge_attribute : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for the edge
        weight. If an edge does not have that attribute, then the value 1 is
        used instead.

    Returns
    -------
    (adjacency_matrix, node_coords)
        adjacency_matrix - represents the link strengths of the graph;
        node_coords - 3d coordinates of the graph nodes in world space;
    """

    # make ordered nodes to produce ordered rows and columns
    # in adjacency matrix
    node_order = sorted(list(G.nodes()))

    # returns the graph adjacency matrix as a NumPy array
    adjacency_matrix = nx.convert_matrix.to_numpy_array(G, nodelist=node_order,
                                                        weight=edge_attribute)

    # store nodes coordinates in NumPy array if nodal coordinates exist
    try:
        node_coords = np.array([G._node[node]["centroids"] for node in node_order])       # noqa
    except KeyError:
        raise TypeError("There are no centroids (nodes coordinates) in the "
                        "Graph. Please initialise BrainNetwork "
                        "with the centroids.")

    return adjacency_matrix, node_coords
