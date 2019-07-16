import warnings

import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from scona.visualisations_helpers import save_fig
from scona.visualisations_helpers import df_sns_barplot
from scona.visualisations_helpers import setup_color_list
from scona.visualisations_helpers import add_colorbar
from scona.visualisations_helpers import anatomical_layout


def plot_rich_club(brain_bundle, real_network, figure_name=None, color=None,
                   show_legend=True, x_max=None, y_max=None):
    """
    This is a visualisation tool for plotting the rich club values per degree
    along with the random rich club values created from a random networks
    with a preserved degree distribution.

    Parameters
    ----------
    brain_bundle : `GraphBundle` object
        a python dictionary with BrainNetwork objects as values
        (:class:`str`: :class:`BrainNetwork` pairs), contains real Graph and random graphs
    real_network: str, required
        This is the name of the real Graph in GraphBundle.
        While instantiating GraphBundle object we pass the real Graph and its name.
        (e.g. bundleGraphs = scn.GraphBundle([H], ['Real Network'])).
        To plot rich club values along with the rich club values from randomed graphs
        it is required to pass the name of the real network (e.g.'Real Network').
    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name")
        or to store in the current directory include just a name ("fig_name");
    color : list of 2 strings, optional
        where the 1st string is a color for rich club values and 2nd - for random
        rich club values. You can specify the color using an html hex string
        (e.g. color =["#06209c","#c1b8b1"]) or you can pass an (r, g, b) tuple,
        where each of r, g, b are in the range [0,1]. Finally, legal html names
        for colors, like "red", "black" and so on are supported.
    show_legend: bool (optional, default=True)
        if True - show legend, otherwise - do not display legend.
    x_max : int, optional
        the max length of the x-axis of the plot
    y_max : int, optional
        the max length of the y-axis of the plot

    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name file.

    """

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # calculate rich club coefficients for each graph in Graph Bundle
    rich_club_df = brain_bundle.report_rich_club()

    # get the degrees
    degree = rich_club_df.index.values

    # select the values of the 1st Graph in Graph Bundle - Real Graph
    rc_real = np.array(rich_club_df[real_network])

    # create a dataframe of random Graphs (exclude Real Graph)
    rand_df = rich_club_df.drop(real_network, axis=1)

    # re-organize rand_df dataframe in a suitable way
    # so that there is one column for the degrees data, one for rich club values
    # required for seaborn plotting with error bars

    # create array to store the degrees
    rand_degree = []

    # create array to store a rich_club values according to the degree
    rc_rand = []

    # append each column in rand_df to a list
    for i in range(len(rand_df.columns)):
        rand_degree = np.append(rand_degree, rand_df.index.values)
        rc_rand = np.append(rc_rand, rand_df.iloc[:, i])

    new_rand_df = pd.DataFrame({'Degree': rand_degree, 'Rich Club': rc_rand})

    # create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # set the default colors of plotted values if not provided
    if color is None:
        color = ["#00C9FF", "grey"]
    elif len(color) == 1:              # in case only to plot only real values
        color.append("grey")

    # if the user provided color not as a list of size 2 - show warning, use default colors
    if not isinstance(color, list) and len(color) != 2:
        warnings.warn("Please, provide a *color* parameter as a "
                      "python list object, e.g. [\"green\", \"pink\"]. "
                      "Right now the default colors will be used")
        color = ["#00C9FF", "grey"]

    # plot the random rich club values of random graphs
    ax = sns.lineplot(x="Degree", y="Rich Club", data=new_rand_df,
                      err_style="band", ci=95, color=color[1],
                      label="random rich-club coefficient")

    # plot the rich club values of real Graph
    ax = sns.lineplot(x=degree, y=rc_real, label="rich-club coefficient",
                      color=color[0])

    # set the max values of x & y - axis if not provided
    if x_max is None:
        x_max = max(degree)

    if y_max is None:
        y_max = max(rc_real) + 0.1   # let y-axis be longer -> looks better

    # set the x and y axis limits
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))

    # set the number of bins to 4
    ax.locator_params(nbins=4)

    # set the x and y axis labels
    ax.set_xlabel("Degree")
    ax.set_ylabel("Rich Club")

    # create a legend if show_legend = True, otherwise - remove
    if show_legend:
        ax.legend(fontsize="x-small")
    else:
        ax.legend_.remove()

    # remove the top and right spines from plot
    sns.despine()

    # adjust subplot params so that the subplot fits in to the figure area
    plt.tight_layout()

    # display the figure
    plt.show()

    # save the figure if the location-to-save is provided
    if figure_name:
        # use the helper-function from module helpers to save the figure
        save_fig(fig, figure_name)
        # close the file after saving to a file
        plt.close(fig)


def plot_network_measures(brain_bundle, real_network, figure_name=None,
                          color=None, ci=95, show_legend=True):
    """
    This is a visualisation tool for plotting network measures values
    along with the random network values created from a random networks.

    Parameters
    ----------
    brain_bundle : :class:`GraphBundle`
        a python dictionary with BrainNetwork objects as values
        (:class:`str`: :class:`BrainNetwork` pairs), contains real Graph and
        random graphs
    real_network: str, required
        This is the name of the real Graph in GraphBundle.
        While instantiating GraphBundle object we pass the real Graph and its
        name (e.g. bundleGraphs = scn.GraphBundle([H], ['Real Network'])).
        To plot real network measures along with the random network values it is
        required to pass the name of the real network (e.g.'Real Network').
    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name") # noqa
        or to store in the current directory include just a name ("fig_name");
    color : list of 2 strings, optional
        where the 1st string is a color for real network measures and the 2nd
        - for measures values of random graphs.
        You can specify the color using an html hex string
        (e.g. color =["#06209c","#c1b8b1"]) or you can pass an (r, g, b) tuple,
        where each of r, g, b are in the range [0,1]. Finally, legal html names
        for colors, like "red", "black" and so on are supported.
    show_legend: bool (optional, default=True)
        if True - show legend, otherwise - do not display legend.
    ci: float or “sd” or None (optional, default=95)
        Size of confidence intervals to draw around estimated values. If “sd”,
        skip bootstrapping and draw the standard deviation of the observations.
        If None, no bootstrapping will be performed, and error bars will not be
        drawn.
    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name file.
    """

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # calculate network measures for each graph in brain_bundle
    # if each graph in GraphBundle has already calculated global measures,
    # this step will be skipped
    bundleGraphs_measures = brain_bundle.report_global_measures()

    # build a new DataFrame required for seaborn.barplot
    seaborn_data = df_sns_barplot(bundleGraphs_measures, real_network)

    # set the default colors of barplot values if not provided
    if color is None:
        color = [sns.color_palette()[0], "lightgrey"]
    elif len(color) == 1:            # in case we want to plot only real values
        color.append("lightgrey")

    # if the user provided color not as a list of size 2 - show warning
    # use default colors
    if not isinstance(color, list) and len(color) != 2:
        warnings.warn("Please, provide a *color* parameter as a "
                      "python list object, e.g. [\"green\", \"pink\"]. "
                      "Right now the default colors will be used")
        color = [sns.color_palette()[0], "lightgrey"]

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot global measures with error bars
    ax = sns.barplot(x="measure", y="value", hue="TypeNetwork",
                     data=seaborn_data, palette=[color[0], color[1]], ci=ci)

    # make a line at y=0
    ax.axhline(0, linewidth=0.8, color='black')

    # set labels for y axix
    ax.set_ylabel("Global network measures")
    ax.set_xlabel("")   # empty -> no x-label

    # create a legend if show_legend = True, otherwise - remove
    if show_legend:
        ax.legend(fontsize="xx-small")
    else:
        ax.legend_.remove()

    # remove the top and right spines from plot
    sns.despine()

    # adjust subplot params so that the subplot fits in to the figure area
    plt.tight_layout()

    # display the figure
    plt.show()

    # save the figure if the location-to-save is provided
    if figure_name:
        # use the helper-function from module helpers to save the figure
        save_fig(fig, figure_name)
        # close the file after saving to a file
        plt.close(fig)


def plot_degree_dist(G, binomial_graph=True, seed=10, figure_name=None, color=None):

    """
    This is a visualisation tool for plotting the degree distribution
    along with the degree distribution of an Erdos Renyi random graph
    that has the same number of nodes.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        BrainNetwork object
    binomial_graph : bool (optional, default=True)
        if "True" plot the degree distribution of an Erdos Renyi random graph.
    seed : integer (default=10), random_state, or None
        Seed for random number generator. In case it is needed to create random
        Erdos Renyi Graph, set `seed` to None.
    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name")
        or to store in the current directory include just a name ("fig_name");
    color : list of 2 strings, optional
        where the 1st string is a color for rich club values and 2nd - for random
        rich club values. You can specify the color using an html hex string
        (e.g. color =["#06209c","#c1b8b1"]) or you can pass an (r, g, b) tuple,
        where each of r, g, b are in the range [0,1]. Finally, legal html names
        for colors, like "red", "black" and so on are supported.

    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name file.

    """

    # set the default colors of plotted values if not provided
    if color is None:
        color = [sns.color_palette()[0], "grey"]

    # if the user provided color not as a list of size 2 - show warning, use default colors
    if not isinstance(color, list) and len(color) == 2:
        warnings.warn("Please, provide a *color* parameter as a "
                      "python list object, e.g. [\"green\", \"pink\"]. "
                      "Right now the default colors will be used")
        color = [sns.color_palette()[0], "grey"]

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # calculate the degrees from the graph
    degrees = np.array(list(dict(G.degree()).values()))

    # calculate the Erdos Renyi graph from the main graph
    nodes = len(G.nodes())
    cost = G.number_of_edges() * 2.0 / (nodes*(nodes-1))    # probability for edge creation
    G_ER = nx.erdos_renyi_graph(nodes, cost, seed=seed)

    # calculate the degrees for the ER graph
    degrees_ER = np.array(list(dict(G_ER.degree()).values()))

    # create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot distribution of graph's degrees
    ax = sns.distplot(degrees, color=color[0])

    # plot a Erdos Renyi graph density estimate
    if binomial_graph:
        ax = sns.kdeplot(degrees_ER, color=color[1])

    # fix the x axis limits - without the gap between the 1st column and x = 0 - start from 1
    ax.set_xlim((1, max(degrees)))

    # set the number of bins to 5
    ax.locator_params(axis="x", nbins=5)

    # Set the x and y axis labels
    ax.set_xlabel("Degree")
    ax.set_ylabel("Probability")

    # remove the top and right spines from plot
    sns.despine()

    # adjust subplot params so that the subplot fits in to the figure area
    plt.tight_layout()

    # display the figure
    plt.show()

    # save the figure if the location-to-save is provided
    if figure_name:
        # use the helper-function from module helpers to save the figure
        save_fig(fig, figure_name)
        # close the file after saving to a file
        plt.close(fig)


def plot_anatomical_network(G10, G02, measure="module", orientation="sagittal",
                            node_list=None, node_shape='o', node_size=300,
                            rc_node=None, rc_node_shape='s',
                            edge_list=None, edge_color='lightgrey', edge_width=1,   # noqa
                            cmap_name="tab10", sns_palette=None, continuous=False,  # noqa
                            vmin=None, vmax=None, figure_name=None):
    """
    Plots each node in the graph in one of three orientations
    (sagittal, axial or coronal).
    The nodes are sorted according to the orientation given
    (default value: sagittal) and then plotted in that order. The color for each
    node is determined by the measure value (default measure: module).

    Parameters
    ----------
    G10 : :class:`BrainNetwork`
        A binary graph with 10% as many edges as the complete graph G

    G02 : :class:`BrainNetwork`
        A binary graph with 2% as many edges as the complete graph G

    measure: str, (optional, default="module")
        The name of a nodal measure.

    orientation : str, (optional, default="sagittal")
        The anatomical plane of the brain, i.g. sagittal, coronal, axial.

    node_list: list, optional
        Draw only specified nodes (default G.nodes()). By default, all nodes
        of graph will be displayed.

    node_shape : string
        The shape of the node.  Specification is as matplotlib.scatter marker,
        one of 'so^>v<dph8' (default='o').

    node_size : scalar or array
        Size of nodes (default=300).
        If an array is specified it must be the same length as node_list.
        The number in node_size at index <i> - is the size of the node at index
        <i> in node_list. For example, node_list=[5,6,7] and node_size=[100,200,
        300], the sizes for nodes will be {5:100, 6:200, 7:300}
        (100=node_size[0], node_list[0]=5 -> 5:100). If we plot all nodes
        (node_list=None), each element in node_size at index <i> is the size of
        <i> node in Graph.

    rc_node: list, optional
        List of nodes in the Graph that are rich club.

    rc_node_shape: string
        The shape of rich club nodes. Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8' (default='s').

    edge_list : list of tuples, optional
        Draw only specified edges(default=G.edges()).
        List of edges, where each edge is represented as a tuple, e.g. (0,1).
        Edges of a graph can be accessed by executing `Graph.edges()`.

    edge_color: color string, or array of floats
        Edge color. Can be a single color format string (default='lightgrey'),
        or a sequence of colors with the same length as edgelist.

    edge_width: float, or array of floats
        Line width of edges (default=1.0).

    cmap : Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None).

    sns_palette: seaborn palette, (optional, default=None)
        Discrete color palette only for discrete data. List of colors defining
        a color palette (list of RGB tuples from seaborn color palettes).

    continuous: bool, (optional, default=False)
        Indicate whether the data values are is discrete (False) or
        continuous (True).

    vmin,vmax : floats
        Minimum and maximum for node colormap scaling (default=None).
        Measure values will be normalized into the [vmin, vmax] interval.
        If vmin or vmax is not given (None), they are initialized from
        the minimum and maximum value respectively of the first input processed.

    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name")   # noqa
        or to store in the current directory include just a name ("fig_name").

    Returns
    -------
        Plot the Figure, save it in a figure_name file, if figure_name is given.   # noqa

    """

    # set seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=2)

    # get the graph's nodes if not provided
    if node_list is None:
        node_list = G10.nodes()
        node_list = sorted(node_list)

    # store {node : size_node} as a dict
    node_size_list = dict()

    # handle different types of node_size (scalar or array)
    if isinstance(node_size, int) or isinstance(node_size, float):
        for i in node_list:
            node_size_list[i] = node_size
    elif (isinstance(node_size, list) or isinstance(node_size, np.ndarray)) \
            and len(node_size) == len(node_list):
        for i, node in enumerate(node_list):
            node_size_list[node] = node_size[i]
    else:
        raise ValueError("node_size parameter must be either scalar or array. "
              "If an array is specified it must be the same length as nodelist")

    # set shape of nodes
    node_shape_list = dict()
    for i in node_list:
        node_shape_list[i] = node_shape

    # set shape of rich_club nodes if rc_node given
    if rc_node:
        for i in rc_node:
            node_shape_list[i] = rc_node_shape

    # report the attributes of each node in BrainNetwork Graph
    nodal_measures = G10.report_nodal_measures()

    # no centroids -> no plot of graph
    if not G10.graph["centroids"]:
        raise TypeError("There are no centroids (nodes coordinates) in the "
                        "Graph. Please initialise BrainNetwork with centroids.")

    # get the appropriate positions for each node based on the orientation
    pos = dict()

    for node in G10.nodes:
        pos[node] = anatomical_layout(G10.node[node]['x'], G10.node[node]['y'],
                                      G10.node[node]['z'],
                                          orientation=orientation)

    # we're going to figure out the best way to plot these nodes
    # so that they're sensibly on top of each other
    sort_dict = {}
    sort_dict['axial'] = 'z'
    sort_dict['coronal'] = 'y'
    sort_dict['sagittal'] = 'x'

    # sort nodes [from min_coordinate to max_coordinate]
    # where coordinate determined by the provided orientation
    # this is the order of displaying nodes based on the orientation
    node_order = np.argsort(nodal_measures[sort_dict[orientation]].values)

    # now remove all the nodes that are not in the node_list
    node_order = [x for x in node_order if x in node_list]

    # get the color for each node based on the nodal measure
    if measure in nodal_measures.columns:
        colors_list = setup_color_list(df=nodal_measures, cmap_name=cmap_name,
                                       sns_palette=sns_palette, measure=measure,
                                       continuous=continuous, vmin=vmin, vmax=vmax)   # noqa
    else:
        warnings.warn(
            "Measure \"{}\" does not exist in the nodal attributes of Graph. "
            "The default color will be used for all nodes.".format(measure))
        colors_list = ["mediumturquoise"] * len(node_order)

    # define the size of a figure for each orientation
    fig_size_dict = {}
    fig_size_dict['axial'] = (9, 12)
    fig_size_dict['sagittal'] = (12, 9)
    fig_size_dict['coronal'] = (11, 9)

    # create the figure
    fig = plt.figure(figsize=fig_size_dict[orientation])

    # create grid for axes1 - brain_plotting, axes2 - colorbar
    # num_rows=2, num_col=1, height rations of the rows (50,1) and no space
    # between brain plot and colorbar plot
    grid = mpl.gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[50, 1],
                                 hspace=0)

    # Add an axis containing brain-plot to the figure
    ax_brain = plt.Subplot(fig, grid[0])
    fig.add_subplot(ax_brain)

    # loop through each node and plot one after each other
    for node in node_order:
        nx.draw_networkx_nodes(G10,
                               pos=pos,
                               node_color=colors_list[node],
                               node_size=node_size_list[node],
                               node_shape = node_shape_list[node],
                               nodelist=[node],
                               ax=ax_brain)

    # get the graph's edges if not given
    if edge_list is None:
        # plot edges of G02 (thresholded at cost 2)-(nodes important, not edges)
        edge_list = list(G02.edges())

    # plot edges
    nx.draw_networkx_edges(G10,
                           pos=pos,
                           edgelist=edge_list,
                           edge_color=edge_color,
                           width=edge_width,
                           ax=ax_brain)

    # if plotting continuous data, add the colorbar to the plot
    if continuous:

        # calculate vmin and vmax of data values (if not given) for colorbar
        if (vmin is None) and (measure in nodal_measures.columns):
            vmin = min(nodal_measures[measure].values)
        if (vmax is None) and (measure in nodal_measures.columns):
            vmax = max(nodal_measures[measure].values)

        add_colorbar(fig, grid[1], cb_name=measure, cmap_name=cmap_name,
                     vmin=vmin, vmax=vmax)

        # adjust the size of the grid (left=0, right=1, bottom=0, top=1)
        grid.update(bottom=0.1, top=1)

    # remove all spines from plot
    sns.despine(top=True, right=True, left=True, bottom=True)

    # display the figure
    plt.show()

    # save the figure if the location-to-save is given
    if figure_name:
        # use the helper-function from module helpers to save the figure
        save_fig(fig, figure_name)
        # close the file after saving to a file
        plt.close(fig)
