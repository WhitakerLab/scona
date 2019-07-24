import warnings

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting

from scona.visualisations_helpers import save_fig
from scona.visualisations_helpers import df_sns_barplot
from scona.visualisations_helpers import graph_to_nilearn_array
from scona.visualisations_helpers import setup_color_list


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


def view_nodes_3d(
        G,
        node_size=5.,
        node_color='black',
        measure=None,
        cmap_name=None,
        sns_palette=None,
        continuous=False):
    """
    Plot nodes of a BrainNetwork using
    :func:`nilearn.plotting.view_markers()` tool.

    Insert a 3d plot of markers in a brain into an HTML page.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"

    node_size : float or array-like, optional (default=5.)
        Size of the nodes showing the seeds in pixels.

    node_color : str or list of str (default 'black')
        node_colour determines the colour given to each node.
        If a single string is given, this string will be interpreted as a
        a colour, and all nodes will be rendered in this colour.
        If a list of colours is given, it must be the same length as the length
        of nodes coordinates.

    measure: str, (optional, default=None)
        The name of a nodal measure.

    cmap_name : Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None).

    sns_palette: seaborn palette, (optional, default=None)
        Discrete color palette only for discrete data. List of colors defining
        a color palette (list of RGB tuples from seaborn color palettes).

    continuous: bool, (optional, default=False)
        Indicate whether the data values are is discrete (False) or
        continuous (True).

    Returns
    -------
    ConnectomeView : plot of the nodes.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :
        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.
    """

    # get the nodes coordinates
    adj_matrix, node_coords = graph_to_nilearn_array(G)

    # apply color to all nodes in Graph if node_color is string
    if isinstance(node_color, str):
        node_color = [node_color for _ in range(len(node_coords))]

    # report the attributes of each node in BrainNetwork Graph
    nodal_measures = G.report_nodal_measures()

    # get the color for each node based on the nodal measure
    if measure:
        if measure in nodal_measures.columns:
            node_color = setup_color_list(df=nodal_measures, measure=measure,
                                           cmap_name=cmap_name,
                                           sns_palette=sns_palette,
                                           continuous=continuous)
        else:
            warnings.warn(
              "Measure \"{}\" does not exist in the nodal attributes of Graph. "
              "The default color will be used for all nodes.".format(measure))
            node_color = [node_color for _ in range(len(node_coords))]

    # plot nodes
    ConnectomeView = plotting.view_markers(node_coords, marker_color=node_color,
                                           marker_size=node_size)

    return ConnectomeView


def view_connectome_3d(
        G,
        edge_threshold=None,
        edge_cmap="Spectral_r",
        symmetric_cmap=True,
        linewidth=6.,
        node_size=3.):
    """
    Insert a 3d plot of a connectome into an HTML page.

    Plot a BrainNetwork using :func:`nilearn.plotting.view_connectome()` tool.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids".

    edge_threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.

    edge_cmap : str or matplotlib colormap, optional
        Colormap for displaying edges.

    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).

    linewidth : float, optional (default=6.)
        Width of the lines that show connections.

    node_size : float, optional (default=3.)
        Size of the markers showing the seeds in pixels.

    Returns
    -------
    ConnectomeView : plot of the connectome.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :
        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    """

    # get the nodes coordinates
    adj_matrix, node_coords = graph_to_nilearn_array(G)

    # plot connectome
    ConnectomeView = plotting.view_connectome(adjacency_matrix=adj_matrix,
                                              node_coords=node_coords,
                                              edge_threshold=edge_threshold,
                                              edge_cmap=edge_cmap,
                                              symmetric_cmap=symmetric_cmap,
                                              linewidth=linewidth,
                                              node_size=node_size)

    return ConnectomeView
