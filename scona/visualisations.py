import warnings

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting

from scona.visualisations_helpers import save_fig
from scona.visualisations_helpers import create_df_sns_barplot
from scona.visualisations_helpers import graph_to_nilearn_array
from scona.visualisations_helpers import setup_color_list


def plot_rich_club(brain_bundle, original_network, figure_name=None,
                   color=None, show_legend=True, x_max=None, y_max=None):
    """
    This is a visualisation tool for plotting the rich club values per degree
    along with the random rich club values created from a random networks
    with a preserved degree distribution.

    Parameters
    ----------
    brain_bundle : `GraphBundle` object
        a python dictionary with BrainNetwork objects as values
        (:class:`str`: :class:`BrainNetwork` pairs), contains original Graph
        and random graphs.
    original_network: str, required
        This should index the particular network in `brain_bundle` that you
        want the figure to highlight. A distribution of all the other networks
        in `brain_bundle` will be rendered for comparison.
    figure_name : str, optional
        path to the file to store the created figure in
        (e.g. "/home/Desktop/name") or to store in the current directory
        include just a name ("fig_name");
    color : list of 2 strings, optional
        where the 1st string is a color for rich club values and 2nd - for
        random rich club values. You can specify the color using an html hex
        string (e.g. color =["#06209c","#c1b8b1"]) or you can pass an
        (r, g, b) tuple, where each of r, g, b are in the range [0,1].
        Legal html names for colors, like "red", "black" and so on are also
        supported.
    show_legend: bool (optional, default=True)
        if True - show legend, otherwise - do not display legend.
    x_max : int, optional
        the max length of the x-axis of the plot
    y_max : int, optional
        the max length of the y-axis of the plot

    Returns
    -------
        Plot the figure and if figure_name is given then save the image
        in a file named according to the figure_name variable.
    """

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # calculate rich club coefficients for each graph in Graph Bundle
    rich_club_df = brain_bundle.report_rich_club()

    # get the degrees
    degree = rich_club_df.index.values

    # select the values of the 1st Graph in Graph Bundle - Real Graph
    try:
        rc_orig = np.array(rich_club_df[original_network])
    except KeyError:
        raise KeyError(
            "Please check the name of the initial Graph (the proper network, "
            "the one you got from the mri data) in GraphBundle. There is"
            " no graph keyed by name \"{}\"".format(original_network))

    # create a dataframe of random Graphs (exclude Real Graph)
    rand_df = rich_club_df.drop(original_network, axis=1)

    # re-organize rand_df dataframe in a suitable way
    # so that there is one column for the degrees data, one for rich club
    # values required for seaborn plotting with error bars

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
    elif len(color) == 1:              # if you only want to plot the original
        color.append("grey")           # network (no random networks)

    # if the user provided color not as a list of size 2 - show a warning
    # and then carry on but using the default colors

    if not isinstance(color, list) and len(color) != 2:
        warnings.warn("Please, provide a *color* parameter as a "
                      "python list object, e.g. [\"green\", \"pink\"]. "
                      "Right now the default colors will be used")
        color = ["#00C9FF", "grey"]

    # plot the rich club values of real Graph
    ax = sns.lineplot(x=degree, y=rc_orig, label="Observed network", zorder=1,
                      color=color[0])

    # plot the random rich club values of random graphs
    ax = sns.lineplot(x="Degree", y="Rich Club", data=new_rand_df,
                      err_style="band", ci=95, color=color[1],
                      label="Random network", zorder=2)

    # set the max values of x & y - axis if not given
    if x_max is None:
        x_max = max(degree)

    if y_max is None:
        y_max = max(rc_orig) + 0.1   # let y-axis be longer -> looks better

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


def plot_network_measures(brain_bundle, original_network, figure_name=None,
                          color=None, ci=95, show_legend=True):
    """
    This is a visualisation tool for plotting network measures values
    along with the random network values created from a random networks.

    Parameters
    ----------
    brain_bundle : :class:`GraphBundle`
        a python dictionary with BrainNetwork objects as values
        (:class:`str`: :class:`BrainNetwork` pairs), contains real Graph and
        random graphs.
    original_network: str, required
        This should index the particular network in `brain_bundle` that you
        want the figure to highlight. A distribution of all the other networks
        in `brain_bundle` will be rendered for comparison.
    figure_name : str, optional
        path to the file to store the created figure in
        (e.g. "/home/Desktop/name") or to store in the current directory
        include just a name ("fig_name").
    color : list of 2 strings, optional
        where the 1st string is a color for original network measures and the
        2nd is for the values from the random graphs.
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
        Plot the Figure and if figure_name provided, save it in a figure_name
        file.
    """

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # build a new DataFrame required for seaborn.barplot
    seaborn_data = create_df_sns_barplot(brain_bundle, original_network)

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


def plot_degree_dist(G, binomial_graph=True, seed=10, figure_name=None,
                     color=None):
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
        Plot the Figure and if figure_name given, save it in a figure_name file.
    """

    # set the default colors of plotted values if not provided
    if color is None:
        color = [sns.color_palette()[0], "grey"]

    # if the user provided color not as a list of size 2
    # show warning, use default colors
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

    # set the cost as the probability for edge creation
    cost = G.number_of_edges() * 2.0 / (nodes*(nodes-1))
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

    # fix the x axis limits without the gap between the 1st column and x = 0
    # start from 1
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
        continuous=False,
        vmin=None,
        vmax=None):
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
        Indicate whether the data values are discrete (False) or
        continuous (True).

    vmin : scalar or None, optional
        The minimum value used in colormapping *data*. If *None* the minimum
        value in *data* is used.

    vmax : scalar or None, optional
        The maximum value used in colormapping *data*. If *None* the maximum
        value in *data* is used.

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
                                          continuous=continuous,
                                          vmin=vmin,
                                          vmax=vmax)
        else:
            warnings.warn(
              "Measure \"{}\" does not exist in nodal attributes of graph. "
              "The default color will be used for all nodes.".format(measure))
            node_color = [node_color for _ in range(len(node_coords))]

    # plot nodes
    ConnectomeView = plotting.view_markers(node_coords,
                                           marker_color=node_color,
                                           marker_size=node_size)

    return ConnectomeView


def view_connectome_3d(
        G,
        edge_threshold="98%",
        edge_cmap="Spectral_r",
        symmetric_cmap=False,
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

    edge_threshold : str, number or None, optional (default="2%")
        If None, no thresholding.
        If it is a number only connections of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only connections of amplitude above the
        given percentile will be shown.

    edge_cmap : str or matplotlib colormap, optional
        Colormap for displaying edges.

    symmetric_cmap : bool, optional (default=False)
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

    # get the adjacency matrix and nodes coordinates
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


def plot_connectome(
        G,
        node_color='auto', node_size=50,
        edge_cmap=plt.cm.bwr,
        edge_vmin=None, edge_vmax=None,
        edge_threshold=None,
        output_file=None, display_mode='ortho',
        figure=None, axes=None, title=None,
        annotate=True, black_bg=False,
        alpha=0.7,
        edge_kwargs=None, node_kwargs=None,
        colorbar=False):
    """
    Plot connectome on top of the brain glass schematics.

    The plotted image should be in MNI space for this function to work
    properly.

    In the case of ‘l’ and ‘r’ directions (for hemispheric projections),
    markers in the coordinate x == 0 are included in both hemispheres.

    Plot a BrainNetwork using :func:`nilearn.plotting.plot_connectome()` tool.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids".

    node_color : color or sequence of colors, optional
        color(s) of the nodes. If string is given, all nodes
        are plotted with same color given in string.

    node_size : scalar or array_like, optional (default=50)
        size(s) of the nodes in points^2.

    edge_cmap : colormap, optional (default="bwr")
        colormap used for representing the strength of the edges.

    edge_vmin : float, optional (default=None)

    edge_vmax : float, optional (default=None)
        If not None, either or both of these values will be used to
        as the minimum and maximum values to color edges. If None are
        supplied the maximum absolute value within the given threshold
        will be used as minimum (multiplied by -1) and maximum
        coloring levels.

    edge_threshold : str or number, optional (default=None)
        If it is a number only the edges with a value greater than
        edge_threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only the edges with a abs(value) above
        the given percentile will be shown.

    output_file : string, or None, optional (default=None)
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    display_mode : string, optional (default='ortho')
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only, 'ortho' - three cuts are
        performed in orthogonal directions. Possible values are: 'ortho',
        'x', 'y', 'z', 'xz', 'yx', 'yz', 'l', 'r', 'lr', 'lzr', 'lyr',
        'lzry', 'lyrz'.

    figure : integer or matplotlib figure, optional (default=None)
        Matplotlib figure used or its number. If None is given, a
        new figure is created.

    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height),
           optional (default=None)
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.

    title : string, optional (default=None)
        The title displayed on the figure.

    annotate : boolean, optional (default=True)
        If annotate is True, positions and left/right annotation
        are added to the plot.

    black_bg : boolean, optional (default=False)
        If True, the background of the image is set to be black. If
        you wish to save figures with a black background, you
        will need to pass "facecolor='k', edgecolor='k'"
        to matplotlib.pyplot.savefig.

    alpha : float between 0 and 1, optional (default=0.7)
        Alpha transparency for the brain schematics.

    edge_kwargs : dict, optional (default=None)
        will be passed as kwargs for each edge matlotlib Line2D.

    node_kwargs : dict, optional (default=None)
        will be passed as kwargs to the plt.scatter call that plots all
        the nodes in one go.

    colorbar : bool, optional (default=False)
        If True, display a colorbar on the right of the plots.
        By default it is False.

    """

    # get the adjacency matrix and nodes coordinates
    adj_matrix, node_coords = graph_to_nilearn_array(G)

    # plot connectome
    display = plotting.plot_connectome(adjacency_matrix=adj_matrix,
                                       node_coords=node_coords,
                                       node_color=node_color,
                                       node_size=node_size,
                                       edge_cmap=edge_cmap,
                                       edge_vmin=edge_vmin,
                                       edge_vmax=edge_vmax,
                                       edge_threshold=edge_threshold,
                                       output_file=output_file,
                                       display_mode=display_mode,
                                       figure=figure,
                                       axes=axes,
                                       title=title,
                                       annotate=annotate,
                                       alpha=alpha,
                                       black_bg=black_bg,
                                       edge_kwargs=edge_kwargs,
                                       node_kwargs=node_kwargs,
                                       colorbar=colorbar)

    return display
