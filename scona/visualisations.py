import warnings

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scona.helpers import save_fig


def plot_rich_club(brain_bundle, figure_name=None, color=None,
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
    rc_real = np.array(rich_club_df.iloc[:, 0])

    # create a dataframe of random Graphs (exclude Real Graph)
    rand_df = rich_club_df.drop(rich_club_df.columns.tolist()[0], axis=1)

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
    if not isinstance(color, list) and len(color) == 2:
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


def plot_network_measures(network_measures, rand_network_measures, figure_name=None,
                          color=None, label_bar=True):
    """
    This is a visualisation tool for plotting network measures values
    along with the random network values values created from a random network.

    Parameters
    ----------
    network_measures : dict
        real network measures
        Note: the dict could be obtained from calculate_global_measures()
    rand_network_measures : dict
        random network measure values
    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name")
        or to store in the current directory include just a name ("fig_name");
    color : list of 2 strings, optional
        where the 1st string is a color for rich club values and 2nd - for random
        rich club values. You can specify the color using an html hex string
        (e.g. color =["#06209c","#c1b8b1"]) or you can pass an (r, g, b) tuple,
        where each of r, g, b are in the range [0,1]. Finally, legal html names
        for colors, like "red", "black" and so on are supported.
    label_bar : bool (optional, default=True)
        show a measure value on top of each bar. Note - the value is rounded to
        2 decimals.

    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name file.

    """

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # make sure that values of the measures in network_measures
    # and rand_network_measures are aligned with each other
    sorted_net_measures = sorted(network_measures.keys())

    sorted_net_values = []
    sorted_random_net_values = []

    for i in sorted_net_measures:
        sorted_net_values.append(network_measures[i])

        try:
            sorted_random_net_values.append(rand_network_measures[i])
        except KeyError:
            warnings.warn( "There is no measure *{}* in random network mesures."
                           " The value *0* will used for measure - {}".format(i, i))
            sorted_random_net_values.append(0)

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set position of bar on X axis
    barWidth = 0.2
    r1 = np.arange(len(sorted_net_values))
    r2 = [x + barWidth + 0.05 for x in r1]

    # set the default colors of plotted values if not provided
    if color is None:
        color = [sns.color_palette()[0], "lightgrey"]

    # plot bar chart for network measures
    rects1 = ax.bar(r1, sorted_net_values, color=color[0], width=barWidth,
                    edgecolor='white', label='Network Measures')

    # plot bar chart for random network measures
    rects2 = ax.bar(r2, sorted_random_net_values, color=color[1], width=barWidth,
                    edgecolor='white', label='Random Network Measures')

    # autolabel each bar column with the value
    if label_bar:
        for rect in rects1+rects2:
            height = round(rect.get_height(),2)
            if height > 0:
                ax.annotate('{}'.format(height),                                # text - what to show
                            xy=(rect.get_x() + rect.get_width() / 2, height),   # (xy) coordinate where it should be
                            ha="center",                                        # ha - horizontal alignment
                            va="bottom",                                        # va - vertical alignment
                            size=14)                                            # size - the size of the text
            elif height < 0:
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            ha="center",
                            va="top",
                            size=14)

    # set abbreviations for measures
    abbreviation = {'assortativity': 'a', 'average_clustering': 'C',
                    'average_shortest_path_length': 'L',
                    'efficiency': 'E', 'modularity': 'M'}
    barsLabels = []
    for i in sorted_net_measures:
        barsLabels.append(abbreviation[i])

    # set the current tick locations and labels of the x-axis
    ax.set_xticks([r + barWidth/2 for r in range(len(r1))])
    ax.set_xticklabels(barsLabels)

    # make a line at y=0
    ax.axhline(0, linewidth=0.7, color='black')

    # set the number of bins to 5
    ax.locator_params(axis='y', nbins=5)

    # set labels for y axix
    ax.set_ylabel("Network Values")

    # create a legend
    ax.legend(fontsize="xx-small")

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


def plot_degree_dist(G, binomial_graph=True, figure_name=None, color=None):

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

    # set the seaborn style and context in the beginning!
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # calculate the degrees from the graph
    degrees = np.array(list(dict(G.degree()).values()))

    # calculate the Erdos Renyi graph from the main graph
    nodes = len(G.nodes())
    cost = G.number_of_edges() * 2.0 / (nodes*(nodes-1))    # probability for edge creation
    G_ER = nx.erdos_renyi_graph(nodes, cost)

    # calculate the degrees for the ER graph
    degrees_ER = np.array(list(dict(G_ER.degree()).values()))

    # create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # set the default colors of plotted values if not provided
    if color is None:
        color = [sns.color_palette()[0], "grey"]

    # if the user provided color not as a list of size 2 - show warning, use default colors
    if not isinstance(color, list) and len(color) == 2:
        warnings.warn("Please, provide a *color* parameter as a "
                      "python list object, e.g. [\"green\", \"pink\"]. "
                      "Right now the default colors will be used")
        color = ["#00C9FF", "grey"]

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
