import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rich_club(rc_coef, rc_coef_rand, figure_name=None, color=None,
                   x_max=None, y_max=None):
    """
    This is a visualisation tool for plotting the rich club values per degree
    along with the random rich club values created from a random network
    with a preserved degree distribution.

    Parameters
    ----------
    rc_coef : dict
        rich club coefficient values
    rc_coef_rand : :dict
        random rich club coefficient values
    figure_name : str, optional
        path to the file to store the created figure in (e.g. "/home/Desktop/name")
        or to store in the current directory include just a name ("fig_name");
    color : list of 2 strings, optional
        where the 1st string is a color for rich club values and 2nd - for random
        rich club values. You can specify the color using an html hex string
        (e.g. color =["#06209c","#c1b8b1"]) or you can pass an (r, g, b) tuple,
        where each of r, g, b are in the range [0,1]. Finally, legal html names
        for colors, like "red", "black" and so on are supported.
    x_max : int, optional
        the max length of the x-axis of the plot
    y_max : int, optional
        the max length of the y-axis of the plot

    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name file.

    """

    # get the degrees
    degree = list(rc_coef.keys())

    # get the rich club coefficient values
    rc = list(rc_coef.values())

    # get the random rich club coefficient values
    rc_rand = list(rc_coef_rand.values())

    # create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # set the seaborn context and style
    sns.set(style="white")
    sns.set_context("poster", font_scale=1)

    # set the default colors of plotted values if not provided
    if color is None:
        color = [sns.color_palette()[0], "lightgrey"]

    # plot the rich club values and random rich club values
    ax = sns.lineplot(x=degree, y=rc, label="rich-club coefficient", color = color[0])
    ax = sns.lineplot(x=degree, y=rc_rand, label="random rich-club coefficient", color = color[1])

    # set the max values of x & y - axis if not provided
    if x_max is None:
        x_max = max(degree)

    if y_max is None:
        y_max = max(rc) + 0.1   # let y-axis be longer -> looks better

    # set the x and y axis limits
    ax.set_xlim((0, x_max))
    ax.set_ylim((0, y_max))

    # set the number of bins to 4
    ax.locator_params(nbins=4)

    # set the x and y axis labels
    ax.set_xlabel("Degree")
    ax.set_ylabel("Rich Club")

    # create a legend
    ax.legend(fontsize="x-small")

    # remove the top and right spines from plot
    sns.despine()

    # adjust subplot params so that the subplot fits in to the figure area
    plt.tight_layout()

    # display the figure
    plt.show()

    # save the figure if the location-to-save is provided
    if figure_name:
        fig.savefig(figure_name, bbox_inches=0, dpi=100)

        plt.close(fig)


def plot_network_measures(network_measures, rand_network_measures, figure_name=None,
                          color=None, labelBar=True):
    """
    This is a visualisation tool for plotting network measures values
    along with the random network values values created from a random network.

    Parameters
    ----------
    network_measures : dict
        real network measures
        Note: the dict could be obtained from calculate_global_measures()
    rand_network_measures : :dict
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
    labelBar : bool, optional
        if True show a value on top of each bar. Note - the value is rounded to
         2 decimals. by default - true.

    Returns
    -------
        Plot the Figure and if figure_name provided, save it in a figure_name file.

    """

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

    # set seaborn style and context
    sns.set_style('white')
    sns.set_context("poster", font_scale=1)

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
    if labelBar:
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
        fig.savefig(figure_name, bbox_inches=0, dpi=100)

        plt.close(fig)
