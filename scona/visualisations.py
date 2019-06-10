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
        color = [(0.3,0.45,0.7), "lightgrey"]

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
