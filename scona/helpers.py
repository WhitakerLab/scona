import warnings
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


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

    # if path_name ends with "/" - do not save, e.g. "/home/user/dir1/dir2/"
    if os.path.basename(path_name):

        # get the directory path (exclude file_name in the end of path_name)
        dir_path = os.path.dirname(path_name)

        # if dir *path_name* does not exist - create
        # os.path.dirname(path_name) - make sure not to create a dir if path_name="myfile.png"
        if not os.path.exists(dir_path) and os.path.dirname(path_name):
            warnings.warn("The path - {} does not exist. But we will create this "
                          "directory for you and store the figure there.".format(dir_path))

            # get the dirname to create "/home/dir1/myfile.png" -> "/home/dir1"
            dir_create = os.path.dirname(path_name)
            os.makedirs(dir_create)
    else:
        warnings.warn("The location name - {} you gave us ends with \"/\". "
                      "Please give us the name of the file like /home/dir1/myfile.png "
                      "or to save in the current directory just the name *myfile.png*".format(path_name))
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

    cmap_name : str or Colormap instance
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

    # If vmin or vmax not passed, calculate the min and max of the column (measure)
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
            warnings.warn("ValueError: Colormap {} is not recognized. ". format(cmap_name) +
                            "Default colormap jet will be used.")
            cmap = mpl.cm.get_cmap("jet")

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
        The label for the colorbar. Name of data values this colorbar represents.

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

    # put a colorbar in a specified axes, and make colorbar for a given colormap
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
