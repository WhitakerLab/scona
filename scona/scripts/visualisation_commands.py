#!/usr/bin/env python

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib as mpl
import os
import sys
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import itertools as it
import matplotlib.patches as mpatches


def rescale(fname, suff='png'):
    '''
    Journals generally like to make life easier for reviewers
    by sending them a manuscript that is not going to crash
    their computers with its size, so we're going to create
    a smaller version of the input figure (fname) that is
    8 inches wide at 200 dpi. It will be saved out in whatever
    format specified by the suff parameter, and the name
    will be the same as the original but with _LowRes appended
    '''

    from PIL import Image
    import numpy as np

    # Open the file and figure out what size it is
    img = Image.open(fname+'.'+suff)
    size = img.size

    # Calculate the scale factor that sets the width
    # of the figure to 1600 pixels
    scale_factor = 1600.0/size[0]

    # Apply this scale factor to the width and height
    # to get the new size
    new_size = (np.int(size[0]*scale_factor), np.int(size[1]*scale_factor))

    # Resize the image
    small_img = img.resize(new_size, Image.ANTIALIAS)

    # Define the output name
    new_name = ''.join([os.path.splitext(fname)[0],
                                            '_LowRes.',
                                            suff])

    # Save the image
    small_img.save(new_name, optimize=True, quality=95)

    # And you're done!


def view_corr_mat(corr_mat,
                  output_name,
                  cmap_name='RdBu_r',
                  cost=None,
                  bin=False):
    '''
    This is a visualisation tool for correlation matrices

    Parameters
    ----------
    corr_mat : :class:`pandas.DataFrame` or :class:`str` or :class:`numpy.array`
        corr_mat could be a DataFrame object to represent a correlation matrix
        or a string object - Path to the File, containing the matrix.
        or a numpy.array representing a correlation matrix
    output_name : :class:`str`
        the name of the file you want to save your visualization
        of correlation matrix to in.
    cmap_name : string or Colormap, optional
        A Colormap instance or registered colormap name.
        The colormap maps scalar data to colors. It is ignored for RGB(A) data.
        Defaults to 'RdBu_r'.

    Returns
    -------
        The visualization of the correlation matrix is saved in the file.

    '''

    # If cost is given then roughly threshold at that cost.
    # NOTE - this is not actually the EXACT network that you're analysing
    # because it doesn't include the minimum spanning tree. But it will give
    # you a good sense of the network structure.
    # #GoodEnough ;)

    if isinstance(corr_mat, str):
        M = np.loadtxt(corr_mat)                  # Read in the data
    elif isinstance(corr_mat, pd.DataFrame):
        M = corr_mat.to_numpy()                   # Convert the DataFrame to a NumPy array
    elif isinstance(corr_mat, np.ndarray):
        M = corr_mat                              # support numpy array as input to the function
    else:
        raise TypeError("corr_mat argument must be a 1)pandas.DataFrame object or 2) numpy.array or 3)a path to the file containing the matrix")

    if M.shape[0] != M.shape[1]:
        raise ValueError("The correlation matrix must be n x n, where n is the number of nodes")

    if cost:
        thr = np.percentile(M.reshape(-1), 100-cost)
        M[M<thr] = 0

        vmin=0
        vmax=1
        ticks_dict = { 'locations' : [ 0, 1 ],
                       'labels'    : [ '0', '1' ] }
    else:
        vmin=-1
        vmax=1
        ticks_dict = { 'locations' : [ -1, 0, 1 ],
                       'labels'    : [ '-1', '0', '1' ] }

    if bin:
        M[M>0] = 1

    # Create an axis
    fig, ax = plt.subplots(figsize=(6,5))
    ax.axis('off')

    # Show the network measures
    mat_ax = ax.imshow(M,
                        interpolation='none',
                        cmap=cmap_name,
                        vmin=vmin,
                        vmax=vmax)

    # Put a box around your data
    ax.add_patch(
     mpatches.Rectangle(
        (ax.get_xlim()[0], ax.get_ylim()[1]),
        ax.get_xlim()[1],
        ax.get_ylim()[0],
        fill=False,      # remove background
        color='k',
        linewidth=1) )

    # Add colorbar, make sure to specify tick locations to match desired tick labels
    cbar = fig.colorbar(mat_ax, ticks=ticks_dict['locations'])
    cbar.ax.set_yticklabels(ticks_dict['labels'])  # vertically oriented colorbar

    plt.tight_layout()

    # Save the picture
    fig.savefig(output_name, bbox_inches=0, dpi=100)

    plt.close(fig)
