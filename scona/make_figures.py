#!/usr/bin/env python

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib as mpl
import matplotlib.patches as mpatches
import nilearn as nl


def view_corr_mat(corr_mat_file, output_name, cmap_name='RdBu_r', cost=None, bin=False):
    
    # Read in the data
    M = np.loadtxt(corr_mat_file)

    # If cost is given then roughly threshold at that cost.
    # NOTE - this is not actually the EXACT network that you're analysing
    # because it doesn't include the minimum spanning tree. But it will give
    # you a good sense of the network structure.
    # #GoodEnough ;)

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

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(mat_ax, ticks=ticks_dict['locations'])
    cbar.ax.set_yticklabels(ticks_dict['labels'])  # vertically oriented colorbar

    plt.tight_layout()

    # Save the picture
    fig.savefig(output_name, bbox_inches=0, dpi=100)

    plt.close(fig)


def get_anatomical_layouts(G):
    '''
    This code takes in a BrainNetwork that has x, y, z coordinates and
    integer node labels (0 to n-1) for n nodes and returns three dictionaries
    containing appropriate pairs of coordinates for sagittal, coronal and
    axial slices.
    '''

    axial_dict = {}
    sagittal_dict = {}
    coronal_dict = {}

    for node in G.nodes:
        nx.set_node_attribute(node, name="axial", value=np.array())
        axial_dict[node] = np.array([df['x'].loc[df['node']==node].values[0],
                                        df['y'].loc[df['node']==node].values[0]])
        coronal_dict[node] = np.array([df['x'].loc[df['node']==node].values[0],
                                        df['z'].loc[df['node']==node].values[0]])
        sagittal_dict[node] = np.array([df['y'].loc[df['node']==node].values[0],
                                        df['z'].loc[df['node']==node].values[0]])

    return axial_dict, sagittal_dict, coronal_dict

def axial_layout(x, y, z):
    return np.array(x, y)


def sagittal_layout(x, y, z):
    return np.array(x, z)


def coronal_layout(x, y, z):
    return np.array(y,z)


def anatomical_layout(x, y, z, orientation='sagittal'):
    if orientation == 'sagittal':
        return sagittal_layout(x, y, z)
    if orientation == 'axial':
        return axial_layout(x, y, z)
    if orientation == 'coronal':
        return coronal_layout(x, y, z)
    else:
        raise ValueError("{} is not recognised as an anatomical layout. orientation values should be one of 'sagittal', 'axial' or 'coronal'.".format(orientation))


def plot_anatomical_network(G, measure='module', orientation='sagittal', cmap_name='jet_r', continuous=False, vmax=None, vmin=None, sns_palette=None, edge_list=None, edge_color='k', edge_width=0.2, node_list=None, rc_node_list=[], node_shape='o', rc_node_shape='s', node_size=500, node_size_list=None, figure=None, ax=None):
    '''
    Plots each node in the graph in one of three orientations
    (sagittal, axial or coronal).
    The nodes are sorted according to the measure given
    (default value: module) and then plotted in that order.
    '''
    
    if edge_list is None:
        edge_list = list(G.edges())

    if node_list is None:
        node_list = G.nodes()
        node_list = sorted(node_list)

    # Put the measures you care about together
    # in a data frame
    fields = ['degree','module','x','y','z']
    if measure not in fields:
        fields.append(measure)

    # Add in a node index which relates to the node names in the graph
    df['node'] = range(len(df['degree']))

    # Then use these node values to get the appropriate positions for each node
    pos = {node: anatomical_layout(
        G[node]['x'], G[node]['y'], G[node]['z'], orientation=orientation)
           for node in G.nodes}

    # Create a colors_list for the nodes
    colors_list = setup_color_list(df,
                                    cmap_name=cmap_name,
                                    sns_palette=sns_palette,
                                    measure=measure,
                                    vmin=vmin,
                                    vmax=vmax,
                                    continuous=continuous)

    # If the node size list is none then
    # it'll just be the same size for each node
    if node_size_list is None:
        node_size_list = [ node_size ] * len(df['degree'])

    # If you have no rich club nodes then all the nodes will
    # have the same shape
    node_shape_list = [ node_shape ] * len(df['degree'])
    # If you have set rich nodes then you'll need to replace
    # those indices with the rc_node_shape
    for i in rc_node_list:
        node_shape_list[i] = 's'

    # We're going to figure out the best way to plot these nodes
    # so that they're sensibly on top of each other
    sort_dict = {}
    sort_dict['axial'] = 'z'
    sort_dict['coronal'] = 'y'
    sort_dict['sagittal'] = 'x'

    node_order = np.argsort(df[sort_dict[orientation]]).values

    # Now remove all the nodes that are not in the node_list
    node_order = [ x for x in node_order if x in node_list ]

    # If you've given this code an axis and figure then use those
    # otherwise just create your own
    if not ax:
        # Create a figure
        fig_size_dict = {}
        fig_size_dict['axial'] = (9,12)
        fig_size_dict['sagittal'] = (12,8)
        fig_size_dict['coronal'] = (9,8)

        fig, ax = plt.subplots(figsize=fig_size_dict[orientation])

        # Set the seaborn context and style
        sns.set(style="white")
        sns.set_context("poster", font_scale=2)

    else:
        fig = figure

    # Start by drawing in the edges:
    nx.draw_networkx_edges(G,
                            pos=pos,
                            edgelist=edge_list,
                            width=edge_width,
                            edge_color=edge_color,
                            ax=ax)

    # And then loop through each node and add it in order
    for node in node_order:
        nx.draw_networkx_nodes(G,
                                pos=pos,
                                node_color=colors_list[node],
                                node_shape=node_shape_list[node],
                                node_size=node_size_list[node],
                                nodelist=[node],
                                with_labels=False,
                                ax=ax)

    axis_limits_dict = {}
    axis_limits_dict['axial'] = [ -70, 70, -105, 70]
    axis_limits_dict['coronal'] = [ -70, 70, -45, 75 ]
    axis_limits_dict['sagittal'] = [ -105, 70, -45, 75 ]

    ax.set_xlim(axis_limits_dict[orientation][0],axis_limits_dict[orientation][1])
    ax.set_ylim(axis_limits_dict[orientation][2],axis_limits_dict[orientation][3])
    ax.axis('off')

    return ax