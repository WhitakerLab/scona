from nilearn import plotting
import networkx as nx
import numpy as np


def graph_to_nilearn_array(
        G,
        node_colour_att=None,
        node_size_att=None,
        edge_attribute="weight"):
    """
    Derive from G the necessary inputs for the `nilearn` graph plotting
    functions.
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"
    node_colour_att : str, optional
        index a nodal attribute to scale node colour by
    node_size_att : str, optional
        index a nodal attribute to scale node size by
    edge_attribute : str, optional
        index an edge attribute to scale edge colour by
    """
    node_order = sorted(list(G.nodes()))
    adjacency_matrix = nx.to_numpy_matrix(
        G,
        nodelist=node_order,
        weight=edge_attribute)
    node_coords = np.array([G._node[node]["centroids"] for node in node_order])
    if node_colour_att is not None:
        node_colour_att = [G._node[node][node_colour_att] for node
                           in node_order]
    if node_size_att is not None:
        node_size_att = [G._node[node][node_size_att] for node in node_order]
    return adjacency_matrix, node_coords, node_colour_att, node_size_att


def plot_connectome_with_nilearn(
        G,
        node_colour_att=None,
        node_colour="auto",
        node_size_att=None,
        node_size=50,
        edge_attribute="weight",
        edge_cmap="Spectral_r",
        edge_vmin=None,
        edge_vmax=None,
        output_file=None,
        display_mode='ortho',
        figure=None,
        axes=None,
        title=None,
        annotate=True,
        black_bg=False,
        alpha=0.7,
        edge_kwargs=None,
        node_kwargs=None,
        colorbar=False):
    """
    Plot a BrainNetwork using :func:`nilearn.plotting.plot_connectome()` tool.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"
    node_colour_att : str, optional
        index a nodal attribute to scale node colour by
    node_size_att : str, optional
        index a nodal attribute to scale node size by
    edge_attribute : str, optional
        index an edge attribute to scale edge colour by

    other parameters are passed to :func:`nilearn.plotting.plot_connectome()`
    """
    adjacency_matrix, node_coords, colour_list, size_list = graph_to_nilearn_array(G, node_colour=node_colour_att, node_size=node_size_att, edge_attribute=edge_attribute)

    if node_colour_att is not None:
        node_colour = [node_colour(x) for x in colour_list]
    if node_size_att is not None:
        node_size = [x*node_size for x in size_list]

    plotting.plot_connectome(
        adjacency_matrix, node_coords, node_color=node_colour,
        node_size=node_size, edge_cmap="Spectral_r", edge_vmin=edge_vmin,
        edge_vmax=edge_vmax, edge_threshold=0.01, output_file=output_file,
        display_mode=display_mode, figure=figure, axes=axes, title=title,
        annotate=annotate, black_bg=black_bg, alpha=alpha,
        edge_kwargs=edge_kwargs, node_kwargs=node_kwargs, colorbar=colorbar)


def view_connectome_with_nilearn(
        G,
        edge_attribute="weight",
        edge_cmap="Spectral_r",
        symmetric_cmap=True,
        edgewidth=6.0,
        node_size=3.0,
        node_colour_att=None,
        node_colour='black'):
    #   node_colour_list=None):
    """
    Plot a BrainNetwork using :func:`nilearn.plotting.view_connectome()` tool.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"
    node_colour_att : str, optional
        index a nodal attribute to scale node colour by
    edge_attribute : str, optional
        index an edge attribute to scale edge colour by

    other parameters are passed to :func:`nilearn.plotting.view_connectome()`
    """
    adjacency_matrix, node_coords, colour_list, z = graph_to_nilearn_array(
        G,
        edge_attribute=edge_attribute,
        node_colour=node_colour_att)
    return plotting.view_connectome(
        adjacency_matrix,
        node_coords,
        threshold=None,
        cmap=edge_cmap,
        symmetric_cmap=symmetric_cmap,
        linewidth=edgewidth,
        marker_size=node_size)
#    if colour_list is None:
#        colours = [node_colour for i in range(len(node_coords))]
#    else:
#        colours = np.array([node_colour(x) for x in colour_list])
#
#    connectome_info = plotting.html_connectome._get_markers(node_coords,
#                                                            colours)
#    connectome_info.update(plotting.html_connectome._get_connectome(
#        adjacency_matrix, node_coords, threshold=None, cmap=edge_cmap,
#        symmetric_cmap=symmetric_cmap))
#    connectome_info["line_width"] = edgewidth
#    connectome_info["marker_size"] = node_size
#    return plotting.html_connectome._make_connectome_html(connectome_info)


def view_markers_with_nilearn(
        G,
        node_size=5.,
        node_colouring='black'):
    """
    Plot nodes of a BrainNetwork using
    :func:`nilearn.plotting.view_connectome()` tool.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal
        attribute "centroids"
    node_colouring : str or list of str, default 'black'
        node_colouring will determine the colour given to each node.
        If a single string is given, this string will be interpreted as a
        a colour, and all nodes will be rendered in this colour.
        If a list of colours is given,
    """
    a, node_coords, colour_list, z = graph_to_nilearn_array(
        G,)
    if isinstance(node_colouring, str):
        colours = [node_colouring for i in range(len(node_coords))]
    elif colour_list is not None:
        colours = np.array([node_colouring(x) for x in colour_list])
    return plotting.view_markers(
        node_coords, colors=colours, marker_size=node_size)
