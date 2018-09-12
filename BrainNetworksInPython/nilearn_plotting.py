from nilearn import plotting
import networkx as nx
import numpy as np


def graph_to_nilearn_array(G, node_colour=None, node_size=None, edge_attribute="weight"):
    """
    Create from G the necessary inputs for `nilearn` graph plotting functions.
    G : :class:`networkx.Graph`
        G should have nodal locations in MNI space indexed by nodal attribute "centroids"
    node_colour : str, optional
        index a nodal attribute to scale node colour by
    node_size : str, optional
        index a nodal attribute to scale node size by
    edge_attribute : str, optional
        index an edge attribute to scale edge colour by
    """
    node_order = sorted(list(G.nodes()))
    adjacency_matrix = nx.to_numpy_matrix(G, nodelist=node_order, weight=edge_attribute)
    node_coords = np.array([G._node[node]["centroids"] for node in node_order])
    if node_colour is not None :
        node_colour = [G._node[node][node_colour] for node in node_order]
    if node_size is not None:
        node_size = [G._node[node][node_size] for node in node_order]
    return adjacency_matrix, node_coords, node_colour, node_size


def plot_connectome_with_nilearn(G, node_colour_att=None, node_colour="auto", node_size_att=None, node_size=50, edge_attribute="weight", edge_cmap="Spectral_r", edge_vmin=None, edge_vmax=None, output_file=None, display_mode='ortho', figure=None, axes=None, title=None, annotate=True, black_bg=False, alpha=0.7, edge_kwargs=None, node_kwargs=None, colorbar=False):
    """
    Parameters
    ----------
    G : :class:`networkx.Graph`
    node_colour_att : str, optional
        Defines a nodal attribute.
    node_colour : str or :class:`matplotlib.colors.Colormap`, optional
        
    node_size_att : str, optional
        
    node_size : int, float, optional
        
    edge_attribute : str, optional
    edge_cmap : :class:`matplotlib.colors.Colormap`
        colormap used for representing `edge_attribute`.
    edge_vmin, edge_vmax : float, optional
        If not None, either or both of these values will be used to as the minimum and maximum values to color edges. If None are supplied the maximum absolute value within the given threshold will be used as minimum (multiplied by -1) and maximum coloring levels.
    output_file : str or None, optional
        The name of an image file to export the plot to. Valid extensions are .png, .pdf, .svg. If output_file is not None, the plot is saved to a file, and the display is closed.
    display_mode : str, optional
        Choose the direction of the cuts: ‘x’ - sagittal, ‘y’ - coronal, ‘z’ - axial, ‘l’ - sagittal left hemisphere only, ‘r’ - sagittal right hemisphere only, ‘ortho’ - three cuts are performed in orthogonal directions. Possible values are: ‘ortho’, ‘x’, ‘y’, ‘z’, ‘xz’, ‘yx’, ‘yz’, ‘l’, ‘r’, ‘lr’, ‘lzr’, ‘lyr’, ‘lzry’, ‘lyrz’.
    figure : int or :class:`matplotlib.figure`, optional
        Matplotlib figure used or its number. If None is given, a new figure is created.
    axes : :class:`matplotlib.axes` or tuple, optional
        The axes, or the coordinates (four tuple: (xmin, ymin, width, height)) in matplotlib figure space, of the axes used to display the plot.  If None, the complete figure is used.
    title : str, optional
        The title displayed on the figure.
    annotate : bool, optional
        If annotate is True, positions and left/right annotation are added to the plot.
    black_bg : bool, optional
        If True, the background of the image is set to be black. If you wish to save figures with a black background, you will need to pass “facecolor=’k’, edgecolor=’k’” to matplotlib.pyplot.savefig.
    alpha : float between 0 and 1
        Alpha transparency for the brain schematics.
    edge_kwargs : dict
        will be passed as kwargs for each edge matlotlib Line2D.
    node_kwargs : dict
        will be passed as kwargs to the plt.scatter call that plots all the nodes in one go
    colorbar : bool, optional
        If True, display a colorbar on the right of the plots. By default it is False.
    """
    adjacency_matrix, node_coords, colour_list, size_list = graph_to_nilearn_array(G, node_colour=node_colour_att, node_size=node_size_att, edge_attribute=edge_attribute)

    if node_colour_att is not None:
        node_colour = [node_colour(x) for x in colour_list]
    if node_size_att is not None:
        node_size = [x*node_size for x in size_list]

    plotting.plot_connectome(adjacency_matrix, node_coords, node_color=node_colour, node_size=node_size, edge_cmap="Spectral_r", edge_vmin=edge_vmin, edge_vmax=edge_vmax, edge_threshold=None, output_file=output_file, display_mode=display_mode, figure=figure, axes=axes, title=title, annotate=annotate, black_bg=black_bg, alpha=alpha, edge_kwargs=edge_kwargs, node_kwargs=node_kwargs, colorbar=colorbar)


def view_connectome_with_nilearn(G, edge_attribute="weight", edge_cmap="Spectral_r", symmetric_cmap=True, edgewidth=6.0, node_size=3.0, node_colour_att=None, node_colour='black', node_colour_list=None):
    """
    Parameters
    ----------
    G : :class:`networkx.Graph`
    edge_attribute : str, optional
        index an edge attribute to scale edge colour by
    node_colour_att : str, optional
        index a nodal attribute to scale node colour by.
    node_colour : str or :class:`matplotlib.colors.Colormap`, optional
    node_colour_list : array type, optional
        pass a list of colours (formats accepted by matplotlib, see
        https://matplotlib.org/users/colors.html#specifying-colors).
        The list should be ordered as so that `node_colour_list[i]` 
        is the colour of node `node_order[i]` where
        `node_order=sorted(list(G.nodes()))`
    edge_cmap : str or :class:`matplotlib.colormap`, optional
    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).
    edgewidth : float, optional (default=6.)
        Width of the lines that represent edges.
    node_size : float, optional (default=3.)
        Size of node markers.
    """
    adjacency_matrix, node_coords, colour_list, z = graph_to_nilearn_array(G, edge_attribute=edge_attribute, node_colour=node_colour_att)
    return plotting.view_connectome(adjacency_matrix, node_coords, threshold=None, cmap=edge_cmap, symmetric_cmap=symmetric_cmap, linewidth=edgewidth, marker_size=node_size)
#    if colour_list is None:
#        colours = [node_colour for i in range(len(node_coords))]
#    else:
#        colours = np.array([node_colour(x) for x in colour_list])
#
#    connectome_info = plotting.html_connectome._get_markers(node_coords, colours)
#    connectome_info.update(plotting.html_connectome._get_connectome(
#        adjacency_matrix, node_coords, threshold=None, cmap=edge_cmap,
#        symmetric_cmap=symmetric_cmap))
#    connectome_info["line_width"] = edgewidth
#    connectome_info["marker_size"] = node_size
#    return plotting.html_connectome._make_connectome_html(connectome_info)


def view_markers_with_nilearn(G, colours=None, node_size=5., node_colour_att=None, node_colour='black', node_colour_list=None):
    a, node_coords, colour_list, z = graph_to_nilearn_array(G, node_colour=node_colour_att)
    if colour_list is None:
        colours = [node_colour for i in range(len(node_coords))]
    else:
        colours = np.array([node_colour(x) for x in colour_list])
    return plotting.view_markers(node_coords, colors=colours, marker_size=node_size)