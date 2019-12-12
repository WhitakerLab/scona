from scona.classes import BrainNetwork, GraphBundle
from scona.make_corr_matrices import corrmat_from_regionalmeasures

def network_analysis_from_matrix(
        M,
        cost,
        n_rand,
        name="critical_network",
        seed=None,
        parcellation=None,
        centroids=None):
    
    # Initialise graph
    weighted_network = BrainNetwork(
        network=M,
        parcellation=parcellation,
        centroids=centroids)

    # Threshold graph
    binary_network = weighted_network.threshold(cost)

    # Calculate the modules, distance and hemispheric attributes
    # and the nodal measures
    binary_network.partition()
    binary_network.calculate_spatial_measures()
    binary_network.calculate_nodal_measures()
    

    # Create setup for comparing binary_network against random graphs
    # (note that this takes a bit of time because you're generating random
    # graphs)
    bundle = GraphBundle(graph_dict={name: binary_network})
    bundle.create_random_graphs(name, n_rand, seed=seed)
    
    # Add the small world coefficient to global measures
    small_world = bundle.report_small_world(network_name)

    for gname, network in bundle.items():
        network.graph['global_measures'].update(
            {"sw coeff against " + network_name: small_world[gname]})
 
    return bundle, binary_network.report_nodal_measures(), bundle.report_global_measures(), bundle.report_rich_club


def standard_analysis(
        df,
        names,
        covars=None,
        centroids=None,
        parcellation=None,
        method='pearson',
        name="critical_network",
        seed=None):

    M = corrmat_from_regionalmeasures(
        df,
        names,
        covars=covars,
        method=method)

    return network_analysis_from_matrix(
        M,
        cost,
        n_rand,
        name=name,
        seed=seed,
        parcellation=parcellation,
        centroids=centroids)

def groupwise_analysis(
        df,
        names,
        group_var,
        cost,
        covars=None,
        method='pearson',
        seed=None):
    # run first for true group assignments
    groupwise_bundle = GraphBundle.from_regional_measures(
        df,
        names,
        groupby=group_var,
        covars=covars,
        method=method)
    groupwise_bundle.threshold(cost)

    for i in range(args.n_shuffle):
        gb = GraphBundle.from_regional_measures(
            df,
            names,
            groupby=group_var,
            covars=covars,
            method=method,
            shuffle=True,
            seed=seed)
        gb.threshold(cost)
        gb.report_global_measures()

def moving_window_analysis(
        df,
        names,
        cost,
        window_var,
        window_size,
        covars=None,
        method='pearson',
        seed=None):

    moving_window_bundle = GraphBundle.from_regional_measures(
        df, names, covars=covars, method=method,
        windowby=window_var, window_size=window_size)
    moving_window_bundle.threshold(cost)
    

    shuffle_list = []
    for i in range(args.n_shuffle):
            gb = GraphBundle.from_regional_measures(
                df, names, covars=covars, method=method,
                windowby=window_var, window_size=window_size,
                shuffle=True, seed=seed)
            gb.threshold(cost)
