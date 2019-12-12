#!/usr/bin/env python
import scona.make_corr_matrices as mcm
from scona.wrappers.corrmat_from_regionalmeasures import corrmat_from_regionalmeasures
from scona.wrappers.network_analysis_from_corrmat import network_analysis_from_corrmat

def standard_analysis(args):

    M = corrmat_from_regionalmeasures(args)
    
    network_analysis_from_corrmat(args, corrmat=M)

def groupwise_analysis(args):
    # run for true group assignments
    groupwise_bundle = GraphBundle(corrmat_from_regionalmeasures(args))
    groupwise_bundle.report_global_measures
    groupwise_bundle.report_rich_club

    # Now run for random group assignments
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)
    shuffle_list = []
    for i in range(args.n_shuffle):
        shuffle_list.append(
            GraphBundle.from_regional_measures(
                df,
                names,
                groupby=group_var,
                covars=covars_list,
                method=args.method,
                shuffle=True)))
        
    

def movingwindow_analysis(args):
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)
    
    moving_window_bundle = GraphBundle.from_regional_measures(
        df, names, covars=covars_list, method=args.method,
        windowby=args.window_var, window_size=args.window_size)

    shuffle_list = []
    for i in range(args.n_shuffle):
        shuffle_list.append(
            GraphBundle.from_regional_measures(
                df, names, covars=covars_list, method=args.method,
                windowby=args.window_var, window_size=args.window_size,
                shuffle=True))

    

