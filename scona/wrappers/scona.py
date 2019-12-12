import scona
import numpy as np
import os
import scona.make_corr_matrices as mcm
from scona.scripts.useful_functions import read_in_data, \
    write_out_measures, list_from_file

def corrmat_from_regionalmeasures(args):
    '''
    Read in regional measures, names and covariates files to compute
    and return a correlation matrix and write it to output_name.

    Parameters
    ----------
        regional_measures_file : str
            A csv file of regional measures data containing any required
            brain regions and covariates as columns. 
            The first row of regional_measures should be column headings,
            with subsequent rows corresponding to individual subjects.
        names_file : str, optional
            a text file containing names of brain regions. One name
            per line of text. These names will specify the columns
            in the regional_measures_file to correlate through. 
            Failing to pass an argument either to both names_file and
            names_list will cause an exception to be raised.
        names_list : list, optional
            a list of names of brain regions. These names will specify
            the columns in the regional_measures_file to correlate
            through. 
            Failing to pass an argument to both names_file and
            names_list will cause an exception to be raised.
        covars_file : str, optional
            a text file containing a list of covariates to account
            for. One covariate per line of text. These covariates 
            should index columns in regional_measure_file. 
        output_name : str, optional
            a file name to save output matrix to. 
            If the output directory does not yet exist it will be
            created.
            If an argument is passed to group_var the group correlation
            matrices will be written to a file with the corresponding
            group coding appended.
            If no name is passed the output matrix (or matrices) will
            be returned but not saved.
        group_var : str, optional
            This variable can be used to specify a column in 
            regional_measures_file containing the group coding. A
            correlation matrix will be produced for each patient group.
            

    Returns
    -------
    :class:`pandas.DataFrame or dict
        A correlation matrix, or, if a group_var is passed, a dictionary of group codings mapping to
        correlation matrices. 
    '''
    # Read in the data
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)

    if args.group_var is None:
        # create correlation matrix
        M = mcm.corrmat_from_regionalmeasures(
            df, names, covars=covars_list, method=args.method)
        if args.output_name is not None:
            # Save the matrix
            mcm.save_mat(
                M,
                os.path.join(args.output_dir, args.output_name))
        return M

    else:
        # create matrices
        matrix_by_group = mcm.corrmat_by_group(
            df,
            names,
            args.group_var,
            covars=covars_list,
            method=args.method)
        # if necessary write out matrices
        if args.output_name is not None:
            for group_code, M in matrix_by_group:
                mcm.save_mat(
                    M,
                    os.path.join(args.output_dir,
                                 str(group_code)+args.output_name))
        return matrix_by_group


def network_analysis_from_corrmat(args, corrmat=None):
    # Read in the data
    if corrmat is None:
        M, names, a, centroids = read_in_data(
            args.corrmat_file,
            args.names_file,
            centroids_file=args.centroids_file,
            data_as_df=False)
        
    else:
        M = corrmat
        names = list_from_file(args.names_file)
        if args.centroids_file is not None:
            centroids = list(np.loadtxt(args.centroids_file))
        else:
            centroids = None

    # if possible, name network after corrmat_file
    if args.corrmat_file is None:
        network_name = ""
    else:
        network_name = args.corrmat_file

    # run standard analysis
    bundle, nodal_df, global_df, rc_df = scona.network_analysis_from_matrix(
        M, args.cost, args.n_rand, name=network_name, seed=args.seed, parcellation=names, centroids=centroids)

    # write out each of the outputs
    nodal_name = 'NodalMeasures_{}_cost{:03.0f}.csv'.format(
        network_name, args.cost)
    write_out_measures(
        nodal_df, args.output_dir, nodal_name, first_columns=['name'])

    global_name = 'GlobalMeasures_{}_cost{:03.0f}.csv'.format(
        network_name, args.cost)
    write_out_measures(
        global_df, args.output_dir, global_name, first_columns=[network_name])
    
    rc_name = 'rich_club_{}_cost{:03.0f}.csv'.format(
        network_name, args.cost)
    write_out_measures(
        rc_df, args.output_dir, rc_name, first_columns=['degree', network_name])

def standard_analysis(args):

    M = corrmat_from_regionalmeasures(args)
    
    network_analysis_from_corrmat(args, corrmat=M)

def groupwise_analysis(args):
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)
    
    scona.analyses.groupwise_analysis(
        df,
        names,
        args.cost,
        args.group_var,
        covars=covars_list,
        method=args.method,
        seed=args.seed)
        
    

def movingwindow_analysis(args):
    df, names, covars_list, *a = read_in_data(
        args.regional_measures_file,
        args.names_file,
        covars_file=args.covars_file)

    scona.analyses.moving_window_analysis(
        df,
        names,
        args.cost,
        args.window_var,
        args.window_size,
        covars=covars_list,
        method=args.method,
        seed=args.seed)

    

