#!/usr/bin/env python

def make_corr_matrices(corrmat_dir, covars_dict, ct_data_file, aparc_names):
    '''
    A function that makes all the required correlation matrices
    for the named regions (aparc_names) in the cortical thickness data
    file accounting for each of the combinations of covariates in covars_dict.

    If the matrices do not already exist they are saved as text files in
    corrmat_dir. If they do exist then they're read in from those files.

    The function returns a dictionary of correlation matrices.
    '''
    #==========================================================================
    # IMPORTS
    #==========================================================================
    import os
    import numpy as np

    from useful_functions import read_in_df

    #==========================================================================
    # Print to screen what you're up to
    #==========================================================================
    print "--------------------------------------------------"
    print "Making or loading correlation matrices"

    #==========================================================================
    # Create an empty dictionary
    #==========================================================================
    mat_dict = {}

    #==========================================================================
    # Loop through all the covariates and age groups
    #==========================================================================
    for covars_name, covars_list in covars_dict.items():

        for age_group in [ 'ALL', 'ADULT' ]:

            key = 'CT_{}_COVARS_{}'.format(age_group, covars_name.upper())

            mat_name = os.path.join(corrmat_dir,
                                        'COVARS_{}'.format(covars_name.upper()),
                                        'Mat_CT_Corr_{}.txt'.format(age_group))

            print '    {}'.format(key)

            #======================================================================
            # If it doesn't already exist, then make it
            #======================================================================
            if not os.path.isfile(mat_name):
                # Read in the data
                df_ct = read_in_df(ct_data_file, aparc_names)

                # Filter the data if necessary
                if age_group == 'ADULT':
                    df_ct = df_ct.loc[df_ct['age_scan']>20, :]

                # Make the correlation matrix
                # and save into the mat_dict
                mat_dict[key] = create_mat(df_ct, aparc_names, covars_list)[1]

                # Write to text file to speed this up
                # in the future (see below!)
                save_mat(mat_dict[key], mat_name)

            #======================================================================
            # Otherwise just load it into the dictionary
            #======================================================================
            else:
                mat_dict[key] = np.loadtxt(mat_name)

    return mat_dict


def create_residuals_df(df, names, covars_list):
    '''
    df is a pandas data frame that contains the data you're going
      to correlate.
    names is a list of regions you care about.
    covars_list is a list of columns in df that you want to
      "correct for" before correlating the regions.
    '''
    import numpy as np
    import pandas as pd
    from useful_functions import residuals

    # Make a new data frame that will contain
    # the residuals for each column after correcting for
    # the covariates in covars
    df_res = df.copy()

    # Create your covariates array
    if len(covars_list) > 1:
        x = np.vstack([df[covars_list]])
    elif len(covars_list) == 1:
        x = df[covars_list]
    else:
        x = np.ones_like(df.iloc[:, 0])

    # Calculate the residuals
    for name in names:
        df_res.loc[:, name] = residuals(x.T, df.loc[:, name])

    # Return the residuals data frame
    return df_res


def create_corrmat(df, names, covars_list):
    '''
    This is a function designed to replace the old create_mat function.

    df is the data frame containing regional values for each paticipant.
    names is a list of regions you care about.
    covars_list is a list of measures (in df) that you want to "correct for".
    '''
    # Correct for covariates you want to control for
    df_res = create_residuals_df(df, names, covars_list)

    # Create the correlation matrix
    M = df_res.loc[:, names].corr().values

    return M


def create_mat(df, aparc_names, covar, demean=False):
    '''
    df contains the data you're going to correlate
    aparc_names are all the regions you care about
    covar needs to be either a column in df OR a
    list of columns
    '''
    import numpy as np
    from scipy.stats import pearsonr
    import pandas as pd

    from useful_functions import residuals

    # Create the very fast correlation matrix between these values
    mat_corr = df[aparc_names].corr().iloc[:,:]

    # Create a matrix of ones that is the same shape
    mat_corr_covar = np.ones_like(mat_corr)

    # Create your covariates array
    if len(covar) > 1:
        x = np.vstack([df[covar]])
    else:
        x = df[covar]

    # Only run the correlations for the upper triangle of the
    # matrix as it is symmetric (we do not have directed
    # connections
    triu_i, triu_j = np.triu_indices(len(aparc_names))

    for i, j in zip(triu_i, triu_j):
        if i%20 == 0 and j == len(aparc_names)-1:
            print 'Processing row {}'.format(i)

        # Calculate the residuals for the two
        # regions you care about
        res_i = residuals(x.T, df[aparc_names[i]])
        res_j = residuals(x.T, df[aparc_names[j]])

        # Correlate these residuals together
        mat_corr_covar[i, j] = pearsonr(res_i, res_j)[0]

    # Reflect the matrix to fill in the lower triangle
    mat_corr = mat_corr * mat_corr.T
    mat_corr_covar = mat_corr_covar * mat_corr_covar.T

    return mat_corr, mat_corr_covar


def save_mat(M, M_text_name):
    '''
    A very little function that saves matrices
    as a text file
    '''
    import numpy as np
    import os

    # Check to see if the output directory
    # exists, and make it if it does not
    dirname = os.path.dirname(M_text_name)

    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    # Save the matrix as a text file
    np.savetxt(M_text_name,
                   M,
                   fmt='%.5f',
                   delimiter='\t',
                   newline='\n')
