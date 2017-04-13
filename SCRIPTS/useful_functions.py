#!/usr/bin/env python


def read_in_df(data_file, aparc_names):
    '''
    A very useful command for NSPN behavmerge data frames
    Beware though - this is quite specific and there are 
    a few versions floating around! Be careful
    '''
    import pandas as pd
    import numpy as np
    import os
    
    # Read in the data file
    df = pd.read_csv(data_file, sep=',')
    
    # Only keep the first scan!
    df = df.loc[df.occ==0, :]

    # Strip "thickness" or "thicknessstd" from the column
    # names so they match with the aparc_names names
    data_cols = [ x.replace('_{}'.format('thicknessstd'), '') for x in df.columns ]
    df.columns = data_cols
    data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df.columns ]
    df.columns = data_cols
    
    # Define a few variables you'll want in the data frame
    df['ones'] = df['age_scan'] * 0 + 1
    df['age'] = df['age_scan']
    
    df['Global'] = df[aparc_names].mean(axis=1)
    df['Global_std'] = df[aparc_names].mean(axis=1)

    # If there is a corresponding standard deviation
    # file then read in the standard deviation
    if 'mean' in data_file:
        std_data_file = data_file.replace('mean', 'std')
    else:
        std_data_file = data_file.replace('thickness', 'thicknessstd')
    
    if os.path.isfile(std_data_file):
    
        # Repeating the steps really
        # Read in the file
        df_std = pd.read_csv(std_data_file, sep=',')
        # Only keep the first occ
        df_std = df_std.loc[df_std.occ==0, :]
        # Change the names so they match up
        data_cols = [ x.replace('_{}'.format('thicknessstd'), '') for x in df_std.columns ]
        df_std.columns = data_cols
        data_cols = [ x.replace('_{}'.format('thickness'), '') for x in df_std.columns ]
        df_std.columns = data_cols
        
        # Now write the std across all aparc names into the original data frame
        # by averaging the variances
        df['Global_std'] = np.sqrt(np.average(df_std[aparc_names]**2, axis=1))
    
    # Convert the values to floats
    df[aparc_names] = df[aparc_names].astype('float')
    
    # If this is an MT, R2s, synthetic, MD, L1 or L23 file
    # then you have to divide the values by 1000
    # However there have been problems here in the past with
    # mixing multiplied with non-multiplied values
    # so we'll actually just check for values greater than a
    # reasonable maximum and divide those ones.
    cols_list = aparc_names+['Global']+['Global_std']
    if 'MT' in os.path.basename(data_file):
        df.loc[df['Global']>50, cols_list] = df.loc[df['Global']>50, cols_list]/1000.0
    if 'synthetic' in os.path.basename(data_file):
        df.loc[df['Global']>50, cols_list] = df.loc[df['Global']>50, cols_list]/1000.0
    if 'R2s' in os.path.basename(data_file):
        df.loc[df['Global']>1, cols_list] = df.loc[df['Global']>1, cols_list]/1000.0
    if 'L1' in os.path.basename(data_file):
        df.loc[df['Global']>0.01, cols_list] = df.loc[df['Global']>0.01, cols_list]/1000.0
    if 'L23' in os.path.basename(data_file):
        df.loc[df['Global']>0.01, cols_list] = df.loc[df['Global']>0.01, cols_list]/1000.0
    if 'MD' in os.path.basename(data_file):
        df.loc[df['Global']>0.01, cols_list] = df.loc[df['Global']>0.01, cols_list]/1000.0
    
    return df

    
def residuals(x, y):
    '''
    A useful little function that correlates
    x and y together to give their residual
    values. These can then be used to calculate
    partial correlation values
    '''
    import numpy as np
    
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    A = np.vstack([x, np.ones(x.shape[-1])]).T
    B = np.linalg.lstsq(A, y)[0]
    m = B[:-1]
    c = B[-1]
    pre = np.sum(m * x.T, axis=1) + c
    res = y - pre
    return res
    
