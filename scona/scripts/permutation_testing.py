#!/usr/bin/env python

# This is a collection of tools to run permutation tests

def permutation_ols(x_orig, y_orig, covars_orig=[], n_perm=1000, categorical=True):
    '''
    Define a permuation test for multiple regression
    in which we first calculate the real model fit,
    and then extract the slope from 1000 random shuffles
    of the dependent variable. Note that it is only the 
    y variable that is shuffled, all the x data and 
    covariates remain the same.
    
    If categorical is set to True then you'll conduct the
    permutation test on the model's F statistic, otherwise
    on the slope of the x variable.
    
    x_orig, y_orig are 1D numpy arrays with n data points
    covars_orig is a list of 1D numpy arrays with n data points
    corresponding to each of the covariates you'd like to
    control for.
    '''
    
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    
    # Make a copy of the original data
    # because the shuffle command does this in place
    # and we don't want any mistakes!
    x = np.copy(x_orig)
    y = np.copy(y_orig)
    
    # Create the data frame
    df =  pd.DataFrame({'x' : x,
                        'y' : y})
                            
    # Create your formula
    if categorical:
        formula = 'y ~ C(x)'
    else:    
        formula = 'y ~ x'

    # Add in the covariates
    for i, covars in enumerate(covars_orig):
        if not np.std(covars) == 0.0:
            df['c_{}'.format(i)] = covars
            formula += ' + c_{}'.format(i)
    
    # Fit the model
    model = sm.OLS.from_formula(formula, df)
    results = model.fit()
    
    # Get the real measure you care about
    # m for continuous regression,
    # F for categorical variables
    if categorical:
        m = results.fvalue
    else:
        m = results.params['x']
    
    # Create an m_array that will hold the shuffled
    # slope values
    m_array = np.ones([n_perm])
    
    # Now simply correlate the residuals of x and y
    # after correcting for covars n_perm times with
    # shuffled data
    for i in range(n_perm):
    
        if i%10 == 5 : print i,
        
        np.random.shuffle(y)
        
        permutation_dict = permutation_correlation(x, y, covars_orig)
        
    if n_perm > 1 : print ''
    
    return results, permutation_dict['perm_p']
    
    

def regional_linregress(df, x, names, covars=[], n_perm=1000, categorical=False):
    '''
    A function that calls a multiple regression model repeatedly for
    all variable names passed (as names) in the data frame as the dependent
    variable, with the x column as the independent variable and the 
    names in covars as covariates.
    
    INPUTS: 
        df ------------- pandas data frame
        x -------------- independent variable name (must be column in df)
        names ---------- list of variable names (columns in df) to loop
                           through as dependent variables (ys) for the regression
        covars --------- list containing variable names that should be controlled for
                           (must be columns in df and the same for all 
                           dependent variables)
                           Default value: [ ]
        n_perm --------- number of permutations for permutation testing
                           Default value: 1000
    
    RETURNS:
        m_array ------------------ numpy array containing slopes for each region
        c_array ------------------ numpy array containing intercepts (at 0) for each region
        c14_array ---------------- numpy array containing intercepts at 14 for each region
        r_array ------------------ numpy array containing partial r (correcting for covariates) for each region
        p_array ------------------ numpy array containing raw p values for each region
        perm_p_array ------------- numpy array containing raw permutation p values for each region
        p_fdr_array -------------- numpy array containing fdr corrected p values for each region
        perm_p_fdr_array --------- numpy array containing fdr corrected permutation p values for each region
        m_masked_p_array --------- numpy array containing the slope values for regions which
                                     are indivudially significant, otherwise -99 markers
        m_masked_perm_p_array ---- numpy array containing the slope values for regions which
                                     are indivudially significant according to permutation
                                     testing, otherwise -99 markers
        m_fdr_masked_array ------- numpy array containing the slope values for regions which
                                     pass fdr correction otherwise -99 markers
        m_perm_fdr_masked_array -- numpy array containing the slope values for regions which
                                     pass fdr correction according to permutation testing,
                                     otherwise -99 markers
    '''
    #----------------------------------------------------------------
    # Import what you need
    from statsmodels.sandbox.stats.multicomp import fdrcorrection0 as fdr
    import numpy as np
    from scipy.stats import linregress
    
    #----------------------------------------------------------------
    # Set up an empty dictionary to save all these values
    regional_linregress_dict = {}
    
    #----------------------------------------------------------------
    # Set up your covars_list
    # This should contain all the data for each covar as a different
    # element in the list
    # You're going to save the index at which wbic appears in the list
    # because you'll need to add the param for this covar to the 
    # intercept value later to get a value for a woman (if male is 
    # included in the covariates list) at wbic.
    # (You don't have to correct your intercept for being a woman 
    # because the male covariate is coded as 0 for women, but you do
    # have to correct for wbic because there 0 represents CBU)
    covars_list = []
    wbic_i = None
    
    for i, covar in enumerate(covars):
        covars_list += [df[covar].values]
        if covar == 'wbic':
            wbic_i = i
    
    #----------------------------------------------------------------
    # Set up some empty arrays to contain:
    #    m: slope of the regression line
    #    c: intercept as x = 0 + the parameter estimate
    #                          for the wbic scanner location if passed
    #    c14: intercept when x = 14 (c + 14 * m)
    #    r: partial r
    #    p: p from the ols regression (for x)
    #    perm_p: p from the permutation test
    m_array = np.ones(len(names))
    c_array = np.ones(len(names))
    c14_array = np.ones(len(names))
    r_array = np.ones(len(names))
    p_array = np.ones(len(names))
    perm_p_array = np.ones(len(names))

    #----------------------------------------------------------------
    # Loop through all the regions and first regress out the
    # covariates and then record m, c, r, p and perm_p
    # for each region
    for i, roi in enumerate(names):
    
        # Run the permutation test
        linregress_dict = permutation_correlation(df[x].values,
                                                    df[roi].values,
                                                    covars_orig=covars_list, 
                                                    n_perm=n_perm)
        
        # Run the regular ols to get the correct intercept values
        results, c, c14 = ols_correlation(df[x].values,
                                                    df[roi].values, 
                                                    covars=covars_list,
                                                    wbic_covars_index=wbic_i)
        
        # Add these values to the linregress_dict
        # (which means overwriting 'c' and adding in 'c14')
        linregress_dict['c'] = c
        linregress_dict['c14'] = c14

        # Fill up your empty arrays with the useful values
        #== Beta =========
        m_array[i] = results.params['x']
        
        #== Intercept ====
        c_array[i] = c
        
        #== Int at 14 ====
        c14_array[i] = c14
        
        #== Partial r ====
        r_array[i] = linregress_dict['r']

        #== p & perm_p ===
        p_array[i] = linregress_dict['p']
        perm_p_array[i] = linregress_dict['perm_p']
        
    #----------------------------------------------------------------
    # Calculate the fdr p values
    p_fdr_array = fdr(p_array)[1]
    p_fdr_mask = fdr(p_array)[0]
    
    perm_p_fdr_array = fdr(perm_p_array)[1]
    perm_p_fdr_mask = fdr(perm_p_array)[0]
    
    #----------------------------------------------------------------
    # Create masked versions of the slope array
    m_masked_p_array = np.copy(m_array)
    m_masked_p_array[p_array>0.05] = -99
    
    m_masked_perm_p_array = np.copy(m_array)
    m_masked_perm_p_array[perm_p_array>0.05] = -99
    
    m_masked_p_fdr_array = np.copy(m_array)
    m_masked_p_fdr_array[p_fdr_array>0.05] = -99
    
    m_masked_perm_p_fdr_array = np.copy(m_array)
    m_masked_perm_p_fdr_array[perm_p_fdr_array>0.05] = -99
    
    #----------------------------------------------------------------
    # Now save each of these arrays into the dictionary
    regional_linregress_dict['m'] = m_array
    regional_linregress_dict['c'] = c_array
    regional_linregress_dict['c14'] = c14_array
    regional_linregress_dict['r'] = r_array
    regional_linregress_dict['p'] = p_array
    regional_linregress_dict['perm_p'] = perm_p_array
    regional_linregress_dict['p_fdr'] = p_fdr_array
    regional_linregress_dict['perm_p_fdr'] = m_array
    regional_linregress_dict['m_masked_p'] = m_masked_p_array
    regional_linregress_dict['m_masked_perm_p'] = m_masked_perm_p_array
    regional_linregress_dict['m_masked_p_fdr'] = m_masked_p_fdr_array
    regional_linregress_dict['m_masked_perm_p_fdr'] = m_masked_perm_p_fdr_array
    
    # Return the regional regression dictionary
    return regional_linregress_dict
    
    
def regional_linregress_byregion(df_x, df_y, names, covars=[], n_perm=1000, categorical=False):
    '''
    A function that calls a multiple regression model repeatedly for
    each variable name (in names) with the data in df_y as the dependent
    variable, and the data in df_x as the independent variable. Data in 
    df_x named as in covars are passed as covariates.

    
    INPUTS: 
        df_x ----------- pandas data frame containing x axis values
        df_y ----------- pandas data frame containing y axis values
        covars --------- list containing variable names that should be controlled for
                           (must be columns in df_x and the same for all 
                           dependent variables)
                           Default value: [ ]
        names ---------- list of variable names (columns in df_x and df_y)
                           to loop though and conduct pairwise regressions
        n_perm --------- number of permutations for permutation testing
                           Default value: 1000
        categorical ---- boolean indicating whether you want to permute
                           and return the Fstatistic (if True) or the parameter
                           estimate of the x variable (if False)
                           Default value: False
    
    RETURNS:
        m_array ------------------ numpy array containing slopes for each region
        c_array ------------------ numpy array containing intercepts (at 0) for each region
        c14_array ---------------- numpy array containing intercepts at 14 for each region
        r_array ------------------ numpy array containing partial r (correcting for covariates) for each region
        p_array ------------------ numpy array containing raw p values for each region
        perm_p_array ------------- numpy array containing raw permutation p values for each region
        p_fdr_array -------------- numpy array containing fdr corrected p values for each region
        perm_p_fdr_array --------- numpy array containing fdr corrected permutation p values for each region
        m_masked_p_array --------- numpy array containing the slope values for regions which
                                     are indivudially significant, otherwise -99 markers
        m_masked_perm_p_array ---- numpy array containing the slope values for regions which
                                     are indivudially significant according to permutation
                                     testing, otherwise -99 markers
        m_fdr_masked_array ------- numpy array containing the slope values for regions which
                                     pass fdr correction otherwise -99 markers
        m_perm_fdr_masked_array -- numpy array containing the slope values for regions which
                                     pass fdr correction according to permutation testing,
                                     otherwise -99 markers
    '''
    #----------------------------------------------------------------
    # Import what you need
    from statsmodels.sandbox.stats.multicomp import fdrcorrection0 as fdr
    import numpy as np
    from scipy.stats import linregress
    
    #----------------------------------------------------------------
    # Set up an empty dictionary to save all these values
    regional_linregress_dict = {}
    
    #----------------------------------------------------------------
    # Set up your covars_list
    # This should contain all the data for each covar as a different
    # element in the list
    # You're going to save the index at which wbic appears in the list
    # because you'll need to add the param for this covar to the 
    # intercept value later to get a value for a woman (if male is 
    # included in the covariates list) at wbic.
    # (You don't have to correct your intercept for being a woman 
    # because the male covariate is coded as 0 for women, but you do
    # have to correct for wbic because there 0 represents CBU)
    covars_list = []
    wbic_i = None
    
    for i, covar in enumerate(covars):
        covars_list += [df_x[covar].values]
        if covar == 'wbic':
            wbic_i = i
    
    #----------------------------------------------------------------
    # Set up some empty arrays to contain:
    #    m: slope of the regression line
    #    c: intercept as x = 0 + the parameter estimate
    #                          for the wbic scanner location if passed
    #    c14: intercept when x = 14 (c + 14 * m)
    #    r: partial r
    #    p: p from the ols regression (for x)
    #    perm_p: p from the permutation test
    m_array = np.ones(len(names))
    c_array = np.ones(len(names))
    c14_array = np.ones(len(names))
    r_array = np.ones(len(names))
    p_array = np.ones(len(names))
    perm_p_array = np.ones(len(names))

    #----------------------------------------------------------------
    # Merge the data frames together
    df_xy = df_x.merge(df_y, on='nspn_id', how='inner')
    
    #----------------------------------------------------------------
    # Loop through all the regions and first regress out the
    # covariates and then record m, c, p and perm_p
    # for each region
    for i, roi in enumerate(names):
    
        results, perm_p = permutation_ols(df_xy['{}_x'.format(roi)].values,
                                            df_xy['{}_y'.format(roi)].values,
                                            covars_orig=covars_list, 
                                            categorical=categorical,
                                            n_perm=n_perm)
        
        # Fill up your empty arrays with the useful values
        # from the OLS results
        #== Beta =========
        m_array[i] = results.params['x']
        
        #== Intercept ====
        if wbic_i:
            wbic_param = results.params['c_{}'.format(wbic_i)]
        else:
            wbic_param = 0
        c_array[i] = results.params['Intercept'] + wbic_param
        
        #== Int at 14 ====
        c14_array[i] = c_array[i] + 14*m_array[i]
        
        #== Partial r ====
        t = results.tvalues['x']
        df_resid = results.df_resid
        if t < 0:
            direction = -1
        else:
            direction = 1
        r_array[i] = np.sqrt(t**2 / (t**2 + df_resid)) * direction

        #== p & perm_p ===
        p_array[i] = results.pvalues['x']
        perm_p_array[i] = perm_p
        
    #----------------------------------------------------------------
    # Calculate the fdr p values
    p_fdr_array = fdr(p_array)[1]
    p_fdr_mask = fdr(p_array)[0]
    
    perm_p_fdr_array = fdr(perm_p_array)[1]
    perm_p_fdr_mask = fdr(perm_p_array)[0]
    
    #----------------------------------------------------------------
    # Create masked versions of the slope array
    m_masked_p_array = np.copy(m_array)
    m_masked_p_array[p_array>0.05] = -99
    
    m_masked_perm_p_array = np.copy(m_array)
    m_masked_perm_p_array[perm_p_array>0.05] = -99
    
    m_masked_p_fdr_array = np.copy(m_array)
    m_masked_p_fdr_array[p_fdr_array>0.05] = -99
    
    m_masked_perm_p_fdr_array = np.copy(m_array)
    m_masked_perm_p_fdr_array[perm_p_fdr_array>0.05] = -99
    
    #----------------------------------------------------------------
    # Now save each of these arrays into the dictionary
    regional_linregress_dict['m'] = m_array
    regional_linregress_dict['c'] = c_array
    regional_linregress_dict['c14'] = c14_array
    regional_linregress_dict['r'] = r_array
    regional_linregress_dict['p'] = p_array
    regional_linregress_dict['perm_p'] = perm_p_array
    regional_linregress_dict['p_fdr'] = p_fdr_array
    regional_linregress_dict['perm_p_fdr'] = m_array
    regional_linregress_dict['m_masked_p'] = m_masked_p_array
    regional_linregress_dict['m_masked_perm_p'] = m_masked_perm_p_array
    regional_linregress_dict['m_masked_p_fdr'] = m_masked_p_fdr_array
    regional_linregress_dict['m_masked_perm_p_fdr'] = m_masked_perm_p_fdr_array
    
    # Return the regional regression dictionary
    return regional_linregress_dict
    
    
def fill_linregress_dict(results, perm_p, wbic_i=None):

    import numpy as np
    
    linregress_dict = {}
    
    # Fill up your empty arrays with the useful values
    # from the OLS results
    #== Beta =========
    linregress_dict['m'] = results.params['x']
    
    #== Intercept ====
    if not wbic_i is None:
        wbic_param = results.params['c_{}'.format(wbic_i)]
    else:
        wbic_param = 0
    linregress_dict['c'] = results.params['Intercept'] + wbic_param
        
    #== Int at 14 ====
    linregress_dict['c14'] = linregress_dict['c'] + 14*linregress_dict['m']
        
    #== Partial r ====
    t = results.tvalues['x']
    df_resid = results.df_resid
    if t < 0:
        direction = -1
    else:
        direction = 1
    linregress_dict['r'] = np.sqrt(t**2 / (t**2 + df_resid)) * direction

    #== p & perm_p ===
    linregress_dict['p'] = results.pvalues['x']
    linregress_dict['perm_p'] = perm_p
    
    return linregress_dict
    
    
def permutation_correlation(x_orig, y_orig, covars_orig=[], n_perm=1000):
    '''
    A simple permutation test for linear regression
    between x and y correcting for covars, which should
    be a list of arrays the same length as x and y
    '''
    import numpy as np
    from scipy.stats import linregress 
    import pandas as pd
    
    from useful_functions import residuals
    
    # Make a copy of the original data
    # because the shuffle command does this in place
    # and we don't want any mistakes!
    x = np.copy(x_orig)
    y = np.copy(y_orig)
    if len(covars_orig) > 0:
        covars = np.copy(covars_orig)
    else:
        covars = np.ones_like(x)
        
    # Get the residuals of x and y against covars
    res_x = residuals(covars, x)
    res_y = residuals(covars, y)
    
    # Run the unpermuted correlation
    m, c, r, p, sterr = linregress(res_x, res_y)
    
    # Create an m_array that will hold the shuffled
    # slope values
    m_array = np.ones([n_perm])
    
    # Now loop through all the shuffles and calculate
    # the regression for each, saving the slopes in the
    # m_array you created above
    for i in range(n_perm):
        np.random.shuffle(y)
        res_y = residuals(covars, y)
        m_shuff, c_shuff, r_shuff, p_shuff, sterr_shuff = linregress(res_x, res_y)
        m_array[i] = m_shuff
    
    # If the true slope is positive then we want to look
    # for the proportion of shuffled slopes that are
    # larger than the true slope
    if m < 0:
        perm_p = len(m_array[m_array<m])/np.float(n_perm)
    # If the true slope is negative then we want to look
    # for the proportion of shuffled slopes that are
    # *more negative* than the true slope
    if m > 0:
        perm_p = len(m_array[m_array>m])/np.float(n_perm)
    
    # We're doing a 2 tailed test so we have to multiply
    # the perm_p value by 2
    perm_p = perm_p*2.0
    
    linregress_dict = {}
    linregress_dict['m'] = m
    linregress_dict['c'] = c
    linregress_dict['r'] = r
    linregress_dict['p'] = p
    linregress_dict['perm_p'] = perm_p
    
    return linregress_dict


def ols_correlation(x, y, covars=[], wbic_covars_index=None):
    '''
    x_orig, y_orig are 1D numpy arrays with n data points
    covars_orig is a list of 1D numpy arrays with n data points
    corresponding to each of the covariates you'd like to
    control for.
    '''
    
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd
    
    # Create the data frame
    df =  pd.DataFrame({'x' : x,
                        'y' : y})
                            
    # Create your formula
    formula = 'y ~ x'

    # Add in the covariates
    for i, covars in enumerate(covars):
        if not np.std(covars) == 0.0:
            df['c_{}'.format(i)] = covars
            formula += ' + c_{}'.format(i)
    
    # Fit the model
    model = sm.OLS.from_formula(formula, df)
    results = model.fit()
    
    # We want to know the intercept
    # but that means we have to take into account
    # whether wbic has been included as a covariate
    if not wbic_covars_index is None:
        c_wbic = results.params['c_{}'.format(wbic_covars_index)]
    else:
        c_wbic = 0
        
    # Add this value to the intercept and it will give you
    # the intercept corrected for wbic (you should always have
    # ucl in the correlation too...but we define wbic as the 
    # "baseline" scanner
    c = results.params['Intercept'] + c_wbic
    c_14 = results.params['x'] * 14 + c
    
    # Return all of the results, but also the slope
    # because that's likely what you actually care about!
    return results, c, c_14
    
