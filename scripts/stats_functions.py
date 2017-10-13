#!/usr/bin/env python

def residuals(x, y):
    '''
    A useful little function that correlates
    x and y together to give their residual
    values. These can then be used to calculate
    partial correlation values.
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
    
def partial_r(x, y, covars):

    import numpy as np
    from scipy.stats import pearsonr
    
    res_i = residuals(covars, x)
    res_j = residuals(covars, y)
    part_r = pearsonr(res_i, res_j)[0]
    return part_r
    
def variance_partition(x1, x2, y):
    '''
    Describe the independent and shared explanatory
    variance of two (possibly correlated) variables on 
    the dependent variable (y)
    '''
    from statsmodels.formula.api import ols
    import numpy as np
    from scipy.stats import pearsonr
    import pandas as pd
    
    # Set up the data frame
    df = pd.DataFrame( { 'Y' : y ,
                         'X1' : x1,
                         'X2' : x2 } )
                         
    # Get the overall r squared value for the 
    # multiple regression
    Rsq = ols('Y ~ X1 + X2', data=df).fit().rsquared
    
    # Next calculate the residuals of X1 after correlating
    # with X2 (so units will be in those of X1) and vice versa
    df['res_X1givenX2'] = residuals(df['X2'], df['X1'])
    df['res_X2givenX1'] = residuals(df['X1'], df['X2'])

    # Now calculate the pearson regressions for
    # the residuals against the dependent variable to give
    # the fraction of variance that each explains independently
    # (a and c), along with the fraction of variance 
    # that is shared across both explanatory variables (b).
    # d is the fraction of variance that is not explained
    # by the model.
    a = (pearsonr(df['res_X1givenX2'], df['Y'])[0])**2
    c = (pearsonr(df['res_X2givenX1'], df['Y'])[0])**2    
    b = Rsq - a - c
    d = 1.0 - Rsq
    
    # Return these four fractions
    return a, b, c, d
    