#!/usr/bin/env python

# Run PLS using the matlab code

def pls_commands(measure_dict, mpm, pls_dir, scripts_dir, matlab_dir, data_dir):
    '''
    A useful wrapper to call all the various pls commands 
    using the values in measure_dict
    
    Inputs:
      measure_dict --------- so we have the data
      mpm ------------------ whether you're running this for MT, R1 etc
      pls_dir -------------- where the pls output will be saved
      scripts_dir ---------- where the pls_scripts are saved
      matlab_dir ----------- the path to your installed version of matlab
      data_dir ------------- where the allen brain atlas data is saved
    '''
    
    #----------------------------------------------------------------
    # Get the gene indices
    measure_dict['308']['gene_indices'] = get_gene_indices(data_dir)
    
    #----------------------------------------------------------------
    # Define the covars dictionary
    covars_dict = { 'gender'      : ['male'],
                    'site'        : ['wbic', 'ucl'],
                    'gender_site' : ['male', 'wbic', 'ucl'],
                    'none'        : [] }

    #----------------------------------------------------------------
    # Now run PLS for each covar
    
    for covars_name in covars_dict.keys():
        
        call_pls(measure_dict, 
                        pls_dir, 
                        scripts_dir, 
                        matlab_dir, 
                        data_dir,
                        mpm_key='MT_projfrac+030', 
                        gene_indices=measure_dict['308']['gene_indices'], 
                        covars_name=covars_name)
    
        # Now read in the different measures to your measure_dict
        measure_dict = save_pls_results(measure_dict, pls_dir, data_dir, covars_name=covars_name, mpm_key='MT_projfrac+030')
    
    return measure_dict

def call_pls(measure_dict, pls_dir, scripts_dir, matlab_dir, gene_dir, mpm_key='MT_projfrac+030', gene_indices=None, covars_name='none'):
    '''
    An important little function that writes a bash script to call the appropriate
    PLS commands in matlab
    
    Inputs:
      measure_dict --------- so we have the data
      pls_dir -------------- where the pls output will be saved
      scripts_dir ---------- where the pls_scripts are saved
      matlab_dir ----------- the path to your installed version of matlab
      mpm_key -------------- which MPM measure (and depth) you'd like to include
                              in the response variables (along with CT)
      gene_indicies -------- which MRI regions to include in the PLS analysis
      covars_name ---------- which covariate to correct the MRI measures for
    '''
    
    import os
    
    # First we need to set up the input files to the matlab PLS code
    write_pls_input_data(measure_dict, pls_dir, mpm_key=mpm_key, gene_indices=gene_indices, covars_name=covars_name)
    
    # Define your covars directory inside the pls_dir
    covars_dir = os.path.join(pls_dir, 'COVARS_{}'.format(covars_name))

    # Define the four input files to PLS
    mri_response_vars_file = os.path.join(covars_dir, 'PLS_MRI_response_vars.csv')
    gene_predictor_vars_file = os.path.join(gene_dir, 'PLS_gene_predictor_vars.csv')
    candidate_genes_SZ = os.path.join(gene_dir, 'Candidate_genes_schizophrenia.csv')
    candidate_genes_OL = os.path.join(gene_dir, 'Candidate_genes_oligo.csv')
    
    # Next write out the matlab commands to a bash script file
    script_file = os.path.join(covars_dir, 'matlab_pls_commands.sh')
    with open(script_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        
        if not os.path.isfile(os.path.join(covars_dir, 'PLS2_ROIscores.csv')):
            f.write('# First, run PLS\n')
            f.write('\'{}\' -nodisplay -nosplash -nodesktop -nojvm '.format(matlab_dir))
            f.write('-r "addpath(\'{}\'); '.format(scripts_dir))
            f.write('addpath(\'{}\'); '.format(pls_dir))
            f.write('PLS_bootstrap( \'{}\' , \'{}\' , \'{}\' ); '.format(mri_response_vars_file,
                                                                        gene_predictor_vars_file,
                                                                        covars_dir))
            f.write('exit;"')
            f.write('\n')
        else:
            f.write('# Not running PLS as it has already been run\n')
            f.write('# Remove PLS2_ROIscores.csv file if you would like to overwrite the results\n')
            print '  PLS has already been run'
            
        if not os.path.isfile(os.path.join(covars_dir, 'PLS_stats.csv')):
            f.write('# Next, calculate the significance of the PLS\n')
            f.write('\'{}\' -nodisplay -nosplash -nodesktop  -nojvm '.format(matlab_dir))
            f.write('-r "addpath(\'{}\'); '.format(scripts_dir))
            f.write('PLS_calculate_stats( \'{}\' , \'{}\' , \'{}\' ); '.format(mri_response_vars_file,
                                                                        gene_predictor_vars_file,
                                                                        covars_dir))
            f.write('exit;"')
            f.write('\n')
        else:
            f.write('# Not running PLS as it has already been run\n')
            f.write('# Remove PLS_stats.csv file if you would like to overwrite the results\n')
        
        if not os.path.isfile(os.path.join(covars_dir, 'PLS2_schizophrenia.csv')):
            f.write('# Finally, assess the significance of the candidate genes\n')
            f.write('\'{}\' -nodisplay -nosplash -nodesktop '.format(matlab_dir))
            f.write('-r "addpath(\'{}\'); '.format(scripts_dir))
            for pls in [ '1', '2']:
                OL_filename=os.path.join(covars_dir, 'PLS{}_oligo.csv'.format(pls))
                SZ_filename=os.path.join(covars_dir, 'PLS{}_schizophrenia.csv'.format(pls))
                gene_weights_file = os.path.join(covars_dir, 'PLS{}_geneWeights.csv'.format(pls))
                
                f.write('PLS_candidate_genes( \'{}\' , \'{}\' , \'{}\' , false ); '.format(gene_weights_file,
                                                                            candidate_genes_OL,
                                                                            OL_filename))
                f.write('PLS_candidate_genes( \'{}\' , \'{}\' , \'{}\' , true ); '.format(gene_weights_file,
                                                                            candidate_genes_SZ,
                                                                            SZ_filename))
            f.write('exit;"')
            f.write('\n')
        else:
            f.write('# Not running candidate gene analysis as it has already been run\n')
            f.write('# Remove PLS2_schizophrenia.csv file if you would like to overwrite the results\n')
            print '  Candidate gene analysis has already been run'
        
    os.system('chmod +x {}'.format(script_file))
    os.system(script_file)



def write_pls_input_data(measure_dict, pls_dir, mpm_key='MT_projfrac+030', gene_indices=None, covars_name='none'):
    '''
    Create a csv file that has 306 entries and five columns:
      * ROI_Name
      * CT -------- CT at age 14
      * MT -------- MT at age 14
      * dCT ------- change in CT with age
      * dMT ------- change in MT with age
    
    And save it in the appropriate covars dir inside the cohort PLS directory
    '''
    import numpy as np
    import pandas as pd
    import os
    
    if gene_indices is None:
        gene_indices = range(len(measure_dict['308']['aparc_names']))
    
    # Make the gene indices an array rather than a list
    gene_indices = np.array(gene_indices)
    
    df = pd.DataFrame( { 'ROI_name' : np.array(measure_dict['308']['aparc_names'])[gene_indices],
                         'CT' : measure_dict['308']['COVARS_{}'.format(covars_name)]['CT_regional_corr_age_c14'][gene_indices],
                         'MT' : measure_dict['308']['COVARS_{}'.format(covars_name)]['{}_regional_corr_age_c14'.format(mpm_key)][gene_indices],
                         'dCT' : measure_dict['308']['COVARS_{}'.format(covars_name)]['CT_regional_corr_age_m'][gene_indices],
                         'dMT' : measure_dict['308']['COVARS_{}'.format(covars_name)]['{}_regional_corr_age_m'.format(mpm_key)][gene_indices] } )
                         
    covars_dir = os.path.join(pls_dir, 'COVARS_{}'.format(covars_name))
    if not os.path.isdir(covars_dir):
        os.makedirs(covars_dir)
        
    df.to_csv(os.path.join(covars_dir, 'PLS_MRI_response_vars.csv'), 
                            columns = ['ROI_name', 'CT', 'MT', 'dCT', 'dMT'],
                            index=False)
    
    
def get_gene_indices(data_dir):
    '''
    The gene indices are the regions that have good
    allen data. They are in the header of the PLS_gene_predictor_vars.csv file
    '''
    import numpy as np
    import os
    
    gene_predictor_vars_file = os.path.join(data_dir, 'PLS_gene_predictor_vars.csv')
    
    with open(gene_predictor_vars_file, 'r') as f:
        header = f.readline().strip()
        gene_indices = np.array([ int(x) for x in header.split(',')[1:] ])
    
    return gene_indices


def get_mbp(data_dir):
    '''
    The gene indices are the regions that have good
    allen data. They are in the header of the PLS_gene_predictor_vars.csv file
    '''
    import numpy as np
    import os
    import pandas as pd
    
    gene_predictor_vars_file = os.path.join(data_dir, 'PLS_gene_predictor_vars.csv')
    
    df = pd.read_csv(gene_predictor_vars_file)
    
    mbp = df.loc[df['Gene'] =='MBP', :].values[0][1:]

    return mbp




def save_pls_results(measure_dict, pls_dir, data_dir, covars_name='none', mpm_key='MT_projfrac+030'):
    
    import numpy as np
    import pandas as pd
    import os
    
    covars_dir = os.path.join(pls_dir, 'COVARS_{}'.format(covars_name))
        
    gene_indices = measure_dict['308']['gene_indices']

    mbp = get_mbp(data_dir)
    
    # Write pls values into a data frame
    pls1 = np.loadtxt(os.path.join(covars_dir, 'PLS1_ROIscores.csv'))
    pls2 = np.loadtxt(os.path.join(covars_dir, 'PLS2_ROIscores.csv'))
    df = pd.DataFrame( { 'PLS1' : pls1, 'PLS2' : pls2 } )
    
    # Add in hemisphere and dk_region name
    df['name'] = np.array(measure_dict['308']['aparc_names'])[gene_indices]
    df['hemi'] = measure_dict['308']['hemi'][gene_indices]
    df['dk_region'] = [ x.split('_')[1] for x in df['name'] ]

    # Loop through the three different parcellations
    for n in [ 308, 68, 34 ]:
        
        # Sort out the grouping and suffices
        if n == 68:
            grouped = df.groupby(['hemi', 'dk_region'])
        elif n == 34:
            grouped = df.groupby(['dk_region'])
        else:
            grouped = df.groupby('name', sort=False)
        
        pls1_list = []
        pls2_list = []
    
        for name, data in grouped:
            pls1_list += [ np.percentile(data['PLS1'], 50) ]
            pls2_list += [ np.percentile(data['PLS2'], 50) ]
        
        # Write values to sub_dict
        sub_dict = measure_dict[str(n)]['COVARS_{}'.format(covars_name)]
    
        sub_dict['PLS1'] = pls1_list
        sub_dict['PLS2'] = pls2_list
        
        if n == 308:
            arr308 = np.ones([308]) * -99
            arr308[gene_indices] = sub_dict['PLS1']
            sub_dict['PLS1_with99s'] = np.copy(arr308)
            arr308[gene_indices] = sub_dict['PLS2']
            sub_dict['PLS2_with99s'] = np.copy(arr308)
            
        if n == 308:
            sub_dict['PLS1_SZ'] = np.loadtxt(os.path.join(covars_dir, 'PLS1_schizophrenia.csv'))
            sub_dict['PLS2_SZ'] = np.loadtxt(os.path.join(covars_dir, 'PLS2_schizophrenia.csv'))
            sub_dict['PLS1_OL'] = np.loadtxt(os.path.join(covars_dir, 'PLS1_oligo.csv'))
            sub_dict['PLS2_OL'] = np.loadtxt(os.path.join(covars_dir, 'PLS2_oligo.csv'))
            sub_dict['MBP'] = mbp
    
        measure_dict[str(n)]['COVARS_{}'.format(covars_name)] = sub_dict

    
    return measure_dict