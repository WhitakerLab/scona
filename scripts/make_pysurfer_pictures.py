#!/usr/bin/env python


def pysurfer_308_parcellation(aparc_names, paper_dir):
    import pandas as pd
    import numpy as np
    import os
    
    df = pd.DataFrame(aparc_names)
    df['names_158'] = [ x.split('_',1)[1] for x in aparc_names ]

    color_vals = np.arange(158)
    np.random.shuffle(color_vals)

    for i, (name, data) in enumerate(df.groupby(['names_158'])):
        df.loc[(df['names_158']== name), 'color_val'] = color_vals[i]

    output_dir = os.path.join(paper_dir, 'COMBINED_FIGURES', 'PARCELLATION')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Save the values to text file
    fname = os.path.join(output_dir, 'Parcellation_308_random_matched_hemis.csv')
    np.savetxt(fname, df['color_val'], fmt='%i')
    
    return fname
    
def create_pysurfer_command(roi_file,
                            scripts_dir, 
                            sub_data_dir, 
                            c='jet', 
                            l=None, 
                            u=None, 
                            t=-99, 
                            s='pial', 
                            cst='classic',
                            center=False,
                            c2=None,
                            t2=None):
    '''
    Create a text string containing the appropriate options for
    the pysurfer command
    '''
    import os
    
    # Create the command for pysurfer and run it
    # start by putting in the name of the code with its path
    command_list = [ os.path.join(scripts_dir, 
                                  'pysurfer_plot_500parcellation_surface_values.py') ]
    
    # Set the subject directory
    command_list += [ '-sd {}'.format(sub_data_dir) ]
    
    # Set the surface
    if s: command_list += [ '-s {}'.format(s) ]

    # Set the colormap
    if c: command_list += [ '-c {}'.format(c) ]

    # Set the other colormap
    if c2: command_list += [ '-c2 {}'.format(c2) ]
        
    # Set the colormap limits
    if l: command_list += [ '-l {}'.format(l) ]
    if u: command_list += [ '-u {}'.format(u) ]
        
    # Set the threshold
    if t: command_list += [ '-t {}'.format(t) ]
        
    # Set the other threshold
    if t2: command_list += [ '-t2 {}'.format(t2) ]

    # Center if necessary
    if center: command_list += [ '--center' ]
        
    # Change the cortex style if necessary
    if cst: command_list += [ '-cst {}'.format(cst) ]
        
    # And add the filename
    command_list += [ roi_file ]
        
    # Finally join it all together into one string
    command = ' '.join(command_list)

    return command
    
        
def write_to_text(measure_dict, pysurfer_dir, measure_name, covars_name='none'):
    '''
    mpm is the mpm measure you're investigating (eg: MT)
    
    For example:
        write_to_text(measure_dict, 
                        pysurfer_dir, 
                        'corr_age_coavars_site_c14',
                        'SlopeAge_at14_covars_site')
    '''
    import os
    import numpy as np
    
    covars_dict = measure_dict['308']['COVARS_{}'.format(covars_name)]
    
    output_dir = os.path.join(pysurfer_dir, 'COVARS_{}'.format(covars_name))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Save the values to text file
    x = covars_dict[measure_name.split('_movie')[0]]
    fname = os.path.join(output_dir, '{}.txt'.format(measure_name))
    np.savetxt(fname, x, fmt='%5.5f')
    
    return fname


def get_cmap_min_max_dict(measure_list):

    cmap_min_max_dict = {}
    
    # MOVIE MAPS
    mt_c14_list = [ x for x in measure_list if x.startswith('MT_proj') and x.endswith('regional_corr_age_c14') ]
    for measure in mt_c14_list:
        cmap_min_max_dict['{}_movie'.format(measure)] = ( 'jet', 0.4, 1.8 )
        slope_measure = measure.replace('_c14', '_m')
        cmap_min_max_dict['{}_movie'.format(slope_measure)] = ( 'RdBu_r', -0.007, 0.007 )
    
    # CT maps:
    cmap_min_max_dict['CT_regional_corr_age_c14'] = ( 'jet', 2.5, 3.5 )
    cmap_min_max_dict['CT_regional_corr_age_m'] = ( 'RdBu_r', -0.03, 0.03 )
    cmap_min_max_dict['CT_regional_corr_age_m_masked_p_fdr'] = ( 'winter_r', -0.03, -0.01 )

    # MT_projfrac+030 maps:
    cmap_min_max_dict['MT_projfrac+030_regional_corr_age_c14'] = ( 'jet', 0.8, 1.0 )
    cmap_min_max_dict['MT_projfrac+030_regional_corr_age_m'] = ( 'RdBu_r', -0.007, 0.007 )
    cmap_min_max_dict['MT_projfrac+030_regional_corr_age_m_masked_p_fdr'] = ( 'autumn', 0.002, 0.007 )

    # PLS maps
    cmap_min_max_dict['PLS1_with99s'] = ( 'RdBu_r', -0.07, 0.07 )
    cmap_min_max_dict['PLS2_with99s'] = ( 'RdBu_r', -0.07, 0.07 )
    
    return cmap_min_max_dict

def make_pysurfer_figures(measure_dict, pysurfer_dir, sub_data_dir, scripts_dir, paper_dir, overwrite=False):
    
    import os
    import itertools as it
    
    #----------------------------------------------------------------
    # Define the covars dictionary
    covars_dict = { 'gender'      : ['male'],
                    'site'        : ['wbic', 'ucl'],
                    'gender_site' : ['male', 'wbic', 'ucl'],
                    'none'        : [] }
    
    covars_dict = { 'none' : [] }
    #----------------------------------------------------------------
    # Define the measures we want to write out
    sub_dict = measure_dict['308']['COVARS_none']
    
    measure_list = []
    measure_list += [ x for x in sub_dict.keys() if x.endswith('_regional_corr_age_m') ]
    measure_list += [ x for x in sub_dict.keys() if x.endswith('_regional_corr_age_c14') ]
    measure_list += [ 'CT_regional_corr_age_m_masked_p_fdr' ]
    measure_list += [ 'MT_projfrac+030_regional_corr_age_m_masked_p_fdr' ]
    measure_list += [ 'MT_projfrac+030_regional_corr_age_m_masked_p_fdr' ]
    
    # Get the colormaps and min/max values
    cmap_min_max_dict = get_cmap_min_max_dict(measure_list)
    
    #----------------------------------------------------------------
    for covars_name, measure in it.product(covars_dict.keys(), cmap_min_max_dict.keys()):
        # Start by writing out the text files
        fname = write_to_text(measure_dict, 
                                    pysurfer_dir, 
                                    measure, 
                                    covars_name=covars_name)
    
        # 
        # Create the pysurfer command
        command = create_pysurfer_command(fname,
                                            scripts_dir, 
                                            sub_data_dir, 
                                            c=cmap_min_max_dict[measure][0], 
                                            l=cmap_min_max_dict[measure][1], 
                                            u=cmap_min_max_dict[measure][2], 
                                            t=-98, 
                                            s='pial',
                                            cst='classic', 
                                            center=False)
    
        # Make the pictures if they haven't been made already
        png_file = os.path.join(pysurfer_dir, 
                                'COVARS_{}'.format(covars_name),
                                'PNGS', 
                                '{}_pial_classic_combined.png'.format(measure))
        
        if overwrite or not os.path.isfile(png_file):
            os.system(command)
    
    #----------------------------------------------------------------
    # and lastly the atlas regions
    fname = pysurfer_308_parcellation(measure_dict['308']['aparc_names'], 
                                            paper_dir)
    
    command = create_pysurfer_command(fname,
                                            scripts_dir, 
                                            sub_data_dir, 
                                            c='jet',
                                            s='pial',
                                            cst='classic', 
                                            center=False)
                                            
    # Make the pictures if they haven't been made already
    png_file = os.path.join(paper_dir, 
                            'COMBINED_FIGURES',
                            'PARCELLATION',
                            'PNGS', 
                            'Parcellation_308_random_matched_hemis_pial_classic_combined.png')
    
    if overwrite or not os.path.isfile(png_file):
        os.system(command)