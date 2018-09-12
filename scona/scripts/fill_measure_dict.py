#!/usr/bin/env python

def fill_measure_dict_part1(measure_dict, mpm, data_dir, fsaverage_dict, n_perm=1000):
    '''
    A function that extracts a bunch of interesting values and saves them
    so that they can easily be accessed by the reporting functions (such as those
    that make figures and tables)
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
    print "Filling measure dict with regional values and correlations with age"
    
    #==========================================================================
    # Start by saving the fsaverage_dict name values for easy access
    #--------------------------------------------------------------------------
    # This code returns a measure_dict that contrains three dictionaries with
    # the following three keys:
    #   * 308 ------------- All 308 equally sized regions 
    #   * 68 -------------- Those 308 regions collapsed within atlas region
    #   * 34 -------------- Those 308 regions collapsed within atlas region
    #                         and hemisphere
    #
    # Each of these three dictionaries contain:
    #   * aparc_names ----- Name of the regions
    #   * lobes ----------- Which lobe each region belongs to
    #   * N_SubRegions ---- Number of sub regions that make up this region
    #   * von_economo ----- The von economo lamination type for this region
    #
    # The 308 and 68 dictionaries contain:
    #   * hemi ------------ Which hemisphere each region belongs to
    #
    # The 308 dictionary contains:
    #   * centroids ------- The x, y, z coordinates for each region
    #==========================================================================
    measure_dict = save_name_lists(measure_dict, 
                                    fsaverage_dict['aparc_names'], 
                                    fsaverage_dict['lobes'], 
                                    fsaverage_dict['von_economo'], 
                                    fsaverage_dict['centroids'])

    #==========================================================================
    # Create a list of filenames and another of measure names
    #==========================================================================
    filename_list, measure_name_list = create_file_measure_name_lists(data_dir, mpm)
    
    #==========================================================================
    # Read in the cortical thickness file and save a couple of useful values:
    #   * age as age_scan
    #   * gender as male (1 for male, 0 for female)
    #   * whether scan was at WBIC as wbic (1 for wbic, 0 if not)
    #   * whether scan was at UCL as ucl (1 for ucl, 0 if not)
    #       (Note that this means a scan was at CBU if it has both
    #       a 0 for wbic and a 0 for ucl)
    #==========================================================================
    # Read in the CT data separately before starting the loop
    # as it'll be needed for some of the correlations
    df_ct = read_in_df(filename_list[0], measure_dict['308']['aparc_names'])
    
    # Collapse the values according to the 34 and 68 parcellations
    # and put these at the end of the data frame
    df_ct = append_collapsed_across_regions(df_ct, measure_dict)
    
    # Save the age and gender and scan location of the participants
    # in the 308 measure dict
    measure_dict['308']['age_scan'] = df_ct['age_scan']
    measure_dict['308']['male'] = df_ct['male']
    measure_dict['308']['wbic'] = df_ct['wbic']
    measure_dict['308']['ucl'] = df_ct['ucl']

    #==========================================================================
    # Now loop through all the different MPM measures (and CT) 
    # and save a whole bunch of different values:
    #    
    #    FOR EACH NODE:
    #      * Mean
    #      * Std    
    #      * Corr with age (12 model parameters)
    #    Note that these are done for the three different parcellations
    #     (308, 68, 340) and the correlation is done with four combinations 
    #     of covariates (none, scanner, male and scanner_male)
    #
    #    FOR EACH PERSON:
    #      * Global mean
    #      * Global std
    #      * Corr mean with age (6 model parameters)
    #      * Corr std with age (6 model parameters)
    #    Note that the correlations are done with four combinations 
    #     of covariates (none, scanner, male and scanner_male)
    #==========================================================================
    for filename, measure_name in zip(filename_list, measure_name_list):
        
        df = read_in_df(filename, measure_dict['308']['aparc_names'])
        df = append_collapsed_across_regions(df, measure_dict)
    
        # Save a bunch of values for the 308, 68 and 34 parcellations
        measure_dict = save_regional_values(measure_name, measure_dict, df, df_ct, n_perm=1)
        
        # Save a bunch of values for each individual
        measure_dict = save_global_values(measure_dict, measure_name, df, df_ct, n_perm=n_perm)
    
    return measure_dict


def fill_measure_dict_part2(measure_dict, graph_dict, n_perm=1000):
    '''
    A function that extracts a bunch of interesting values and saves them
    so that they can easily be accessed by the reporting functions (such as those
    that make figures and tables)
    '''
    #==========================================================================
    # IMPORTS
    #==========================================================================
    import os
    import numpy as np
        
    #==========================================================================
    # Print to screen what you're up to
    #==========================================================================
    print "--------------------------------------------------"
    print "Filling measure dict with network values and regional correlations"
    
    #==========================================================================
    # Save the network measures
    #
    #    FOR EACH NODE:
    #      * Degree
    #      * Closeness
    #      * Clustering
    #      * Betweenness
    #      * AverageDist
    #      * TotalDist
    #      * InterhemProp
    #    Note that these measures are calculated using the 308 graph, but the
    #    median values are saved for the 68 and 34 parcellations
    #
    #    GLOBAL:
    #      * Clustering
    #      * Modularity
    #      * ShortestPath
    #      * Efficiency
    #      * Assortativity
    #      * SmallWorld
    #    Note that these measures are saved for the actual graph and for 
    #    random graphs separately
    #      
    #==========================================================================
    measure_dict = save_network_values(measure_dict, graph_dict)
        
    
    #==========================================================================
    # Save the regional correlations
    #
    #    ACROSS NODES: (y vs x)
    #      * Corr slope with age vs intercept at 14
    #      * Corr intercept at 14 vs CT intercept at 14
    #      * Corr slope with age vs CT slope with age
    #      * Corr PLSComponent vs intercept at 14
    #      * Corr PLSComponent vs slope with age
    #      * Corr MBP vs intercept at 14
    #      * Corr MBP vs slope with age
    #      * Corr NetworkMeasure vs intercept at 14
    #      * Corr NetworkMeasure vs slope with age
    #      * Corr NetworkMeasure vs PLS1
    #      * Corr NetworkMeasure vs PLS2
    #    Note that each correlation is made up of 6 different model parameters
    #
    #    PLSComponents are PLS1 and PLS2
    #    NetworkMeasures are Degree, Closeness, Clustering and AverageDistance
    #==========================================================================
    measure_dict = save_regional_correlations(measure_dict, n_perm=n_perm)
    
    return measure_dict


def create_file_measure_name_lists(data_dir, mpm):
    '''
    Loop through all the files you are interested in and save the 
    filenames and the measure names
    '''
    #----------------------------------------------------------------
    # Import what you need
    import os
    import numpy as np

    #----------------------------------------------------------------
    # Initiate the lists with the CT values
    filename_list = [ os.path.join(data_dir, 'PARC_500aparc_thickness_behavmerge.csv') ]
    measure_name_list = [ 'CT' ]
    
    #----------------------------------------------------------------
    # Now add the 308 MPM values calculated across the whole of cortex
    # (rather than at particular depths) and the values for the average
    # signal between the grey/white matter boundary and 2mm into white
    # matter
    filename_list += [ os.path.join(data_dir, 'PARC_500aparc_{}_cortexAv_mean_behavmerge.csv'.format(mpm)) ]
    measure_name_list += [ '{}_cortexAv'.format(mpm) ]
    
    filename_list += [ os.path.join(data_dir, 'PARC_500aparc_{}_wmAv_mean_behavmerge.csv'.format(mpm)) ]
    measure_name_list += [ '{}_wmAv'.format(mpm) ]
    
    #----------------------------------------------------------------
    # Loop through the 11 different fractional depths
    for i in np.arange(0.0,110,10):
        
        filename_list += [ os.path.join(data_dir, 
                        'PARC_500aparc_{}_projfrac{:+04.0f}_mean_behavmerge.csv'.format(mpm, i)) ]
        measure_name_list += [ '{}_projfrac{:+04.0f}'.format(mpm, i) ]
        
    #----------------------------------------------------------------
    # Loop through the 10 different absolute depths
    for i in np.arange(-20,-101,-20):
        
        filename_list += [ os.path.join(data_dir, 
                        'PARC_500aparc_{}_projdist{:+04.0f}_fromBoundary_mean_behavmerge.csv'.format(mpm, i)) ]
        measure_name_list += [ '{}_projdist{:+04.0f}'.format(mpm, i) ]

    return filename_list, measure_name_list
    
    
def save_name_lists(measure_dict, aparc_names, lobes, von_economo, centroids):
    '''
    A useful function that saves a bunch of names to the measure_dict
    for easy access by the reporting functions (those that make figures
    and tables), along with the number of regions in each of the 68 and 
    34 parcellations. These are the Desikan-Killiany atlas regions 
    separated by hemisphere (68) or collapsed across right and left (34).
    '''
    #----------------------------------------------------------------
    # Import what you need
    import numpy as np
    
    measure_dict['308'] = measure_dict.get('308', {})
    measure_dict['34'] = measure_dict.get('34', {})
    measure_dict['68'] = measure_dict.get('68', {})
    
    # ROI names
    measure_dict['308']['aparc_names'] = aparc_names
    measure_dict['34']['aparc_names'] = sorted(list(set([ roi.split('_')[1] for roi in aparc_names ])))
    measure_dict['68']['aparc_names'] = sorted(list(set([ roi.rsplit('_', 1)[0] for roi in aparc_names ])))
    
    # ROI hemispheres
    measure_dict['308']['hemi'] = np.array([ name[0] for name in measure_dict['308']['aparc_names']])
    measure_dict['68']['hemi'] = np.array([ name[0] for name in measure_dict['68']['aparc_names']])
    
    # ROI lobes and von_economo labels 
    measure_dict['308']['lobes'] = lobes
    measure_dict['308']['von_economo'] = von_economo
    
    for measure in [ 'lobes', 'von_economo' ]:
        # 34 regions
        list_34 = []
        for roi in measure_dict['34']['aparc_names']:
            i = np.where(np.array(aparc_names) == 'lh_{}_part1'.format(roi))
            list_34 += [ measure_dict['308'][measure][i[0]] ]
        measure_dict['34']['{}'.format(measure)] = np.array(list_34)
        
        # 68 regions
        list_68 = []
        for roi in measure_dict['68']['aparc_names']:
            i = np.where(np.array(aparc_names) =='{}_part1'.format(roi))
            list_68 += [ measure_dict['308'][measure][i[0]] ]
        measure_dict['68']['{}'.format(measure)] = np.array(list_68)

    # Centroids - for 308 only at the moment!
    measure_dict['308']['centroids'] = centroids
    
    # Record the number of subregions for each DK atlas region
    measure_dict['308']['N_SubRegions'] = np.ones(308)
    # 34
    n_subregions_34 = []
    for roi in measure_dict['34']['aparc_names']:
        n_subregions_34 += [ len([ x for x in measure_dict['308']['aparc_names'] if roi in x ]) ]
    measure_dict['34']['N_SubRegions'] = np.array(n_subregions_34)
    # 68
    n_subregions_68 = []
    for roi in measure_dict['68']['aparc_names']:
        n_subregions_68 += [ len([ x for x in measure_dict['308']['aparc_names'] if roi in x ]) ]
    measure_dict['68']['N_SubRegions'] = np.array(n_subregions_68)
    
    return measure_dict
    

def append_collapsed_across_regions(df, measure_dict):
    '''
    This code adds in additional columns to the end of
    the data frame collapsing across regions but 
    separating the hemispheres (68) and again combining
    the two hemispheres (34)
    
    df is the data frame read in from the FS_ROIS output
    measure_dict must contain the aparc_names for the 
    308, 68 and 34 dictionaries separately
    '''
    for roi in measure_dict['68']['aparc_names']:
        roi_list = [ x for x in measure_dict['308']['aparc_names'] if roi in x ]
        df['{}'.format(roi)] = df[roi_list].mean(axis=1)
    for roi in measure_dict['34']['aparc_names']:
        roi_list = [ x for x in measure_dict['308']['aparc_names'] if roi in x ]
        df['{}'.format(roi)] = df[roi_list].mean(axis=1)
    
    return df

    
def save_regional_values(measure_name, measure_dict, df, df_ct, n_perm=1000):
    '''
    A snazzy little function that does a couple of correlations 
    **at every region separately** for each of the three parcellations 
    and saves the output in measure_dict
    '''
    #----------------------------------------------------------------
    # Import what you need
    import itertools as it
    import numpy as np
    
    from permutation_testing import ( regional_linregress, 
                                        regional_linregress_byregion )
    
    #----------------------------------------------------------------
    # Print to screen what you're up to
    print '    {}'.format(measure_name)
    print '      Corr with age for all regions'
    
    #----------------------------------------------------------------
    # Define the names and suffix dictionaries
    names_dict = { 308 : measure_dict['308']['aparc_names'],
                    68 : measure_dict['68']['aparc_names'],
                    34 : measure_dict['34']['aparc_names'] }
    
    #----------------------------------------------------------------
    # Define the covars dictionary
    covars_dict = { 'gender'      : ['male'],
                    'site'        : ['wbic', 'ucl'],
                    'gender_site' : ['male', 'wbic', 'ucl'],
                    'none'        : [] }
   
    #----------------------------------------------------------------
    # Loop through the different parcellations with the various
    # covariates
    for (covars_name, covars), (n, names) in it.product(covars_dict.items(), 
                                                        names_dict.items()):
        
        # Create the dictionary if it doesn't already exist
        measure_dict[str(n)] = measure_dict.get(str(n), {})
        
        sub_dict = measure_dict[str(n)].get('COVARS_{}'.format(covars_name), {})
        
        if covars_name == covars_dict.keys()[0]:
        # (note that mean and std are independent of the covariates and 
        # therefore are only measured for the first covars combination)
            #----- MEAN -------------------
            # Save the mean values for each region
            if not '{}_regional_mean'.format(measure_name) in measure_dict[str(n)].keys():
                measure_dict[str(n)]['{}_regional_mean'.format(measure_name)] = df[names].mean(axis=0).values
            
            #----- STD --------------------
            if not '{}_regional_std'.format(measure_name) in measure_dict[str(n)].keys():
                measure_dict[str(n)]['{}_regional_std'.format(measure_name)] = df[names].std(axis=0).values

        #----- CORR W AGE -------------
        test_key = '{}_regional_corr_age_perm_p'.format(measure_name)
        
        if not test_key in sub_dict.keys():
            
            # Do the linear regression for each region
            regional_linregress_dict = regional_linregress(df, 
                                                            'age_scan', 
                                                            names, 
                                                            covars=covars,
                                                            n_perm=n_perm)
                            
            # Add these values to the measure_dict
            sub_dict = add_to_measure_dict(regional_linregress_dict, 
                                                            sub_dict, 
                                                            'age', 
                                                            measure_name, 
                                                            regional=True)
            
            # Make sure sub_dict goes back into measure_dict
            # in the right place
            measure_dict[str(n)]['COVARS_{}'.format(covars_name)] = sub_dict
            
    return measure_dict
    

def save_regional_correlations(measure_dict, n_perm=1000):
    '''
    A snazzy little function that correlates a bunch of regional values 
    **across regions** for all the regional measures in measure_dict 
    and saves the output in measure_dict
    '''
    import pandas as pd
    import numpy as np
    import itertools as it
    
    #----------------------------------------------------------------
    # Define the names and suffix dictionaries
    names_dict = { 308 : measure_dict['308']['aparc_names'],
                    68 : measure_dict['68']['aparc_names'],
                    34 : measure_dict['34']['aparc_names'] }
    
    #----------------------------------------------------------------
    # Define the covars dictionary
    covars_dict = { 'gender'      : ['male'],
                    'site'        : ['wbic', 'ucl'],
                    'gender_site' : ['male', 'wbic', 'ucl'],
                    'none'        : [] }

    #----------------------------------------------------------------
    # Get the gene_indices ready to go
    gene_indices = measure_dict['308']['gene_indices']
    
    #----------------------------------------------------------------
    # Define the mpm measure list, the network measures list and 
    # the graphs that have been processed
    first_sub_dict = measure_dict['308']['COVARS_none']
    measure_list = [ x.split('_regional_')[0] for x in first_sub_dict.keys() if x.endswith('_regional_corr_age_c14') ]
    measure_list.sort()
    
    first_graph_dict = measure_dict['308']['Graph_measures']
    network_measures_list = [ 'Degree', 'Closeness', 'AverageDist', 'Clustering' ]
    graph_list = [ x.split('Degree_')[1] for x in first_graph_dict.keys() if x.startswith('Degree') ]
    graph_list.sort()
    graph_list = [ 'CT_ALL_COVARS_ONES_COST_10']
    
    #----------------------------------------------------------------
    # Loop through all the combinations of regions, covariates
    # and MRI measures
    for measure_name, (n, names), (covars_name, covars) in it.product(measure_list,
                                                                        names_dict.items(),
                                                                        covars_dict.items()):
        
        #----------------------------------------------------------------
        # Print to screen what you're up to
        #print covars_name, n, measure_name
        if covars_name == covars_dict.keys()[0] and n == names_dict.keys()[0]:
            print '    {}'.format(measure_name)
            print '      Corr the different regional measures together'

        # Get the dictionaries
        sub_dict = measure_dict[str(n)]['COVARS_{}'.format(covars_name)]
        graph_dict = measure_dict[str(n)]['Graph_measures']
        
        df = pd.DataFrame( { 'CT_int14' : sub_dict['CT_regional_corr_age_c14'],
                'CT_slopeAge' : sub_dict['CT_regional_corr_age_m'],
                '{}_int14'.format(measure_name) : sub_dict['{}_regional_corr_age_c14'.format(measure_name)],
                '{}_slopeAge'.format(measure_name) : sub_dict['{}_regional_corr_age_m'.format(measure_name)] } )
            
        #----------------------------------------------------------------
        # Intercept at 14 vs slope with age (for the same measure)
        #----------------------------------------------------------------
        if not '{}_slopeAge_corr_int14_perm_p'.format(measure_name) in sub_dict.keys():
            sub_dict = corr_to_measure_dict(sub_dict, 
                                                    df, 
                                                    '{}_int14'.format(measure_name), 
                                                    '{}_slopeAge'.format(measure_name), 
                                                    x_label='int14',
                                                    y_label='{}_slopeAge'.format(measure_name),
                                                    n_perm=n_perm, 
                                                    overwrite=False)
        
        #----------------------------------------------------------------
        # Intercept at 14: CT vs MPM
        #----------------------------------------------------------------
        if not '{}_int14_corr_CT_int14_perm_p'.format(measure_name) in sub_dict.keys():
            sub_dict = corr_to_measure_dict(sub_dict, 
                                        df, 
                                        'CT_int14', 
                                        '{}_int14'.format(measure_name), 
                                        n_perm=n_perm, 
                                        overwrite=False)

        #----------------------------------------------------------------
        # Slope with age: CT vs MPM
        #----------------------------------------------------------------
        if not '{}_slopeAge_corr_CT_slopeAge_perm_p'.format(measure_name) in sub_dict.keys():
            sub_dict = corr_to_measure_dict(sub_dict, 
                                        df, 
                                        'CT_slopeAge', 
                                        '{}_slopeAge'.format(measure_name), 
                                        n_perm=n_perm, 
                                        overwrite=False)
                                    
        #----------------------------------------------------------------
        # Intercept at 14 vs PLS
        #----------------------------------------------------------------
        pls_keys = [ x for x in sub_dict.keys() if len(x) == 4 and x.startswith('PLS') ]
        for pls in pls_keys:
            if n == 308:
                df['PLS1'] = sub_dict['PLS1_with99s']
                df['PLS2'] = sub_dict['PLS2_with99s']
                gene_indices = measure_dict['308']['gene_indices']
                
            else:
                df['PLS1'] = sub_dict['PLS1']
                df['PLS2'] = sub_dict['PLS2']
                gene_indices = np.arange(len(df['PLS2']))
                
            if not '{}_corr_{}_int14_perm_p'.format(pls, measure_name) in sub_dict.keys():
                sub_dict = corr_to_measure_dict(sub_dict, 
                                            df.iloc[gene_indices], 
                                            '{}_int14'.format(measure_name), 
                                            pls,
                                            n_perm=n_perm, 
                                            overwrite=False)
                                    
        #----------------------------------------------------------------
        # Slope with age vs PLS
        #----------------------------------------------------------------
        pls_keys = [ x for x in sub_dict.keys() if len(x) == 4 and x.startswith('PLS') ]
        for pls in pls_keys:
            if not '{}_corr_{}_slopeAge_perm_p'.format(pls, measure_name) in sub_dict.keys():
                sub_dict = corr_to_measure_dict(sub_dict, 
                                            df.iloc[gene_indices], 
                                            '{}_slopeAge'.format(measure_name), 
                                            pls,
                                            n_perm=n_perm, 
                                            overwrite=False)
        
        for network_measure, graph in it.product(network_measures_list, graph_list):
            
            sub_graph_dict = measure_dict[str(n)]['COVARS_{}'.format(covars_name)].get('Graph_{}'.format(graph), {})

            df['network_measure'] = graph_dict['{}_{}'.format(network_measure, graph)]

            #----------------------------------------------------------------
            # Int at 14 with Network Measures:
            #    * Degree
            #    * Closeness
            #    * AverageDist
            #    * Clustering
            #----------------------------------------------------------------
            if not '{}_corr_{}_int14_perm_p'.format(network_measure, measure_name) in sub_graph_dict.keys():
                sub_graph_dict = corr_to_measure_dict(sub_graph_dict, 
                                                            df, 
                                                            '{}_int14'.format(measure_name), 
                                                            'network_measure',
                                                            x_label='{}_int14'.format(measure_name),
                                                            y_label='{}'.format(network_measure),
                                                            n_perm=n_perm, 
                                                            overwrite=False)
            
            #----------------------------------------------------------------
            # SlopeAge with Network Measures:
            #    * Degree
            #    * Closeness
            #    * AverageDist
            #    * Clustering
            #----------------------------------------------------------------
            if not '{}_corr_{}_slopeAge_perm_p'.format(network_measure, measure_name) in sub_graph_dict.keys():
                sub_graph_dict = corr_to_measure_dict(sub_graph_dict, 
                                            df, 
                                            '{}_slopeAge'.format(measure_name), 
                                            'network_measure',
                                            x_label='{}_slopeAge'.format(measure_name),
                                            y_label='{}'.format(network_measure),
                                            n_perm=n_perm, 
                                            overwrite=False)
                                            
            #----------------------------------------------------------------
            # PLS with Network Measures:
            #    * Degree
            #    * Closeness
            #    * AverageDist
            #    * Clustering
            #----------------------------------------------------------------
            pls_keys = [ x for x in sub_dict.keys() if len(x) == 4 and x.startswith('PLS') ]
            for pls in pls_keys:
                if not '{}_corr_{}_slopeAge_perm_p'.format(network_measure, pls) in sub_graph_dict.keys():
                    sub_graph_dict = corr_to_measure_dict(sub_graph_dict, 
                                            df.iloc[gene_indices], 
                                            pls, 
                                            'network_measure',
                                            x_label=pls,
                                            y_label='{}'.format(network_measure),
                                            n_perm=n_perm, 
                                            overwrite=False)

        # Write the sub_graph_dict back to the main measure_dict
        measure_dict[str(n)]['COVARS_{}'.format(covars_name)]['Graph_{}'.format(graph)] = sub_graph_dict
        
    # Write the sub_dict back to the main measure_dict
    measure_dict[str(n)]['COVARS_{}'.format(covars_name)] = sub_dict

    return measure_dict
    
def save_mediation_dict(measure_dict, sub_dict, measure_name, n_perm=1000):
    
    import pandas as pd
    from permutation_testing import permutation_correlation
    from scipy.stats import zscore
    
    # This saves the output of a mediation analysis 
    # between the independent variable (age_scan),
    # dependent variable (global cortical thickness),
    # mediated by M (measure_name)
    
    x = measure_dict['308']['age_scan']
    y = sub_dict['CT_global_mean']
    m = sub_dict['{}_global_mean'.format(measure_name)]
    
    # Now calculate the various models
    a_dict = permutation_correlation(zscore(x), zscore(m))
    b_dict = permutation_correlation(zscore(m), zscore(y))
    c_dict = permutation_correlation(zscore(x), zscore(y))
    cdash_dict = permutation_correlation(zscore(x), zscore(y), covars_orig=[zscore(m)])
    
    # Save the standardised slopes and their associated
    # p values in a dictionary
    med_dict = {}
    med_dict['a_m'] = a_dict['m']
    med_dict['b_m'] = b_dict['m']
    med_dict['c_m'] = c_dict['m']
    med_dict['cdash_m'] = cdash_dict['m']
    med_dict['a_p'] = a_dict['perm_p']
    med_dict['b_p'] = b_dict['perm_p']
    med_dict['c_p'] = c_dict['perm_p']
    med_dict['cdash_p'] = cdash_dict['perm_p']
    med_dict['indir'] = med_dict['a_m'] * med_dict['b_m']
    med_dict['frac_mediated'] = med_dict['indir'] * 100.0 / med_dict['c_m']
    
    sub_dict['{}_mediation_age_CT'.format(measure_name)] = med_dict
    
    return sub_dict
    
    
def save_network_values(measure_dict, graph_dict):
    '''
    An easy little function that translates the values
    in the separate NodalMeasures and GlobalMeasures
    graph dictionaries and saves them to measure_dict
    '''
    import pandas as pd
    import numpy as np
    import itertools as it
    
    print '    Load network measures'

    # Loop through the different graphs and the different parcellations
    G_name_list = [ x for x in graph_dict.keys() if x[-3] == '_' ]
    n_list = [ 308, 68, 34 ]
    
    for G_name, n in it.product(G_name_list, n_list):
    
        # Create the sub dictionary that you're going to fill
        sub_dict = measure_dict[str(n)].get('Graph_measures', {})
        
        # Name a test key - skip the measure if it's already in there
        test_key = 'Degree_{}'.format(G_name, n)
        
        if not test_key in sub_dict.keys():
            
            nodal_dict = graph_dict['{}_NodalMeasures'.format(G_name)]
            global_dict = graph_dict['{}_GlobalMeasures'.format(G_name)]
            
            # Read in the nodal dict as a data frame
            nodal_df = pd.DataFrame(nodal_dict)
            nodal_df['name'] = measure_dict['308']['aparc_names']
            nodal_df['hemi'] = measure_dict['308']['hemi']
            nodal_df['dk_region'] = [ x.split('_')[1] for x in measure_dict['308']['aparc_names'] ]

            # Sort out the grouping and suffices
            if n == 68:
                grouped = nodal_df.groupby(['hemi', 'dk_region'])
            elif n == 34:
                grouped = nodal_df.groupby(['dk_region'])
            else:
                grouped = nodal_df.groupby('name', sort=False)
            
            degree_list = []
            pc_list = []
            closeness_list = []
            clustering_list = []
            average_dist_list = []
            total_dist_list = []
            interhem_prop_list = []
            betweenness_list = []
            
            for name, data in grouped:
                degree_list += [ np.percentile(data['degree'], 50) ]
                pc_list += [ np.percentile(data['pc'], 50) ]
                closeness_list += [ np.percentile(data['closeness'], 50) ]
                betweenness_list += [ np.percentile(data['betweenness'], 50) ]
                clustering_list += [ np.percentile(data['clustering'], 50) ]
                average_dist_list += [ np.percentile(data['average_dist'], 50) ]
                total_dist_list += [ np.percentile(data['total_dist'], 50) ]
                interhem_prop_list += [ np.percentile(data['interhem_prop'], 50) ]
                
            # Fill in the measure dict
            sub_dict['Degree_{}'.format(G_name)] = np.array(degree_list)
            sub_dict['PC_{}'.format(G_name)] = np.array(pc_list)
            sub_dict['Closeness_{}'.format(G_name)] = np.array(closeness_list)
            sub_dict['Betweenness_{}'.format(G_name)] = np.array(betweenness_list)
            sub_dict['Clustering_{}'.format(G_name)] = np.array(clustering_list)
            sub_dict['AverageDist_{}'.format(G_name)] = np.array(average_dist_list)
            sub_dict['TotalDist_{}'.format(G_name)] = np.array(total_dist_list)
            sub_dict['InterhemProp_{}'.format(G_name)] = np.array(interhem_prop_list)
                
            if n == 308:
                # Add in these last two that only make sense for n=308
                sub_dict['Module_{}'.format(G_name)] = nodal_dict['module'] + 1
                sub_dict['ShortestPath_{}'.format(G_name)] = nodal_dict['shortest_path']
                
                # Now put in the global measures
                sub_dict['Global_Clustering_{}'.format(G_name)] = global_dict['C']
                sub_dict['Global_Clustering_rand_{}'.format(G_name)] = global_dict['C_rand']
                sub_dict['Global_Modularity_{}'.format(G_name)] = global_dict['M']
                sub_dict['Global_Modularity_rand_{}'.format(G_name)] = global_dict['M_rand']
                sub_dict['Global_ShortestPath_{}'.format(G_name)] = global_dict['L']
                sub_dict['Global_ShortestPath_rand_{}'.format(G_name)] = global_dict['L_rand']
                sub_dict['Global_Efficiency_{}'.format(G_name)] = global_dict['E']
                sub_dict['Global_Efficiency_rand_{}'.format(G_name)] = global_dict['E_rand']
                sub_dict['Global_Assortativity_{}'.format(G_name)] = global_dict['a']
                sub_dict['Global_Assortativity_rand_{}'.format(G_name)] = global_dict['a_rand']
                sub_dict['Global_SmallWorld_{}'.format(G_name)] = global_dict['sigma']
                sub_dict['Global_SmallWorld_rand_{}'.format(G_name)] = global_dict['sigma_rand']

        measure_dict[str(n)]['Graph_measures'] = sub_dict

    return measure_dict
    
    
def save_global_values(measure_dict, measure_name, df, df_ct, n_perm=1000):
    '''
    A little script that saves a few global measures for each person
    and calculates some useful correlations for those measures along
    with a mediation analysis between age, ct and the measure
    '''
    print '      Corr global mean and std with age'
    print '      Calculate mediation of ct vs age'
    #----------------------------------------------------------------
    # Define the covars dictionary
    covars_dict = { 'gender'      : ['male'],
                    'site'        : ['wbic', 'ucl'],
                    'gender_site' : ['male', 'wbic', 'ucl'],
                    'none'        : [] }

    #----------------------------------------------------------------
    # Create or get the Global sub dictionary
    measure_dict['Global'] = measure_dict.get('Global', {})
    
    #----------------------------------------------------------------
    # Loop over the different covariates for the correlations
    for covars_name, covars in covars_dict.items():
        
        # Create the covars sub dict
        measure_dict['Global'] = measure_dict.get('Global', {})
        sub_dict = measure_dict['Global'].get('COVARS_{}'.format(covars_name), {})
    
        # Save the global mean and standard deviations for each person
        if not '{}_global_std'.format(measure_name) in sub_dict.keys():
            sub_dict['{}_global_mean'.format(measure_name)] = df['Global'].values
            sub_dict['{}_global_std'.format(measure_name)] = df['Global_std'].values

        # CORR GLOBAL MEAN WITH AGE
        if not '{}_global_mean_corr_age_perm_p'.format(measure_name) in sub_dict.keys():
            sub_dict = corr_to_measure_dict(sub_dict,
                                                df, 
                                                'age_scan', 
                                                'Global',
                                                x_label='age',
                                                y_label='{}_global_mean'.format(measure_name),
                                                covars=covars, 
                                                n_perm=n_perm, 
                                                overwrite=False)
                                                   
        # CORR GLOBAL STD WITH AGE
        if not '{}_global_std_corr_age_perm_p'.format(measure_name) in sub_dict.keys():
            sub_dict = corr_to_measure_dict(sub_dict, 
                                                df, 
                                                'age_scan', 
                                                'Global_std',
                                                x_label='age',
                                                y_label='{}_global_std'.format(measure_name),
                                                covars=covars, 
                                                n_perm=n_perm, 
                                                overwrite=False)

        # MEDIATION ANALYSIS
        if ( not '{}_mediation_age_CT'.format(measure_name) in sub_dict.keys()
                and not measure_name == 'CT' ):
            sub_dict = save_mediation_dict(measure_dict, 
                                                sub_dict, 
                                                measure_name, 
                                                n_perm=n_perm)
            
        # Write sub_dict back to the main measure_dict
        measure_dict['Global']['COVARS_{}'.format(covars_name)] = sub_dict
    
    return measure_dict
    

def add_to_measure_dict(linregress_dict, 
                            measure_dict, 
                            x_name, 
                            y_name, 
                            regional=False):
    '''
    A little function to save each of the output from either the 
    regional_linregress_dict or the linregress_dict
    to the measure_dictionary
    '''
    if regional:
        key = '{}_regional_corr_{}'.format(y_name, x_name)
    else:
        key = '{}_corr_{}'.format(y_name, x_name)
    
    # Now put the different keys of the linregress_dict
    # at the end of the key, and fill in the measure_dict
    # with the appropriate value
    for k in linregress_dict.keys():
        measure_dict['{}_{}'.format(key, k)] = linregress_dict[k]
            
    return measure_dict


def corr_to_measure_dict(sub_dict, 
                            df, 
                            x_var, 
                            y_var,
                            x_label=None,
                            y_label=None,
                            covars=[], 
                            n_perm=1000, 
                            overwrite=False):
    '''
    A function that takes a data frame, x and y variables,
    and a list of covariates if necessary, and runs a permutation
    multiple regression then adds the output from the
    model fit to the measure_dict
    
    INPUTS:
        sub_dict -------------- so you know which dictionary you're adding to
                                  (usually a sub dictionary from the main
                                   measure_dict)
        df -------------------- data frame
        x_var ----------------- name of x variable in data frame
        y_var ----------------- name of y variable in data frame
        x_label --------------- name of x var for measure_dict entry
        y_label --------------- name of y var for measure_dict entry
        suff ------------------ suffix to go at the end of measure_dict entry
        covars ---------------- list of covar names, all of which need to
                                  be in the data frame
        covars_name ----------- name of covars for measure_dict entry
        n_perm ---------------- number of permutations to complete
        overwrite ------------- True if you want to overwrite any existing
                                  entry, False if you're going to skip
                                  the ols command if it's already there
                                 
    RETURNS:
        measure_dict with entries:
            <y_label>_corr_<x_label>_<linregress_measure>_<suff>
            or
            <y_label>_corr_<x_label>_covars_<covars_name>_<linregress_measure>_<suff>
            
            Where linregress measure can be:
                m, c, c14, r, p, perm_p
            
            For example:
                MT_cortexAv_slopeAge_corr_int14_covars_site_p_308
    '''
    from permutation_testing import permutation_correlation, ols_correlation
    
    # Start by sorting out the labels for when you add 
    # these linear regression values to the measure_dict
    # You never know - if it's already in there you may not
    # have to actually do the regression!!
    if x_label is None:
        x_label = x_var
    
    if y_label is None:
        y_label = y_var
        
    # We're going to check to see if the perm_p values is already
    # there, and if not, then do the correlation
    key = '{}_corr_{}_perm_p'.format(y_label, x_label)

    # Don't do this part if the key is already there
    # (unless you've said to overwrite the entry!)
    if not key in sub_dict.keys() or overwrite:
        
        # Set up your covars_list
        covars_list = []
        wbic_i = None
        
        for i, covar in enumerate(covars):
            covars_list += [df[covar].values]
            if covar == 'wbic':
                wbic_i = i

        # Run the permutation test
        linregress_dict = permutation_correlation(df[x_var].values,
                                                    df[y_var].values, 
                                                    covars_orig=covars_list,
                                                    n_perm=n_perm)
            
        # Run the regular ols to get the correct intercept values
        results, c, c14 = ols_correlation(df[x_var].values,
                                                    df[y_var].values, 
                                                    covars=covars_list,
                                                    wbic_covars_index=wbic_i)
        
        # Add these values to the linregress_dict
        # (which means overwriting 'c' and adding in 'c14')
        linregress_dict['c'] = c
        linregress_dict['c14'] = c14
        
        # And go ahead and add the values
        sub_dict = add_to_measure_dict(linregress_dict, 
                                            sub_dict, 
                                            x_label,
                                            y_label, 
                                            regional=False)
                    
        
    return sub_dict