#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# 27th July 2015
# Contact: kw401@cam.ac.uk
#
# This code creates the tables of demographics comparing the different
# cohorts for the NSPN Cortical Myelination manuscript
#=============================================================================

#=============================================================================
# IMPORTS
#-----------------------------------------------------------------------------
import os
import pandas as pd
import datetime
import itertools as it
from glob import glob
import csv
import matplotlib.pylab as plt
import numpy as np
import sys

#=============================================================================
# FUNCTIONS
#-----------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Define your latex add in caption function
#------------------------------------------------------------------------------
def add_caption(latex_table, caption):
    '''
    Just add in a row on the second line down to include the caption (title)
    for this table
    '''
    latex_table_list = latex_table.split('\n')
    latex_table_list[0] = latex_table_list[0] + '\n\\caption*{{{}}} \\\\'.format(caption)         
    latex_table = '\n'.join(latex_table_list)

    return latex_table
    
   
   
#------------------------------------------------------------------------------
# And here's the main code
#------------------------------------------------------------------------------
   
def make_demo_table(data_dir, paper_dir):
    
    demographics_file = os.path.join(data_dir, 'DemographicData.csv')
    caption = 'Participant Demographics'

    table_file = os.path.join(paper_dir, 'STATS_TABLES', 'DemographicsTable.tex')
    
    df = pd.read_csv(demographics_file)
    
    name_dict = { (0.0, 0.0) : 'Remaining 2K',
                (1.0, 0.0) : 'Discovery',
                (0.0, 1.0) : 'Validation'}
    
    table_dict = {}
    

    for name, data in df.groupby(['discovery', 'validation']):
        
        data_list = []
        col_list = []
        
        #=======================================================
        # NUMBER OF PARTICIPANTS
        #=======================================================
        n = data['nspn_id'].count()
        data_list += [n]
        col_list += ['\\textbf{Number of participants}']
        
        #=======================================================
        # PERCENTAGE MALE
        #=======================================================
        n_male = data.loc[data['sex'] == 'Male', 'sex'].count()
        n_missing = data.loc[data['sex'].isnull(), 'sex'].shape[0]
        data_list += ['{:2.1f}\% male'.format(n_male * 100.0/n)]
        col_list += ['\\textbf{Gender}']
        
        #=======================================================
        # AGES (MEDIAN, IQR)
        #=======================================================
        if name[0] + name[1] == 0:
            age_var = 'age_hqp'
        else:
            age_var = 'age_scan'
        
        # Means and Stds - not used here but just in case
        mean_age = data.loc[data[age_var].notnull(), age_var].mean()
        std_age = data.loc[data[age_var].notnull(), age_var].std()
    
        # Missing values
        n_missing = data.loc[data[age_var].isnull(), age_var].shape[0]
            
        # Median and IQRs
        med_age = np.percentile(data.loc[data[age_var].notnull(), age_var], 50)
        upper_age = np.percentile(data.loc[data[age_var].notnull(), age_var], 75)
        lower_age = np.percentile(data.loc[data[age_var].notnull(), age_var], 25)

        data_list += ['{:2.1f}'.format(med_age) ]
        data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_age, upper_age)]
        
        col_list += ['\\multirow{2}{*}{\\textbf{Age (years)}}']
        col_list += ['']
        
        #=======================================================
        # FULL IQ (MEDIAN, IQR)
        #=======================================================
        # Means and Stds - not used here but just in case
        mean_iq = data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 'wasi_zz_iq_full2_iq'].mean()
        std_iq = data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 'wasi_zz_iq_full2_iq'].std()
    
        # Missing values
        n_missing = data.loc[data['wasi_zz_iq_full2_iq'].isnull(), 'wasi_zz_iq_full2_iq'].shape[0]
            
        # Median and IQRs
        med_iq = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                    'wasi_zz_iq_full2_iq'], 50)
        upper_iq = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                    'wasi_zz_iq_full2_iq'], 75)
        lower_iq = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                    'wasi_zz_iq_full2_iq'], 25)

        if n > 2000:
            data_list += ['\\multirow{2}{*}{NA}']
            data_list += ['']
        else:
            data_list += ['{:2.1f}'.format(med_iq) ]
            data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_iq, upper_iq)]
        
        col_list += ['\\multirow{2}{*}{\\textbf{IQ}}']
        col_list += ['']
    
        #=======================================================
        # HANDEDNESS (MEDIAN, IQR)
        #=======================================================
        # Means and Stds - not used here but just in case
        mean_ehi = data.loc[data['ehi_handedness_score'].notnull(), 'ehi_handedness_score'].mean()
        std_ehi = data.loc[data['ehi_handedness_score'].notnull(), 'ehi_handedness_score'].std()
    
        # Missing values
        n_missing = data.loc[data['ehi_handedness_score'].isnull(), 'ehi_handedness_score'].shape[0]
            
        # Median and IQRs
        med_ehi = np.percentile(data.loc[data['ehi_handedness_score'].notnull(), 
                                                    'ehi_handedness_score'], 50)
        upper_ehi = np.percentile(data.loc[data['ehi_handedness_score'].notnull(), 
                                                    'ehi_handedness_score'], 75)
        lower_ehi = np.percentile(data.loc[data['wasi_zz_iq_full2_iq'].notnull(), 
                                                    'ehi_handedness_score'], 25)

        if n > 2000:
            data_list += ['\\multirow{2}{*}{NA}']
            data_list += ['']
        else:
            data_list += ['{:2.1f}'.format(med_ehi) ]
            data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_ehi, upper_ehi)]
        
        col_list += ['\\multirow{2}{*}{\\textbf{Handedness}}']
        col_list += ['']
            
        #=======================================================
        # INDEX MULTIPLE DEPRIVATION (MEDIAN, IQR)
        #=======================================================
        # Means and Stds - not used here but just in case
        mean_imd = data.loc[data['imd_2007'].notnull(), 'imd_2007'].mean()
        std_imd = data.loc[data['imd_2007'].notnull(), 'imd_2007'].std()
    
        # Missing values
        n_missing = data.loc[data['imd_2007'].isnull(), 'imd_2007'].shape[0]
            
        # Median and IQRs
        med_imd = np.percentile(data.loc[data['imd_2007'].notnull(), 
                                                    'imd_2007'], 50)
        upper_imd = np.percentile(data.loc[data['imd_2007'].notnull(), 
                                                    'imd_2007'], 75)
        lower_imd = np.percentile(data.loc[data['imd_2007'].notnull(), 
                                                    'imd_2007'], 25)

        data_list += ['{:2.1f}'.format(med_imd) ]
        data_list += ['(IQR: {:2.1f}-{:2.1f})'.format(lower_imd, upper_imd)]
        
        col_list += ['\\multirow{2}{*}{\\textbf{IMD}}']
        col_list += ['']
    
        #=======================================================
        # ETHNICITY (% WHITE)
        n_white = data.loc[data['psq_a4_ethnic_group']==1, 'psq_a4_ethnic_group'].count()
        n_mixed = data.loc[data['psq_a4_ethnic_group']==2, 'psq_a4_ethnic_group'].count()
        n_asian = data.loc[data['psq_a4_ethnic_group']==3, 'psq_a4_ethnic_group'].count()
        n_black = data.loc[data['psq_a4_ethnic_group']==4, 'psq_a4_ethnic_group'].count()
        n_other = data.loc[data['psq_a4_ethnic_group']==5, 'psq_a4_ethnic_group'].count()
        n_declined = data.loc[data['psq_a4_ethnic_group']==6, 'psq_a4_ethnic_group'].count()
        n_known_missing = data.loc[data['psq_a4_ethnic_group']==999, 'psq_a4_ethnic_group'].count()
        n_missing = data.loc[data['psq_a4_ethnic_group'].isnull(), 'psq_a4_ethnic_group'].shape[0]

        
        data_list += [ '{:2.1f}\% White'.format(n_white*100.0/n) ]
        data_list += [ '{:2.1f}\% Asian'.format(n_asian*100.0/n) ]
        data_list += [ '{:2.1f}\% Black'.format(n_black*100.0/n) ]
        data_list += [ '{:2.1f}\% Mixed'.format(n_mixed*100.0/n) ]
        data_list += [ '{:2.1f}\% Other'.format(n_other*100.0/n) ]
        data_list += [ '{:2.1f}\% Declined to state'.format(n_declined*100.0/n) ]
            
        col_list += ['\\multirow{6}{*}{\\textbf{Ethnicity}}']
        col_list += [ '' ] *5
    
        table_dict['\\textbf{{{}}}'.format(name_dict[name])] = data_list

    table_df = pd.DataFrame(table_dict)
    table_df = pd.DataFrame(table_dict, index=col_list)
    table_df = table_df.loc[:, ['\\textbf{Discovery}', '\\textbf{Validation}', '\\textbf{Remaining 2K}']]
    
    latex_table = table_df.to_latex(longtable=True, 
                                        index=True, 
                                        escape=False)
    
    latex_table = latex_table.replace('llll', 'lccc')
    latex_table = add_caption(latex_table, caption)
    
    latex_header = '\n'.join([ '\\documentclass{article}', 
                            '\\usepackage{booktabs}', 
                            '\\usepackage[a4paper, left={1cm}, right={1cm}, top={1.5cm}, bottom={1.5cm}, portrait]{geometry}',
                            '\\usepackage{longtable}',
                            '\\usepackage{array}',
                            '\\usepackage{multirow}'
                            '\\begin{document}',
                            '' ])
    
    latex_footer = '\n\\end{document}\n'
    
    # Write the file to the table_filename
    with open(table_file, 'w') as f:
        f.write(latex_header)
        f.write(latex_table)
        f.write(latex_footer)
        
    # Write the file to the table_filename without the latex header and footer
    with open(table_file.replace('.tex', '.txt'), 'w') as f:
        f.write(latex_table)