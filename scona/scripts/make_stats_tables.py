#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# 27th July 2015
# Contact: kw401@cam.ac.uk
#
# This code creates the tables of statistical results comparing the different
# cohorts for the NSPN Cortical Myelination manuscript
#=============================================================================

#=============================================================================
# IMPORTS
#-----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import requests
from glob import glob

from NSPN_functions import *

#=============================================================================
# FUNCTIONS
#=============================================================================
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

def format_p(x):
    '''
    If p is less than 0.001 then return a string of <.001
    '''
    p = '{:.3f}'.format(x)
    p = '$P = {}$'.format(p[1:])
    if x < 0.001:
        p = '$P < .001$'
    return p
        
#=============================================================================
# FIGURES 1 and 2 CREATE DICT
#-----------------------------------------------------------------------------
def create_dict_figure2(measure_dict_dict, mpm='MT', covars_name='none'):
    
    name_dict = { 'DISCOVERY_{}'.format(mpm)  : 'Discovery',
                  'VALIDATION_{}'.format(mpm) : 'Validation',
                  'COMPLETE_{}'.format(mpm)   : 'Complete' }

    table_dict = {}

    for cohort in name_dict.keys():
        measure_dict = measure_dict_dict[cohort]
        
        sub_dict = measure_dict['308']['COVARS_{}'.format(covars_name)]
        global_dict = measure_dict['Global']['COVARS_{}'.format(covars_name)]
        
        print '==={}==='.format(name_dict[cohort])
        
        data_list = []
        col_list = []
        
        #=======================================================
        # Global CT vs Age
        #=======================================================    
        key_root = 'CT_global_mean_corr_age'
        m = global_dict['{}_m'.format(key_root)]
        r = global_dict['{}_r'.format(key_root)]
        p = global_dict['{}_perm_p'.format(key_root)]

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Global CT vs Age}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Global MT vs Age
        #=======================================================    
        key_root = 'MT_projfrac+030_global_mean_corr_age'
        m = global_dict['{}_m'.format(key_root)]
        r = global_dict['{}_r'.format(key_root)]
        p = global_dict['{}_perm_p'.format(key_root)]

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Global MT vs Age}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # MT at 14 vs CT at 14
        #=======================================================    
        key_root = 'MT_projfrac+030_int14_corr_CT_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]
 
        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\MT at 14 vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # dMT vs dCT
        #=======================================================    
        key_root = 'MT_projfrac+030_slopeAge_corr_CT_slopeAge'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]
 
        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs $\\Delta$CT}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta CT vs CT at 14
        #=======================================================    
        key_root = 'CT_slopeAge_corr_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta MT vs MT at 14
        #=======================================================    
        key_root = 'MT_projfrac+030_slopeAge_corr_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs MT at 14}}']
        col_list += [ '' ] * 2

        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

    return table_dict, col_list

    
#=============================================================================
# FIGURE 3 CREATE DICT
#-----------------------------------------------------------------------------
def create_dict_figure3(measure_dict_dict, cohort_list=['DISCOVERY_ALL', 'VALIDATION_ALL', 'COMPLETE_ALL']):
    name_dict = { 'DISCOVERY_EXCLBAD' : 'Discovery ExclBad',
                  'VALIDATION_EXCLBAD' : 'Validation ExclBad',
                  'COMPLETE_EXCLBAD' : 'Complete ExclBad',
                  'DISCOVERY_ALL' : 'Discovery',
                  'VALIDATION_ALL' : 'Validation',
                  'COMPLETE_ALL' : 'Complete'}

    table_dict = {}

    for cohort in cohort_list:
        measure_dict = measure_dict_dict[cohort]
        print '==={}==='.format(name_dict[cohort])

        gene_indices = measure_dict['gene_indices']
        
        data_list = []
        col_list = []
        
        #=======================================================
        # PLS1 vs CT at 14
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['CT_all_slope_age_at14'][gene_indices], 
                                    measure_dict['PLS1'][gene_indices])
        
        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS1 vs delta CT
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['CT_all_slope_age'][gene_indices], 
                                    measure_dict['PLS1'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs $\\Delta$CT}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # PLS1 vs MT at 14
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['MT_projfrac+030_all_slope_age_at14'][gene_indices], 
                                    measure_dict['PLS1'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs MT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS1 vs delta MT
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['MT_projfrac+030_all_slope_age'][gene_indices], 
                                    measure_dict['PLS1'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs $\\Delta$MT}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs CT at 14
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['CT_all_slope_age_at14'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs delta CT
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['CT_all_slope_age'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs $\\Delta$CT}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs MT at 14
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['MT_projfrac+030_all_slope_age_at14'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs MT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs delta MT
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['MT_projfrac+030_all_slope_age'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs $\\Delta$MT}}']
        col_list += [ '' ] * 2

        #=======================================================    
        # Save to a dictionary
        #=======================================================    
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

    return table_dict, col_list

#=============================================================================
# FIGURE 4 CREATE DICT
#-----------------------------------------------------------------------------
def create_dict_figure4(measure_dict_dict, cohort_list=['DISCOVERY_ALL', 'VALIDATION_ALL', 'COMPLETE_ALL']):
    name_dict = { 'DISCOVERY_EXCLBAD' : 'Discovery ExclBad',
                  'VALIDATION_EXCLBAD' : 'Validation ExclBad',
                  'COMPLETE_EXCLBAD' : 'Complete ExclBad',
                  'DISCOVERY_ALL' : 'Discovery',
                  'VALIDATION_ALL' : 'Validation',
                  'COMPLETE_ALL' : 'Complete'}

    table_dict = {}

    for cohort in cohort_list:
        measure_dict = measure_dict_dict[cohort]
        print '==={}==='.format(name_dict[cohort])

        gene_indices = measure_dict['gene_indices']
        
        data_list = []
        col_list = []
        
        #=======================================================
        # Delta CT vs Degree
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Degree_CT_covar_ones_all_COST_10'], 
                                    measure_dict['CT_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta MT vs Degree
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Degree_CT_covar_ones_all_COST_10'], 
                                    measure_dict['MT_projfrac+030_all_slope_age'])

        print m, r, p_perm
        
        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-6}}$'.format(m*1000000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs Degree
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Degree_CT_covar_ones_all_COST_10'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta CT vs Closeness
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Closeness_CT_covar_ones_all_COST_10'], 
                                    measure_dict['CT_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs Closeness}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta MT vs Closeness
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Closeness_CT_covar_ones_all_COST_10'], 
                                    measure_dict['MT_projfrac+030_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # PLS2 vs Closeness
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Closeness_CT_covar_ones_all_COST_10'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # Delta CT vs Average Distance
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['AverageDist_CT_covar_ones_all_COST_10'], 
                                    measure_dict['CT_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs Av Dist}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta MT vs Average Distance
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['AverageDist_CT_covar_ones_all_COST_10'], 
                                    measure_dict['MT_projfrac+030_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-6}}$'.format(m*1000000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs Av Dist}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs Average Distance
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['AverageDist_CT_covar_ones_all_COST_10'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs Av Dist}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # Delta CT vs Clustering
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Clustering_CT_covar_ones_all_COST_10'], 
                                    measure_dict['CT_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs Av Dist}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta MT vs Clustering
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Clustering_CT_covar_ones_all_COST_10'], 
                                    measure_dict['MT_projfrac+030_all_slope_age'])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-6}}$'.format(m*1000000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs Av Dist}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs Clustering
        #=======================================================    
        m, c, r, p, sterr, p_perm = permutation_correlation(measure_dict['Clustering_CT_covar_ones_all_COST_10'][gene_indices], 
                                    measure_dict['PLS2'][gene_indices])

        data_list += [ '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs Av Dist}}']
        col_list += [ '' ] * 2
        
        #=======================================================    
        # Save to a dictionary
        #=======================================================    
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

    return table_dict, col_list
    
    
#=============================================================================
# VON ECONOMO CREATE DICT
#-----------------------------------------------------------------------------
def create_dict_voneconomo(measure_dict_dict, cohort_list=['DISCOVERY_ALL', 'VALIDATION_ALL', 'COMPLETE_ALL']):
    name_dict = { 'DISCOVERY_EXCLBAD' : 'Discovery ExclBad',
                  'VALIDATION_EXCLBAD' : 'Validation ExclBad',
                  'COMPLETE_EXCLBAD' : 'Complete ExclBad',
                  'DISCOVERY_ALL' : 'Discovery',
                  'VALIDATION_ALL' : 'Validation',
                  'COMPLETE_ALL' : 'Complete'}

    table_dict = {}

    for cohort in cohort_list:
        measure_dict = measure_dict_dict[cohort]
        print '==={}==='.format(name_dict[cohort])
        
        data_list = []
        col_list = []
        
        gene_indices = measure_dict['gene_indices']

        #=======================================================
        # CT at 14 
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['CT_all_slope_age_at14'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.1f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{CT at 14}}']
        col_list += [ '' ]
        
        #=======================================================
        # Delta CT 
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['CT_all_slope_age'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{$\\Delta$CT vs Age}}']
        col_list += [ '' ]

        #=======================================================
        # MT at 14 
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['MT_projfrac+030_all_slope_age_at14'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.1f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{MT at 14}}']
        col_list += [ '' ]
        
        #=======================================================
        # Delta MT 
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['MT_projfrac+030_all_slope_age'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{$\\Delta$MT vs Age}}']
        col_list += [ '' ]

        #=======================================================
        # PLS1
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo_genes'],
                                                                measure_dict['PLS1_usable'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{PLS1}}']
        col_list += [ '' ]

        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list
        
        #=======================================================
        # PLS2
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo_genes'],
                                                                measure_dict['PLS2_usable'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{PLS2}}']
        col_list += [ '' ]

        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list
        
        #=======================================================
        # Degree 
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['Degree_CT_covar_ones_all_COST_10'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{Degree}}']
        col_list += [ '' ]
        
        #=======================================================
        # Closeness
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['Closeness_CT_covar_ones_all_COST_10'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{Closeness}}']
        col_list += [ '' ]

        #=======================================================
        # Average Distance
        #=======================================================    
        results, p_perm = permutation_multiple_correlation(measure_dict['von_economo'],
                                                                measure_dict['AverageDist_CT_covar_ones_all_COST_10'], 
                                                                covars=[], 
                                                                n_perm=1000,
                                                                categorical=True)
        
        data_list += [ 'F[{:1.0f},{:1.0f}] = {:2.2f}'.format(results.df_model,
                                                                results.df_resid,
                                                                results.fvalue) ]
        data_list += [ format_p(p_perm) ]
        
        col_list += [ '\\multirow{2}{*}{\\textbf{Average Distance (mm)}}']
        col_list += [ '' ]


    return table_dict, col_list
    
    
#=============================================================================
# MAKE DICT INTO A PANDAS DATA FRAME
#-----------------------------------------------------------------------------
def make_table_df(table_dict, col_list):

    name_dict = { 'DISCOVERY_EXCLBAD' : 'Discovery ExclBad',
                  'VALIDATION_EXCLBAD' : 'Validation ExclBad',
                  'COMPLETE_EXCLBAD' : 'Complete ExclBad',
                  'DISCOVERY_ALL' : 'Discovery',
                  'VALIDATION_ALL' : 'Validation',
                  'COMPLETE_ALL' : 'Complete'}
                  
    table_df = pd.DataFrame(table_dict, index=col_list)
    group_list = [ x for x in table_df.columns if 'Discovery' in x ]
    group_list += [ x for x in table_df.columns if 'Validation' in x ]
    group_list += [ x for x in table_df.columns if 'Complete' in x ]
    
    table_df = table_df.loc[:, group_list]

    return table_df
    
#=============================================================================
# WRITE DATA FRAME TO A LATEX FILE
#-----------------------------------------------------------------------------
def write_latex_table(table_df, table_file, caption):
    latex_table = table_df.to_latex(longtable=True, 
                                        index=True, 
                                        escape=False)

    latex_table = latex_table.replace('\\multirow', '\\rule{0pt}{4ex} \\multirow')
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
        
#=============================================================================
# PUT IT ALL TOGETHER
#-----------------------------------------------------------------------------
def make_stats_table_figure2(measure_dict_dict, 
                                            paper_dir, covars_name='none'):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Cohort comparison: Correlations between CT, MT with age'
    
    table_dir = os.path.join(paper_dir, 'STATS_TABLES', 'COVARS_{}'.format(covars_name))
    table_file = os.path.join(table_dir, 'Stats_Table_Figure2.tex')

    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
        
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_figure2(measure_dict_dict, mpm='MT', covars_name='none')
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)
    
def make_replication_table_figure3(measure_dict_dict, 
                                        paper_dir, 
                                        cohort_list=['DISCOVERY_ALL', 'VALIDATION_ALL', 'COMPLETE_ALL'], 
                                        cohort_list_name='THREE_ALL'):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Cohort comparison: Correlations with PLS scores'
    
    table_dir = os.path.join(paper_dir, cohort_list_name)
    table_file = os.path.join(table_dir, 'Replication_Table_Figure3.tex')

    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_figure3(measure_dict_dict, cohort_list)
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)
    
def make_replication_table_figure4(measure_dict_dict, 
                                        paper_dir, 
                                        cohort_list=['DISCOVERY_ALL', 'VALIDATION_ALL', 'COMPLETE_ALL'], 
                                        cohort_list_name='THREE_ALL'):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Cohort comparison: Correlations with network measures'
    
    table_dir = os.path.join(paper_dir, cohort_list_name)
    table_file = os.path.join(table_dir, 'Replication_Table_Figure4.tex')

    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_figure4(measure_dict_dict, cohort_list)
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)
    
def make_replication_table_voneconomo(measure_dict_dict, paper_dir, cohort_list=['DISCOVERY_ALL', 'VALIDATION_ALL', 'COMPLETE_ALL'], cohort_list_name='THREE_ALL'):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Nodal measures by von Economo type'

    table_dir = os.path.join(paper_dir, cohort_list_name)
    table_file = os.path.join(table_dir, 'VonEconomo_Table.tex')

    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
        
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_voneconomo(measure_dict_dict, cohort_list)
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)
    
# DONE - well done :)
