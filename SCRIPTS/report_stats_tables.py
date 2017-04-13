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
    p = '\\textit{{P}} = {}'.format(p[1:])
    if x < 0.001:
        p = '\\textit{P} $<$ .001'
    return p
        
def format_rsq(x):
    '''
    If rsq is less than 0.01 then return a string of <.01
    '''
    rsq = '{:.2f}'.format(x)
    rsq = '\\textit{{r\\textsuperscript{{2}}}} = {:2.2f}'.format(x)
    if x < 0.01:
        rsq = '\\textit{{r\\textsuperscript{{2}}}} $<$ 0.01'
    return rsq
        
#=============================================================================
# FIGURES 2 CREATE DICT
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
        
        data_list = []
        col_list = []
        
        #=======================================================
        # Global CT vs Age
        #=======================================================    
        key_root = 'CT_global_mean_corr_age'
        m = global_dict['{}_m'.format(key_root)]
        r = global_dict['{}_r'.format(key_root)]
        p = global_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
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

        data_list += [ format_rsq(r**2) ]
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
 
        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{MT at 14 vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # dMT vs dCT
        #=======================================================    
        key_root = 'MT_projfrac+030_slopeAge_corr_CT_slopeAge'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]
 
        data_list += [ format_rsq(r**2) ]
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

        data_list += [ format_rsq(r**2) ]
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

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs MT at 14}}']
        col_list += [ '' ] * 2

        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

    return table_dict, col_list

    
#=============================================================================
# FIGURE 3 CREATE DICT
#-----------------------------------------------------------------------------
def create_dict_figure3(measure_dict_dict, mpm='MT', covars_name='none'):
    
    name_dict = { 'DISCOVERY_{}'.format(mpm)  : 'Discovery',
                  'VALIDATION_{}'.format(mpm) : 'Validation',
                  'COMPLETE_{}'.format(mpm)   : 'Complete' }

    table_dict = {}

    for cohort in name_dict.keys():
        measure_dict = measure_dict_dict[cohort]
        
        sub_dict = measure_dict['308']['COVARS_{}'.format(covars_name)]
        
        data_list = []
        col_list = []
        
        #=======================================================
        # PLS1 vs CT at 14
        #=======================================================    
        key_root = 'PLS1_corr_CT_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]
        
        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS1 vs delta CT
        #=======================================================    
        key_root = 'PLS1_corr_CT_slopeAge'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs $\\Delta$CT}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # PLS1 vs MT at 14
        #=======================================================    
        key_root = 'PLS1_corr_MT_projfrac+030_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs MT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS1 vs delta MT
        #=======================================================    
        key_root = 'PLS1_corr_MT_projfrac+030_slopeAge'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs $\\Delta$MT}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs CT at 14
        #=======================================================    
        key_root = 'PLS2_corr_CT_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs CT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs delta CT
        #=======================================================    
        key_root = 'PLS2_corr_CT_slopeAge'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs $\\Delta$CT}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs MT at 14
        #=======================================================    
        key_root = 'PLS2_corr_MT_projfrac+030_int14'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.3f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs MT at 14}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS2 vs delta MT
        #=======================================================    
        key_root = 'PLS2_corr_MT_projfrac+030_slopeAge'
        m = sub_dict['{}_m'.format(key_root)]
        r = sub_dict['{}_r'.format(key_root)]
        p = sub_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
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
def create_dict_figure4(measure_dict_dict, mpm='MT', covars_name='none'):

    name_dict = { 'DISCOVERY_{}'.format(mpm)  : 'Discovery',
                  'VALIDATION_{}'.format(mpm) : 'Validation',
                  'COMPLETE_{}'.format(mpm)   : 'Complete' }

    table_dict = {}

    for cohort in name_dict.keys():
        measure_dict = measure_dict_dict[cohort]
        
        sub_dict = measure_dict['308']['COVARS_{}'.format(covars_name)]
        graph_dict = sub_dict['Graph_CT_ALL_COVARS_ONES_COST_10']
        
        data_list = []
        col_list = []
        
        #=======================================================
        # CT14 vs Degree
        #=======================================================    
        key_root = 'Degree_corr_CT_int14'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{CT at 14 vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # CT14 vs Closeness
        #=======================================================    
        key_root = 'Closeness_corr_CT_int14'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{CT at 14 vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # Delta CT vs Degree
        #=======================================================    
        key_root = 'Degree_corr_CT_slopeAge'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{3}}$'.format(m/1000.0) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Delta CT vs Closeness
        #=======================================================    
        key_root = 'Closeness_corr_CT_slopeAge'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$CT vs Closeness}}']
        col_list += [ '' ] * 2

        #=======================================================
        # MT at 14 vs Degree
        #=======================================================    
        key_root = 'Degree_corr_MT_projfrac+030_int14'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{MT at 14 vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # MT at 14 vs Closeness
        #=======================================================    
        key_root = 'Closeness_corr_MT_projfrac+030_int14'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{MT at 14 vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # Delta MT vs Degree
        #=======================================================    
        key_root = 'Degree_corr_MT_projfrac+030_slopeAge'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}$\\times10^{{3}}$'.format(m/1000.0) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs Degree}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # Delta MT vs Closeness
        #=======================================================    
        key_root = 'Closeness_corr_MT_projfrac+030_slopeAge'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{$\\Delta$MT vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # PLS1 vs Degree
        #=======================================================    
        key_root = 'Degree_corr_PLS1'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.2f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs Degree}}']
        col_list += [ '' ] * 2

        #=======================================================
        # PLS1 vs Closeness
        #=======================================================    
        key_root = 'Closeness_corr_PLS1'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000.0) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS1 vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # PLS2 vs Degree
        #=======================================================    
        key_root = 'Degree_corr_PLS2'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}'.format(m) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs Degree}}']
        col_list += [ '' ] * 2
        
        #=======================================================
        # PLS2 vs Closeness
        #=======================================================    
        key_root = 'Closeness_corr_PLS2'
        m = graph_dict['{}_m'.format(key_root)]
        r = graph_dict['{}_r'.format(key_root)]
        p = graph_dict['{}_perm_p'.format(key_root)]

        data_list += [ format_rsq(r**2) ]
        data_list += [ format_p(p) ]
        data_list += [ '$\\beta$ = {:2.1f}$\\times10^{{-3}}$'.format(m*1000.0) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{PLS2 vs Closeness}}']
        col_list += [ '' ] * 2
        
        #=======================================================    
        # Save to a dictionary
        #=======================================================    
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

    return table_dict, col_list
    
    
#=============================================================================
# NETWORKS DICT
#-----------------------------------------------------------------------------
def create_dict_network(measure_dict_dict, mpm='MT'):
    
    name_dict = { 'DISCOVERY_{}'.format(mpm)  : 'Discovery',
                  'VALIDATION_{}'.format(mpm) : 'Validation',
                  'COMPLETE_{}'.format(mpm)   : 'Complete' }

    table_dict = {}

    for cohort in name_dict.keys():
        measure_dict = measure_dict_dict[cohort]
        
        graph_dict = measure_dict['308']['Graph_measures']
        
        data_list = []
        col_list = []
        
        #=======================================================
        # Assortativity
        #=======================================================    
        key_root = 'Global_Assortativity_CT_ALL_COVARS_ONES_COST_10'
        real = graph_dict[key_root]
        rand = graph_dict[key_root.replace('CT_ALL', 'rand_CT_ALL')]

        data_list += [ '{:2.2f}'.format(real) ]
        data_list += [ 'rand: {:2.2f}'.format(np.percentile(rand, 50)) ]
        data_list += [ '95\\% CI [{:2.2f}, {:2.2f}]'.format(np.percentile(rand, 5),
                                                           np.percentile(rand, 95) ) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Assortativity}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Clustering
        #=======================================================    
        key_root = 'Global_Clustering_CT_ALL_COVARS_ONES_COST_10'
        real = graph_dict[key_root]
        rand = graph_dict[key_root.replace('CT_ALL', 'rand_CT_ALL')]

        data_list += [ '{:2.2f}'.format(real) ]
        data_list += [ 'rand: {:2.2f}'.format(np.percentile(rand, 50)) ]
        data_list += [ '  95\\% CI [{:2.2f}, {:2.2f}]'.format(np.percentile(rand, 5),
                                                           np.percentile(rand, 95) ) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Clustering}}']
        col_list += [ '' ] * 2

        #=======================================================
        # Efficiency
        #=======================================================    
        key_root = 'Global_Efficiency_CT_ALL_COVARS_ONES_COST_10'
        real = graph_dict[key_root]
        rand = graph_dict[key_root.replace('CT_ALL', 'rand_CT_ALL')]

        data_list += [ '{:2.2f}'.format(real) ]
        data_list += [ 'rand: {:2.2f}'.format(np.percentile(rand, 50)) ]
        data_list += [ '  95\\% CI [{:2.2f}, {:2.2f}]'.format(np.percentile(rand, 5),
                                                           np.percentile(rand, 95) ) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Efficiency}}']
        col_list += [ '' ] * 2
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

        #=======================================================
        # Modularity
        #=======================================================    
        key_root = 'Global_Modularity_CT_ALL_COVARS_ONES_COST_10'
        real = graph_dict[key_root]
        rand = graph_dict[key_root.replace('CT_ALL', 'rand_CT_ALL')]

        data_list += [ '{:2.2f}'.format(real) ]
        data_list += [ 'rand: {:2.2f}'.format(np.percentile(rand, 50)) ]
        data_list += [ '  95\\% CI [{:2.2f}, {:2.2f}]'.format(np.percentile(rand, 5),
                                                           np.percentile(rand, 95) ) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Modularity}}']
        col_list += [ '' ] * 2
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list
        
        #=======================================================
        # Shortest Path
        #=======================================================    
        key_root = 'Global_ShortestPath_CT_ALL_COVARS_ONES_COST_10'
        real = graph_dict[key_root]
        rand = graph_dict[key_root.replace('CT_ALL', 'rand_CT_ALL')]

        data_list += [ '{:2.2f}'.format(real) ]
        data_list += [ 'rand: {:2.2f}'.format(np.percentile(rand, 50)) ]
        data_list += [ '  95\\% CI [{:2.2f}, {:2.2f}]'.format(np.percentile(rand, 5),
                                                           np.percentile(rand, 95) ) ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Shortest Path}}']
        col_list += [ '' ] * 2
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list
        
        #=======================================================
        # Small World
        #=======================================================    
        key_root = 'Global_SmallWorld_CT_ALL_COVARS_ONES_COST_10'
        real = graph_dict[key_root]
        rand = graph_dict[key_root.replace('CT_ALL', 'rand_CT_ALL')]

        data_list += [ '{:2.2f}'.format(np.percentile(real, 50)) ]
        data_list += [ '  95\\% CI [{:2.2f}, {:2.2f}]'.format(np.percentile(real, 5),
                                                           np.percentile(real, 95) ) ]
        data_list += [ 'rand: 1.00' ]
        
        col_list += [ '\\multirow{3}{*}{\\textbf{Small World}}']
        col_list += [ '' ] * 2
        table_dict['\\textbf{{{}}}'.format(name_dict[cohort])] = data_list

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
def write_latex_table(table_df, table_file, caption, center=False):
    latex_table = table_df.to_latex(longtable=True, 
                                        index=True, 
                                        escape=False)

    latex_table = latex_table.replace('\\multirow', '\\rule{0pt}{4ex} \\multirow')
    latex_table = add_caption(latex_table, caption)
    
    if center:
        latex_table = latex_table.replace('llll', 'lccc')
            
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
    
def make_stats_table_figure3(measure_dict_dict, 
                                        paper_dir, covars_name='none'):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Cohort comparison: Correlations between MRI measures and PLS scores'
    
    table_dir = os.path.join(paper_dir, 'STATS_TABLES', 'COVARS_{}'.format(covars_name))
    table_file = os.path.join(table_dir, 'Stats_Table_Figure3.tex')

    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_figure3(measure_dict_dict, mpm='MT', covars_name='none')
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)
    
def make_stats_table_figure4(measure_dict_dict, 
                                    paper_dir, covars_name='none'):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Cohort comparison: Correlations with network measures'
    
    table_dir = os.path.join(paper_dir, 'STATS_TABLES', 'COVARS_{}'.format(covars_name))
    table_file = os.path.join(table_dir, 'Stats_Table_Figure4.tex')
    
    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_figure4(measure_dict_dict, mpm='MT', covars_name='none')
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption)
    
def make_stats_table_network(measure_dict_dict, paper_dir):

    #=============================================================================
    # GET STARTED BY SETTING THE CAPTION AND NAME YOUR OUTPUT FILE
    #-----------------------------------------------------------------------------
    caption = 'Cohort comparison: Global network measures'
    
    table_dir = os.path.join(paper_dir, 'STATS_TABLES', 'NETWORK')
    table_file = os.path.join(table_dir, 'Stats_Table_NetworkMeasures.tex')

    #=============================================================================
    # Make the output folder
    if not os.path.isdir(table_dir):
        os.makedirs(table_dir)
    
    #=============================================================================
    # Make the data frame
    #-----------------------------------------------------------------------------
    table_dict, col_list = create_dict_network(measure_dict_dict, mpm='MT')
    table_df = make_table_df(table_dict, col_list)
    write_latex_table(table_df, table_file, caption, center=True)
    

def make_stats_tables(measure_dict_dict, paper_dir, covars_name='none'):
    
    # A very simple little wrapper to make all the stats tables
   
    # Define the covars dictionary
    covars_dict = { 'gender'      : ['male'],
                    'site'        : ['wbic', 'ucl'],
                    'gender_site' : ['male', 'wbic', 'ucl'],
                    'none'        : [] }
                    
    print 'Making stats tables'
    
    for covars_name in covars_dict.keys():
        make_stats_table_figure2(measure_dict_dict, paper_dir, covars_name=covars_name)
        make_stats_table_figure3(measure_dict_dict, paper_dir, covars_name=covars_name)
        make_stats_table_figure4(measure_dict_dict, paper_dir, covars_name=covars_name)
    make_stats_table_network(measure_dict_dict, paper_dir)
        
# DONE - well done :)
