#!/usr/bin/env python

'''
This code creates the supplemental figures for
the NSPN Cortical Myelination manuscript and seeks 
to combine our millions of supplemental figures
into something a little more paletable (even though 
there's now a WHOLE bunch of information presented
at once!)

It can't be run on the BCNI linux cluster because 
(I think) latex is not up to date and has therefore
been run locally on KJW's windows laptop.
'''

#===================================================================
# IMPORTS
#===================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import sys

#===================================================================
# FUNCTIONS
#===================================================================
def format_demotable(fname):
    
    # Read in the file
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = [ line.strip() for line in lines ]
    
    # There are a few latex tricks that don't work in matplotlib so we
    # have to tidy them up here
    lines[0] = lines[0].replace('longtable', 'tabular')
    del(lines[1])
    lines[1] = '\\hline'
    lines[3] = '\\hline'
    del(lines[4:12])
    lines[-1] = lines[-1].replace('longtable', 'tabular')

    lines = [ line.replace('\\multirow{2}{*}{', '') for line in lines ]
    lines = [ line.replace('\\multirow{6}{*}{', '') for line in lines ]
    lines = [ line.replace('}}', '}') for line in lines ]
    lines = [ line.replace('NA}', 'NA') for line in lines ]
    lines = [ line.replace('IQR: ', '') for line in lines ]

    pad_lines = [ 4, 5, 6, 8, 10, 12, 14 ]
    for i in pad_lines:
        lines[i] = '\\rule{0pt}{2.5ex}'  + lines[i]
    
    table_text = ' '.join(lines)
    
    return table_text
    
def format_table234(fname, setwidth=False):
    
    # Read in the file
    with open(fname, 'r') as f:
        lines = f.readlines()
        lines = [ line.strip() for line in lines ]
    
    # There are a few latex tricks that don't work in matplotlib so we
    # have to tidy them up here
    lines[0] = lines[0].replace('longtable', 'tabular')
    if setwidth:
        lines[0] = lines[0].replace('llll', 'p{2.3cm} p{1.8cm} p{1.8cm} p{1.8cm}')
        lines[0] = lines[0].replace('lccc', 'p{2.3cm} p{1.8cm} p{1.8cm} p{1.8cm}')
    del(lines[1])
    lines[1] = '\\hline'
    lines[3] = '\\hline'
    del(lines[4:12])
    lines[-1] = lines[-1].replace('longtable', 'tabular')

    lines = [ line.replace('\\multirow{3}{*}{', '') for line in lines ]
    lines = [ line.replace('\\multirow{6}{*}{', '') for line in lines ]
    lines = [ line.replace('95\\% CI', '') for line in lines ]
    lines[1:] = [ line.replace('}}', '}', 1) for line in lines[1:] ]

    lines = [ line.replace('\\rule{0pt}{4ex}', '\\rule{0pt}{3ex}') for line in lines ]
        
    table_text = ' '.join(lines)

    return table_text
    
    
def add_parcellation_brains(paper_dir, fig):
    parcellation_brains = os.path.join(paper_dir, 
                                       'COMBINED_FIGURES',
                                       'PARCELLATION',
                                       'PNGS',
                                       'Parcellation_308_random_matched_hemis_FourHorBrains.png')
    
    img = mpimg.imread(parcellation_brains)
    
    # Set up the grid
    grid = gridspec.GridSpec(1,1)
    grid.update(left=0.01, right=0.99, top=0.72, bottom=0.59, wspace=0, hspace=0)
    ax_brains = plt.Subplot(fig, grid[0])
    fig.add_subplot(ax_brains)
    ax_brains.imshow(img, interpolation='none')
    ax_brains.axis('off')
    
    return fig
    
    
def add_figure1_parts(paper_dir, fig):
    import os
    
    # Set up the grid
    grid = gridspec.GridSpec(3,1)
    grid.update(left=0.03, right=0.99, top=0.6, bottom=0, wspace=0, hspace=0.05)
    ax_list = []
    for g_loc in grid:
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])
        
    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        figure1 = os.path.join(paper_dir, cohort, 'FIGS', 'COVARS_none', 'Figure1.png')
        
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(figure1)
        crop_edge = img.shape[1]/100.0 * 34.2
        img_cropped = img[:, crop_edge:,:]
        ax.imshow(img_cropped, interpolation='none')
        ax.axis('off')
        
        # Add in a label for the cohort
        ax.text(-0.05, 0.6, cohort.title(),
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=10)
    
    return fig

    
def add_figure2_parts(paper_dir, fig):
    import os
    
    # Set up the grid
    grid = gridspec.GridSpec(2,2)
    grid.update(left=0.03, right=0.97, top=0.98, bottom=0.2, wspace=0.03, hspace=0.05)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])

    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        figure2 = os.path.join(paper_dir, cohort, 'FIGS', 'COVARS_none', 'Figure2.png')
        
        if i > 0 : i += 1
            
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(figure2)
        ax.imshow(img, interpolation='none')
        ax.axis('off')
        
        # Add in a label for the cohort
        ax.text(-0.05, 0.5, cohort.title(),
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=10)
    
    return fig, ax_list[1]
    
def add_extra_fig2_bits(paper_dir, fig):
    import os

    # Set up the grid
    grid = gridspec.GridSpec(1,3)
    grid.update(left=0.01, right=0.99, top=0.2, bottom=0.03, wspace=0, hspace=0)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])

    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        mediation_figure = os.path.join(paper_dir, cohort, 'FIGS', 'COVARS_none', 'Mediation.png')
        
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(mediation_figure)
        img_cropped = img[50:-50,:,:]
        ax.imshow(img_cropped, interpolation='none')
        ax.axis('off')
        
        # Add in a label for the cohort
        ax.text(0.5, 0.0, cohort.title(),
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes, 
                    fontsize=10)
        
    return fig
    

    
def add_figure3_parts(paper_dir, fig):
    import os
    
    # Set up the grid
    grid = gridspec.GridSpec(3,1)
    grid.update(left=0.03, right=0.55, top=0.8, bottom=0.0, wspace=0, hspace=0)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])

    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        figure3 = os.path.join(paper_dir, cohort, 'FIGS', 'COVARS_none', 'Figure3.png')
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(figure3)
        img_cropped = img[:1200, :, :]
        ax.imshow(img_cropped, interpolation='none')
        ax.axis('off')
        
        # Add in a label for the cohort
        ax.text(-0.05, 0.5, cohort.title(),
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=10)
    
    return fig
    
def add_extra_fig3_bits(paper_dir, fig):
    import os

    # Add in the MBP correlations
    # at the top on the left
    mbp_fig = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_none', 'MBPvsMT14.png')
    
    # Set up the grid
    grid = gridspec.GridSpec(1,1)
    grid.update(left=0.03, right=0.55, top=0.99, bottom=0.8, wspace=0, hspace=0)
    ax = plt.Subplot(fig, grid[0])
    fig.add_subplot(ax)
    
    # Show the figure
    img = mpimg.imread(mbp_fig)
    ax.imshow(img, interpolation='none')
    ax.axis('off')
    
    # Next add the permutation tests
    # at the bottom on the left
    
    # Set up the grid
    grid = gridspec.GridSpec(2,1)
    grid.update(left=0.55, right=0.99, top=0.43, bottom=0, wspace=0, hspace=0)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])
        
    gene_dict = { 'OL' : 'Oligodendrocytes',
                  'SZ' : 'Schizophrenia' }
    
    for i, gene in enumerate(sorted(list(gene_dict.keys()))):
        gene_fig = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_none', 'CandidateGenes_{}.png'.format(gene))
        ax = ax_list[i]

        img = mpimg.imread(gene_fig)
        ax.imshow(img, interpolation='none')
        ax.axis('off')
        
        # Add in a label for the cohort
        ax.text(-0.05, 0.55, gene_dict[gene],
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=8)
        
    return fig
    
def add_figure4_parts(paper_dir, fig):
    import os
    
    # Set up the grid
    grid = gridspec.GridSpec(3,1)
    grid.update(left=0.03, right=0.48, top=0.74, bottom=0.0, wspace=0, hspace=0)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])

    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        figure4 = os.path.join(paper_dir, cohort, 'FIGS', 'COVARS_none', 'Figure4.png')
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(figure4)
        ax.imshow(img, interpolation='none')
        ax.axis('off')
        # Add in a label for the cohort
        ax.text(-0.05, 0.5, cohort.title(),
                    horizontalalignment='left',
                    verticalalignment='center',
                    rotation=90,
                    transform=ax.transAxes,
                    fontsize=10)
    
    return fig
    
def add_extra_fig4_bits(paper_dir, fig):
    import os

    # Set up the grid
    grid = gridspec.GridSpec(1,3)
    grid.update(left=0.01, right=0.99, top=1.0, bottom=0.75, wspace=0, hspace=0)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])

    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        figure4 = os.path.join(paper_dir, cohort, 'FIGS', 'NetworkSummary.png')
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(figure4)
        ax.imshow(img, interpolation='none')
        ax.axis('off')
        
        # Add in a label for the cohort
        ax.text(0.5, 0, cohort.title(),
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize=10)
    return fig
    
    
    
def add_figure5_parts(paper_dir, fig):
    import os
    
    # Set up the grid
    grid = gridspec.GridSpec(1,3)
    grid.update(left=0.01, right=0.99, top=1.0, bottom=0.05, wspace=0, hspace=0)
    ax_list = []
    for i, g_loc in enumerate(grid):
        ax_list += [plt.Subplot(fig, g_loc)]
        fig.add_subplot(ax_list[-1])

    # Loop through the three cohorts
    for i, cohort in enumerate(['DISCOVERY', 'VALIDATION', 'COMPLETE']):
        xyz_fig = os.path.join(paper_dir, cohort, 'FIGS', 'COVARS_none', 'XYZ_vs_Measures.png')
        ax = ax_list[i]
        
        # Show the figure, excluding the first panel
        img = mpimg.imread(xyz_fig)
        ax.imshow(img, interpolation='none')
        ax.axis('off')
        # Add in a label for the cohort
        ax.text(0.5, 0.0, cohort.title(),
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    fontsize=10)
    
    return fig
    

def make_SuppFig1(paper_dir, covars_name='none'):
    
    import matplotlib.gridspec as gridspec
    import matplotlib.image as mpimg
    import matplotlib.pylab as plt
    from matplotlib import rc
    rc('text', usetex=True)
    import numpy as np
    import os
    
    # Get the table text
    demotable_fname = os.path.join(paper_dir, 'STATS_TABLES', 'DemographicsTable.txt')
    table_text = format_demotable(demotable_fname)
    
    # Make a big figure
    # which is 6 inches wide, and 9 inches long
    fig, ax = plt.subplots(figsize=(5.2, 9), facecolor='white')
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    ax.set_zorder(100)
    ax.set_axis_bgcolor('none')

    # Put the table at the top in the middle
    ax.text(0.5, 0.99, table_text, fontsize=7, horizontalalignment='center', verticalalignment='top')

    # Next add in the four brains showing the parcellation scheme
    fig = add_parcellation_brains(paper_dir, fig)
    
    # Add in figure1 for each of the three cohorts
    add_figure1_parts(paper_dir, fig)
    
    # Turn the axes off
    ax.axis('off')
    
    # Add in the letters
    ax.text(0.02, 0.99, 'a', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.02, 0.7, 'b', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.02, 0.59, 'c', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.02, 0.39, 'd', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.02, 0.19, 'e', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    
    # Save the figure
    figures_dir = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_{}'.format(covars_name))
    fname = os.path.join(figures_dir, 'SuppFig1.jpg')
    
    fig.savefig(fname, bbox_inches=0, dpi=300)

def make_SuppFig2(paper_dir, covars_name='none'):
    
    import matplotlib.gridspec as gridspec
    import matplotlib.image as mpimg
    import matplotlib.pylab as plt
    from matplotlib import rc
    rc('text', usetex=True)
    import numpy as np
    import os
    
    # Get the table text
    table2_fname = os.path.join(paper_dir, 'STATS_TABLES', 'COVARS_none', 'Stats_Table_Figure2.txt')
    table_text = format_table234(table2_fname)
    
    # Make a big figure
    # which is 7 inches wide, and 7 inches long
    fig, ax = plt.subplots(figsize=(7.2, 6.5), facecolor='white')
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    ax.set_zorder(100)
    ax.set_axis_bgcolor('none')

    # Add in figure2 for each of the three cohorts
    fig, table_ax = add_figure2_parts(paper_dir, fig)
    
    # Put the table in the top axis
    table_ax.text(0.5, 0.5, table_text, fontsize=6, horizontalalignment='center', verticalalignment='center')
    table_ax.axis('off')
    
    # Next add in the mediation analyses
    fig = add_extra_fig2_bits(paper_dir, fig)
    
    # Turn the axes off
    ax.axis('off')
    
    # Add in the letters
    ax.text(0.01, 0.99, 'a', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.48, 0.99, 'b', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.58, 'c', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.48, 0.58, 'd', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.18, 'e', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')

    # Save the figure
    figures_dir = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_{}'.format(covars_name))
    fname = os.path.join(figures_dir, 'SuppFig2.jpg')
    
    fig.savefig(fname, bbox_inches=0, dpi=300)

    
def make_SuppFig3(paper_dir, covars_name='none'):
    
    import matplotlib.gridspec as gridspec
    import matplotlib.image as mpimg
    import matplotlib.pylab as plt
    from matplotlib import rc
    rc('text', usetex=True)
    import numpy as np
    import os
    
    # Get the table text
    table3_fname = os.path.join(paper_dir, 'STATS_TABLES', 'COVARS_none', 'Stats_Table_Figure3.txt')
    table_text = format_table234(table3_fname)
    
    # Make a big figure
    # which is 7 inches wide, and 7 inches long
    fig, ax = plt.subplots(figsize=(7.2, 6.5), facecolor='white')
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    ax.set_zorder(100)
    ax.set_axis_bgcolor('none')
    
    # Add in figure3 for each of the three cohorts
    fig = add_figure3_parts(paper_dir, fig)
    
    # Put the table in the top right corner
    ax.text(0.97, 0.99, table_text, fontsize=7, horizontalalignment='right', verticalalignment='top')
    
    # Next add in the extra parts for this supplementary figure
    fig = add_extra_fig3_bits(paper_dir, fig)
    
    # Turn the axes off
    ax.axis('off')
    
    # Add in the letters
    ax.text(0.01, 0.99, 'a', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.79, 'b', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.52, 'c', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.24, 'd', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.55, 0.99, 'e', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.55, 0.42, 'f', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.55, 0.2048, 'g', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')

    # Save the figure
    figures_dir = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_{}'.format(covars_name))
    fname = os.path.join(figures_dir, 'SuppFig3.jpg')
    
    fig.savefig(fname, bbox_inches=0, dpi=300)
    

def make_SuppFig4(paper_dir, covars_name='none'):
    
    import matplotlib.gridspec as gridspec
    import matplotlib.image as mpimg
    import matplotlib.pylab as plt
    from matplotlib import rc
    rc('text', usetex=True)
    import numpy as np
    import os
    
    # Get the table text
    table4_fname = os.path.join(paper_dir, 'STATS_TABLES', 'COVARS_none', 'Stats_Table_Figure4.txt')
    table_text = format_table234(table4_fname, setwidth=True)
    network_table_fname = os.path.join(paper_dir, 'STATS_TABLES', 'NETWORK', 'Stats_Table_NetworkMeasures.txt')
    network_table_text = format_table234(network_table_fname, setwidth=True)
    
    # Make a big figure
    # which is 7 inches wide, and 9 inches long
    fig, ax = plt.subplots(figsize=(7.2, 9.7), facecolor='white')
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    ax.set_zorder(100)
    ax.set_axis_bgcolor('none')

    # Add in figure4 for each of the three cohorts
    fig = add_figure4_parts(paper_dir, fig)
    
    # And put the network table on the right just below the 
    # network figures
    ax.text(0.74, 0.74, network_table_text, fontsize=6, horizontalalignment='center', verticalalignment='top')
    
    # Put the table 4 in the bottom right corner
    ax.text(0.74, 0.49, table_text, fontsize=6, horizontalalignment='center', verticalalignment='top')
    
    # Next add in the four brains showing the parcellation scheme
    add_extra_fig4_bits(paper_dir, fig)
    
    # Turn the axes off
    ax.axis('off')
        
    # Add in the letters
    ax.text(0.01, 0.995, 'a', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.34, 0.995, 'b', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.67, 0.995, 'c', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.465, 0.74, 'd', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.74, 'e', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.49, 'f', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.01, 0.24, 'g', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.465, 0.49, 'h', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')

    # Save the figure
    figures_dir = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_{}'.format(covars_name))
    fname = os.path.join(figures_dir, 'SuppFig4.jpg')
    
    fig.savefig(fname, bbox_inches=0, dpi=300)
    
def make_SuppFig5(paper_dir, covars_name='none'):
    
    import matplotlib.gridspec as gridspec
    import matplotlib.image as mpimg
    import matplotlib.pylab as plt
    from matplotlib import rc
    rc('text', usetex=True)
    import numpy as np
    import os
    
    # Make a big figure
    # which is 7 inches wide, and 9 inches long
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor='white')
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.99)
    ax.set_zorder(100)
    ax.set_axis_bgcolor('none')
    
    # Add in figure4 for each of the three cohorts
    fig = add_figure5_parts(paper_dir, fig)
    
    # Turn the axes off
    ax.axis('off')
    
    # Add in the letters
    ax.text(0.01, 0.99, 'a', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.3433, 0.99, 'b', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    ax.text(0.6766, 0.99, 'c', fontsize=9, weight='bold', horizontalalignment='center', verticalalignment='center')
    
    # Save the figure
    figures_dir = os.path.join(paper_dir, 'COMBINED_FIGURES', 'COVARS_{}'.format(covars_name))
    fname = os.path.join(figures_dir, 'SuppFig5.jpg')
    
    fig.savefig(fname, bbox_inches=0, dpi=300)
    
        
def make_supplemental_figures(covars_name='none'):
    
    # Define some locations
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(scripts_dir, '..', 'CT_MT_ANALYSES')
    
    # Print to screen what you're up to!
    print 'Making supplemental figures'
    
    # And go!
    make_SuppFig1(paper_dir, covars_name=covars_name)
    make_SuppFig2(paper_dir, covars_name=covars_name)
    make_SuppFig3(paper_dir, covars_name=covars_name)
    make_SuppFig4(paper_dir, covars_name=covars_name)
    make_SuppFig5(paper_dir, covars_name=covars_name)

make_supplemental_figures(covars_name='none')