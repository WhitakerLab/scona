#!/usr/bin/env python

#=============================================================================
# Created by Kirstie Whitaker
# December 2014
# Contact: kw401@cam.ac.uk
#
# Designed to go hand in hand pysurfer_plot_surface_values.py
# Expects a sorted list of png images and annotation text
#=============================================================================

import matplotlib.animation as animation
import numpy as np
from pylab import *
import matplotlib.image as mpimg
from glob import glob
import sys
import os

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='KW'), bitrate=1800)

png_list_fname = sys.argv[1] 
annot_list_fname = sys.argv[2]

#png_list_fname = 'png_list'
#annot_list_fname = 'png_list'

with open(png_list_fname) as f:
    f_list = f.readlines()
    f_list = [ x.strip() for x in f_list ]
    
f_list = [ f_list[0] ]*3 + f_list + [ f_list[-1] ]*3

with open(annot_list_fname) as f:
    annot_list = f.readlines()
    annot_list = [ x.strip() for x in annot_list ]

annot_list = [ annot_list[0] ]*3 + annot_list + [ annot_list[-1] ]*3

figsize = (15,12)
fig, ax = plt.subplots(figsize = figsize, facecolor='white')

png = mpimg.imread(f_list[0])
img = ax.imshow(png, interpolation='none')
txt = ax.text(0.5, 0.5, annot_list[0], 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize = 20)
    
def animate(i):
    png = mpimg.imread(f_list[i])
    img = ax.imshow(png, interpolation='none')
    txt.set_text(annot_list[i])
    return txt

ax.set_axis_off()

ani = animation.FuncAnimation(fig, animate, 
                                    np.arange(1, len(f_list)),
                                    interval=1000, 
                                    repeat=False)

ani.save(os.path.splitext(png_list_fname)[0] + '.mp4')
plt.close()
