#!/usr/bin/env python

def read_fsaverage_info(fsaverage_dir):
    '''
    Get all the info you need from the study fsaverage directory
    which contains the names of the 308 regions, the location of each
    region's center of mass, the assignment of each region to the 
    appropriate von economo cortical type and and the assignment of
    each region to a lobe label.
    '''
    import numpy as np
    import os

    #=============================================================================
    # Create an empty dictionary for these values
    #=============================================================================
    fsaverage_dict = {}
    
    #=============================================================================
    # Get the names of the files you need
    #=============================================================================
    aparc_names_file = os.path.join(fsaverage_dir, 'parcellation', '500.names.txt' )
    centroids_file = os.path.join(fsaverage_dir, 'parcellation', '500.centroids.txt' )
    von_economo_file = os.path.join(fsaverage_dir, 'parcellation', '500.vonEconomoRegions.txt' )
    lobes_file = os.path.join(fsaverage_dir, 'parcellation', '500.lobes.txt' )

    #=============================================================================
    # Create some useful data lists
    #=============================================================================
    fsaverage_dict['aparc_names'] = get_labels(aparc_names_file)
    
    fsaverage_dict['von_economo'] = get_labels(von_economo_file, convert_to_float=True)

    fsaverage_dict['lobes'] = get_labels(lobes_file)
    
    (fsaverage_dict['centroids'], 
        fsaverage_dict['axial_pos'], 
        fsaverage_dict['sagittal_pos'], 
        fsaverage_dict['coronal_pos'] ) = get_centroids(centroids_file)

    return fsaverage_dict
    

def get_labels(filename, convert_to_float=False):
    '''
    Load each line in the filename, where each line represents one of
    the cortical and subcortical regions in the 308 parcellation, remove 
    the first 41 entries which correspond to the subcortical regions 
    and return the names as a list.
    
    If convert_to_float=True then convert the values in the list to floats
    otherwise leave as strings.
    '''
    labels = [ line.strip() for line in open(filename) ]
    labels = labels[41::]
    
    if convert_to_float:
        labels = [ float(label) for label in labels ]
        
    return labels

def get_centroids(filename):
    '''
    Load the x, y, z coordinates of the cortical and subcortical parcellation
    regions, remove the first 41 entries which correspond to the subcortical
    regions, and return the values as a numpy array, and the axial, sagittal,
    and coronal positions as dictionaries
    '''
    import numpy as np
    
    centroids = np.loadtxt(filename) 
    centroids = centroids[41:,:]
    
    # Use the centroids to create a dictionary for each of the
    # three orthogonal directions
    axial_pos = {key: value for (key, value) in zip(range(len(centroids)),centroids[:,:2])}
    sagittal_pos = {key: value for (key, value) in zip(range(len(centroids)),centroids[:,1:])}
    coronal_pos = {key: value for (key, value) in zip(range(len(centroids)),centroids[:,0::2])}
    
    return centroids, axial_pos, sagittal_pos, coronal_pos

    
