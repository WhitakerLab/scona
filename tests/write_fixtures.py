#--------------------------- Write fixtures ---------------------------
# To regression test our wrappers we need examples. This script 
# generates files. We save these files once, and regression-tests.py
# re-generates these files to tests them for identicality with the
# presaved examples (fixtures). If they are found not to be identical 
# it throws up an error. 
#
# (*someday* I would like to replace saving files with saving hashes)
#
# The point of this is to check that throughout the changes we make to 
# BrainNetworksInPython the functionality of this script stays the same
#
# Currently the functionality of write_fixtures is to generate corrmat 
# and network_analysis data via the functions 
# corrmat_from_regionalmeasures and network_analysis_from_corrmat.
#----------------------------------------------------------------------
                       
def recreate_correlation_matrix_fixture(folder):
    ##### generate a correlation matrix in the given folder using #####
    ##### the data in example_data                                ##### 
    import os
    corrmat_path = os.getcwd()+folder+'/corrmat_file.txt'
    from corrmat_from_regionalmeasures import corrmat_from_regionalmeasures
    corrmat_from_regionalmeasures(
        "example_data/PARC_500aparc_thickness_behavmerge.csv",
        "example_data/500.names.txt", 
        corrmat_path,
        names_308_style=True)
     
def recreate_network_analysis_fixture(folder, corrmat_path):
    ##### generate network analysis in the given folder using the #####
    ##### data in example_data and the correlation matrix given   #####
    ##### by corrmat_path                                         #####  
    import os
    # It is necessary to specify a random seed because 
    # network_analysis_from_corrmat generates random graphs to 
    # calculate global measures
    import random
    random.seed(2984)
    from network_analysis_from_corrmat import network_analysis_from_corrmat
    network_analysis_from_corrmat(corrmat_path,
                              "example_data/500.names.txt",
                              "example_data/500.centroids.txt",
                              os.getcwd()+folder+'/network-analysis',
                              cost=10,
                              n_rand=10, # this is not a reasonable 
                              # value for n, we generate only 10 random
                              # graphs to save time
                              names_308_style=True)
    
def write_fixtures(folder='/tmp'): 
    ## Run functions corrmat_from_regionalmeasures and               ##
    ## network_analysis_from_corrmat to save corrmat in given folder ##
    ##---------------------------------------------------------------##
    #
    # We want to prevent accidentally overwriting fixtures, so if the 
    # function is saving in the test_fixtures folder it prompts for 
    # approval.
    if folder == '/test_fixtures':
        if input('Are you sure you want to overwrite existing fixtures? y/n')!='y':
            if input('Would you like to save new fixtures in a different folder?')!='y':
                return
            else:
                folder = input('what would you like to name this folder')
    # add wrappers, example_data and scripts folders to the syspath
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join('wrappers')))
    sys.path.append(os.path.abspath(os.path.join('example_data')))
    sys.path.append(os.path.abspath(os.path.join('scripts')))
    # if the folder does not exist, create it
    if not os.path.isdir(os.getcwd()+folder):
        os.makedirs(os.getcwd()+folder)
    # generate and save the correlation matrix
    print("generating new correlation matrix") 
    recreate_correlation_matrix_fixture(folder)
    # generate and save the network analysis
    print("generating new network analysis") 
    corrmat_path = 'tests/test_fixtures/corrmat_file.txt'
    recreate_network_analysis_fixture(folder, corrmat_path)

if __name__ == '__main__':
    write_fixtures()
