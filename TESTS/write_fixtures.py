# To regression test our wrappers we need examples. This script generates some example files, and re-generates them for testing. We can then check that the files are identical
                       
def Recreate_correlation_matrix_fixture(folder):
    import os
    ##### generate the correlation matrix using the data in EXAMPLE_DATA #####  
    corrmat_path = os.getcwd()+folder+'/corrmat_file.txt'
    from corrmat_from_regionalmeasures import corrmat_from_regionalmeasures
    corrmat_from_regionalmeasures("EXAMPLE_DATA/PARC_500aparc_thickness_behavmerge.csv", "EXAMPLE_DATA/500.names.txt", corrmat_path, names_308_style=True)
     
def Recreate_network_analysis_fixture(folder, corrmat_path):
    ##### generate the network analysis using the data in EXAMPLE_DATA  and the correlation matrix given by corrmat_path #####  
    import os
    import random
    # We need to specify a random seed to avoid complications with our randomly generated graphs
    random.seed(2984)
    from network_analysis_from_corrmat import network_analysis_from_corrmat
    network_analysis_from_corrmat(corrmat_path,
                              "EXAMPLE_DATA/500.names.txt",
                              "EXAMPLE_DATA/500.centroids.txt",
                              os.getcwd()+folder+'/network-analysis',
                              cost=10,
                              n_rand=10,# this is not a reasonable n, but it saves us a lot of time
                              names_308_style=True)
    
def write_fixtures(folder, corrmat=True, net_analysis=True): ## corrmat and net_analysis arguments allow you to exclude them 
    if (corrmat, net_analysis) == (None,None):
        return
    import os
    import sys
    if '/' not in folder:
        folder = '/'+folder
    sys.path.append(os.path.abspath(os.path.join('WRAPPERS')))
    sys.path.append(os.path.abspath(os.path.join('EXAMPLE_DATA')))
    sys.path.append(os.path.abspath(os.path.join('SCRIPTS')))
    print(sys.path)
    if not os.path.isdir(os.getcwd()+folder):
        os.makedirs(os.getcwd()+folder)
       
    if corrmat == True:
        print("generating new correlation matrix") 
        Recreate_correlation_matrix_fixture(folder)
    
    if net_analysis==True:
        print("generating new network analysis") 
        corrmat_path = 'TESTS/test_fixtures/corrmat_file.txt'
        Recreate_network_analysis_fixture(folder, corrmat_path)
        

def overwrite_fixtures(corrmat=True, net_analysis=True): 
# This function replaces the gold standard fixtures, use it wisely
    if input('Are you sure you want to overwrite existing fixtures?  y/n  ') == 'y':
        write_fixtures('test_fixtures',corrmat=corrmat,net_analysis=net_analysis)
    elif input('Would you like to save new fixtures in a different folder?  ') == 'y':
        folder = input('what would you like to name this folder')
        write_fixtures(folder, corrmat=corrmat,net_analysis=net_analysis)

if __name__ == '__main__':
    overwrite_fixtures()
