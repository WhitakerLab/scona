# To regression test our wrappers we need examples. This script generates our example files, and re-generates them for testing. We can then check that the files are identical

def overwrite_fixtures(corrmat=True, net_analysis=True): 
# This function replaces the gold standard fixtures, use it wisely
    if input('Are you sure you want to overwrite existing fixtures?  y/n  ') == 'y':
        write_fixtures('test_fixtures',corrmat=corrmat,net_analysis=net_analysis)
    elif input('Would you like to save new fixtures in a different folder?  '): == 'y':
        folder = input('what would you like to name this folder)
        write_fixtures(folder, corrmat=corrmat,net_analysis=net_analysis)
  
def write_fixtures(folder, corrmat=True, net_analysis=True):
    if (corrmat, net_analysis) == (None,None):
        break
    import os
    import sys
    if '/' not in folder:
        folder = '/'+folder
	sys.path.extend(['./WRAPPERS','./exemplary_brains','./SCRIPTS/'])
    if not os.path.isdir(os.getcwd()+folder):
        os.makedirs(os.getcwd()+folder)
                       
    ##### Recreate the correlation matrix #####                   
    if corrmat = True:
        corrmat_path = os.getcwd()+folder+'/corrmat_file.txt'
        from corrmat_from_regionalmeasures import corrmat_from_regionalmeasures	
        corrmat_from_regionalmeasures("./exemplary_brains/PARC_500aparc_thickness_behavmerge.csv", "./exemplary_brains/500.names.txt", corrmat_path, names_308_style=True)
                       
    ##### Recreate the network analysis #####   
    if net_analysis=True:
		import random
		random.seed(2984)
        from network_analysis_from_corrmat import network_analysis_from_corrmat
        network_analysis_from_corrmat('/test_fixtures/corrmat_file.txt',
                                  names_file,
                                  centroids_file,
                                  os.getcwd()+folder+'/network-analysis',
                                  cost=10,
                                  n_rand=100,
                                  names_308_style=True)


if __name__ == '__main__':
    overwrite_fixtures()
