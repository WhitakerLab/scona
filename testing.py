import unittest

class Test_corrmat_from_regional_measures(unittest.TestCase):

	def setUp(self):		
		print('in setup')
		import os
		import sys	
		import filecmp
		sys.path.extend(['./WRAPPERS','./exemplary_brains','./SCRIPTS/'])   
		from corrmat_from_regionalmeasures import corrmat_from_regionalmeasures	    
		corrmat_from_regionalmeasures("./exemplary_brains/PARC_500aparc_thickness_behavmerge.csv", "./exemplary_brains/500.names.txt", os.getcwd()+'/corrmat_file_testing.txt', names_308_style=True)
		self.corrmat_path = os.getcwd()+'/corrmat_file_testing.txt'

	def tearDown(self):
		print('in teardown')
		os.remove(self.corrmat_path) 

	def regression_test_corrmat_from_regional_measures(self):
		self.assertTrue(filecmp.cmp(self.corrmat_path, 'golden_corrmat_file.txt'))



