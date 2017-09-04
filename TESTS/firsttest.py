import unittest

class RegressionTest(unittest.TestCase):

	def setUp(self):
		print('in setUp()')
		from write_fixtures import write_fixtures
        write_fixtures('/tmp')
		import filecmp  
        self.corrmat = '/corrmat_file.txt'
        self.global_meas = '/network_analysis/GlobalMeasures_corrmat_file_COST010.csv'
        self.local_meas = '/network_analysis/NodalMeasures_corrmat_file_COST010.csv'
        self.rich_club = '/network_analysis/RICH_CLUB_corrmat_file_COST010.csv'

	def tearDown(self):
		import os
		print('in tearDown()')
		shutil.rmtree(os.getcwd()+'/tmp')

	def corrmat_from_regional_measures_test(self):
		print('regression testing make_corrmat_from_regional_measures')
		self.assertEqual(filecmp.cmp(os.getcwd()+'/tmp'+self.corrmat, os.getcwd()+'/test_fixtures'+self.corrmat))
        
    	def network_analysis_from_corrmat_test(self)
        	print('regression testing network_analysis_from_corrmat_test')
        	self.assertEqual(filecmp.cmp(os.getcwd()+'/tmp'+self.global_meas,os.getcwd()+'/test_fixtures'+self.global_meas)
       	 	self.assertEqual(filecmp.cmp(os.getcwd()+'/tmp'+self.local_meas,os.getcwd()+'/test_fixtures'+self.local_meas)
        	self.assertEqual(filecmp.cmp(os.getcwd()+'/tmp'+self.rich_club,os.getcwd()+'/test_fixtures'+self.rich_club)                         
                         
if __name__ == '__main__':
	unittest.main()
