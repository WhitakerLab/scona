import unittest

class FixturesTest(unittest.TestCase):

	def setUp(self):
		print('in setUp()')
		import os
		import sys
		import filecmp
		sys.path.extend(['./WRAPPERS','./exemplary_brains','./SCRIPTS/'])
		self.corrmat_path = os.getcwd()+'/corrmat_file_testing.txt'
		from corrmat_from_regionalmeasures import corrmat_from_regionalmeasures	
		corrmat_from_regionalmeasures("./exemplary_brains/PARC_500aparc_thickness_behavmerge.csv", "./exemplary_brains/500.names.txt", self.corrmat_path, names_308_style=True)

	def tearDown(self):
		import os
		print('in tearDown()')
		os.remove(self.corrmat_path)

	def test(self):
		print('in test()')
		self.assertEqual(1,1)

if __name__ == '__main__':
	unittest.main()
