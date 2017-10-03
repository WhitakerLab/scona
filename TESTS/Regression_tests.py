import unittest
import filecmp
import os

class FixturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('\nin set up - this takes about 80 secs')
        from write_fixtures import write_fixtures
        # We will be checking that the new files generated by 'write_fixtures' match the old ones
        write_fixtures('/tmp')

        cls.corrmat = '/corrmat_file.txt'
        cls.global_measures = '/network-analysis/GlobalMeasures_corrmat_file_COST010.csv'
        cls.local_measures = '/network-analysis/NodalMeasures_corrmat_file_COST010.csv'
        cls.rich_club = '/network-analysis/RICH_CLUB_corrmat_file_COST010.csv'
    
    @classmethod
    def tearDownClass(cls):
        import shutil
        print('\nin tear down ')
        shutil.rmtree(os.getcwd()+'/tmp')

    def test_corrmat_matches_fixture(self):
        print('\ntesting that corrmat matches fixture')
        self.assertTrue(filecmp.cmp(os.getcwd()+'/tmp'+self.corrmat, os.getcwd()+'/TESTS/test_fixtures'+self.corrmat))

    def test_global_measures_against_fixture(self):
        print('\ntesting that global_measures matches fixture')
        self.assertTrue(filecmp.cmp(os.getcwd()+'/tmp'+self.global_measures,os.getcwd()+'/TESTS/test_fixtures'+self.global_measures))
    
    def test_local_measures_against_fixture(self):
        print('\ntesting that local_measures matches fixture')
        self.assertTrue(filecmp.cmp(os.getcwd()+'/tmp'+self.local_measures,os.getcwd()+'/TESTS/test_fixtures'+self.local_measures))
    
    def test_rich_club_against_fixture(self):
        print('\ntesting that rich_club matches fixture')
        self.assertTrue(filecmp.cmp(os.getcwd()+'/tmp'+self.rich_club,os.getcwd()+'/TESTS/test_fixtures'+self.rich_club))  
    

if __name__ == '__main__':
    unittest.main()
