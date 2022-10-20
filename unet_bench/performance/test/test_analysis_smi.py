import unittest
import os
import sys
import tempfile
import shutil

account = 'bwilson'

input_account = input(f"Enter account name [{account}]: ")
if len(input_account) > 0:
    account = input_account

#sys.path.append(f'/home/{account}/DL/Stocks/ProcessThat/code')
#sys.path.append(f'/home/{account}/DL/Stocks/finnhub-data/code')
sys.path.append(f'/home/{account}/DL/Stocks/analyzethis/code')

from analysis_smi import clean_curve

class TestAnalysisSmi(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir  = tempfile.mkdtemp()

    def tearDown(self):

        # Remove the temp directory after the test
        shutil.rmtree(self.test_dir)


    def test_clean_curve_01(self):
        """
python test/test_analysis_smi.py  TestAnalysisSmi.test_clean_curve_01
        """


        pass




"""
Can use <F5> or <Ctrl+F5> by doing the following:
source ~/venvpower/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/bwilson/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance

If ran from performance directory, you can:
python -m unittest discover -s test

"""

if __name__ == '__main__':
    unittest.main()
