import unittest
import os
import sys
import tempfile
import shutil

account = 'bwilson'

#sys.path.append(f'/home/{account}/DL/Stocks/ProcessThat/code')
#sys.path.append(f'/home/{account}/DL/Stocks/finnhub-data/code')
sys.path.append(f'/home/{account}/DL/Stocks/analyzethis/code')

from Euclid import Euclid

class TestEuclid(unittest.TestCase):

    def setUp(self):
        self.test_Euclid = Euclid()

        # Create a temporary directory
        self.test_dir  = tempfile.mkdtemp()

    def tearDown(self):

        # Remove the temp directory after the test
        shutil.rmtree(self.test_dir)

    def test_getCacheDict(self):
        numStocks = 79
        cpuCount  = 3
        cacheDicts = self.test_Euclid.getCacheDict(numStocks, cpuCount)

        cachDict = cacheDicts[-1]
        lastStop = cachDict.get('stop')
        self.assertEqual(78, lastStop)

    def test_defaultNextTradingDay(self):
        defaultNextTradingDay = self.test_Euclid.defaultNextTradingDay()
        self.assertEqual(0, defaultNextTradingDay)


    def test_bluePredictions(self):
        """
python test/test_ut.py  TestEuclid.test_bluePredictions
        """


        dateStr = '2021-02-27'
        forDateStr = '2021-02-28'
        dictStocksToBuy = {'AAPL': 105.0, 'MSFT': 16.57, 'NVDA': 19.53, 'INTC': 32.71, 'ADBE': 10.92, 'CSCO': 56.94}
        predictionFile  = f'predictions-{dateStr}-for-{forDateStr}.csv'

        self.test_Euclid.bluePredictions(dictStocksToBuy, predictionFile, self.test_dir)

        dirPath = self.test_dir
        filePath = f'blue_predictions_{forDateStr}.txt'

        fullFilePath = os.path.join(dirPath, filePath)

        with open(fullFilePath, 'r') as fileIn:
            stocksToBuyList = []
            for line in fileIn:
                stocksToBuyList.append(line.strip())
            #print(stocksToBuyList)

        count = 0
        for key in dictStocksToBuy.keys():
            self.assertEqual(key, stocksToBuyList[count])
            count += 1





"""
Can use <F5> or <Ctrl+F5> by doing the following:
dlvenv
export PYTHONPATH=$PYTHONPATH:/home/bwilson/DL/Stocks/finnhub-data/code
export PYTHONPATH=$PYTHONPATH:/home/bwilson/DL/Stocks/ProcessThat/code
export PYTHONPATH=$PYTHONPATH:/home/bwilson/DL/Stocks/analyzethis/code
export PYTHONPATH=$PYTHONPATH:/home/bwilson/DL/alpaca/code

If ran from analyzethis directory, you can:=
python -m unittest discover -s test

"""

if __name__ == '__main__':
    unittest.main()
