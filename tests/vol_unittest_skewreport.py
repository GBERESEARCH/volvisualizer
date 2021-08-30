import unittest
import pickle
from volvisualizer.volatility import Volatility
unittest.TestLoader.sortTestMethodsUsing = None

class SkewDataTestCase(unittest.TestCase):

    def test_impliedvol(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        print(spx.vol(maturity='2022-03-31', strike=80, smoothing=True))
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        print(gld.vol(maturity='2021-10-30', strike=120, smoothing=True))
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        print(tsla.vol(maturity='2021-11-30', strike=70))
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        print(aapl.vol(maturity='2021-12-31', strike=100, smoothing=True))


    def test_skewreport(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.skewreport(months=15, direction='full')
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.skewreport(months=9, direction='up')
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.skewreport(months=6)
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.skewreport(direction='down')

if __name__ == '__main__':
    unittest.main()
