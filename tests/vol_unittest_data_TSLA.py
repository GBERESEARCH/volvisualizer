import unittest
import pickle
from volvisualizer.volatility import Volatility
unittest.TestLoader.sortTestMethodsUsing = None

class VolatilityDataTSLATestCase(unittest.TestCase):

    def test_create_data_tsla(self):
        tsla = Volatility(ticker='TSLA', wait=0.5, monthlies=True)
        tsla_pickle = open('tsla_data', 'wb')
        pickle.dump(tsla, tsla_pickle)
        tsla_pickle.close()

if __name__ == '__main__':
    unittest.main()
