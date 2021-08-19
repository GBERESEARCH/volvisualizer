import unittest
import pickle
from volvisualizer.volatility import Volatility
unittest.TestLoader.sortTestMethodsUsing = None

class VolatilityDataSPXTestCase(unittest.TestCase):

    def test_create_data_spx(self):
        spx = Volatility(ticker='^SPX', wait=0.5, monthlies=True, q=0.013)
        spx_pickle = open('spx_data', 'wb')
        pickle.dump(spx, spx_pickle)
        spx_pickle.close()

if __name__ == '__main__':
    unittest.main()
