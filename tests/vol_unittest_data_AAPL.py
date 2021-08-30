import unittest
import pickle
from volvisualizer.volatility import Volatility
unittest.TestLoader.sortTestMethodsUsing = None


class VolatilityDataAAPLTestCase(unittest.TestCase):

    def test_create_data_aapl(self):
        aapl = Volatility(ticker='AAPL', wait=0.5, monthlies=True)
        aapl_pickle = open('aapl_data', 'wb')
        pickle.dump(aapl, aapl_pickle)
        aapl_pickle.close()

if __name__ == '__main__':
    unittest.main()
