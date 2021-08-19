import unittest
import pickle
from volvisualizer.volatility import Volatility
unittest.TestLoader.sortTestMethodsUsing = None

class VolatilityDataGLDTestCase(unittest.TestCase):

    def test_create_data_gld(self):
        gld = Volatility(ticker='GLD', wait=0.5, monthlies=True)
        gld_pickle = open('gld_data', 'wb')
        pickle.dump(gld, gld_pickle)
        gld_pickle.close()

if __name__ == '__main__':
    unittest.main()
