import unittest
import pickle
import volvisualizer.volatility as vol
unittest.TestLoader.sortTestMethodsUsing = None


class VolatilityDataAAPLTestCase(unittest.TestCase):

    def test_create_data_aapl(self):
        aapl = vol.Volatility()
        aapl.create_option_data(ticker='AAPL', start_date='2021-04-08', wait=0.5, monthlies=True, divisor=5)
        aapl_pickle = open('aapl_data', 'wb')
        pickle.dump(aapl, aapl_pickle)
        aapl_pickle.close()        
        

if __name__ == '__main__':
    unittest.main()

