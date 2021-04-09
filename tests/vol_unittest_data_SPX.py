import unittest
import pickle
import volvisualizer.volatility as vol
unittest.TestLoader.sortTestMethodsUsing = None

class VolatilityDataSPXTestCase(unittest.TestCase):
   
    def test_create_data_spx(self):
        spx = vol.Volatility()
        spx.create_option_data(ticker='^SPX', start_date='2021-04-08', wait=0.5, monthlies=True, q=0.013)
        spx_pickle = open('spx_data', 'wb')
        pickle.dump(spx, spx_pickle)
        spx_pickle.close()        


if __name__ == '__main__':
    unittest.main()