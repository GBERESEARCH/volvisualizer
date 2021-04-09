import unittest
import pickle
import volvisualizer.volatility as vol
unittest.TestLoader.sortTestMethodsUsing = None


class VolatilityDataGLDTestCase(unittest.TestCase):

    def test_create_data_gld(self):
        gld = vol.Volatility()
        gld.create_option_data(ticker='GLD', start_date='2021-04-08', wait=0.5, monthlies=True)
        gld_pickle = open('gld_data', 'wb')
        pickle.dump(gld, gld_pickle)
        gld_pickle.close()        
        

if __name__ == '__main__':
    unittest.main()

