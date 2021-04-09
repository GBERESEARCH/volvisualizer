import unittest
import pickle
import volvisualizer.volatility as vol
unittest.TestLoader.sortTestMethodsUsing = None


class VolatilityDataTSLATestCase(unittest.TestCase):        
        
    def test_create_data_tsla(self):
        tsla = vol.Volatility()
        tsla.create_option_data(ticker='TSLA', start_date='2021-04-08', wait=0.5, monthlies=True)
        tsla_pickle = open('tsla_data', 'wb')
        pickle.dump(tsla, tsla_pickle)
        tsla_pickle.close()                


if __name__ == '__main__':
    unittest.main()
