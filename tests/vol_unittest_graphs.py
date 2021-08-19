import unittest
import pickle
from volvisualizer.volatility import Volatility
unittest.TestLoader.sortTestMethodsUsing = None

class VolatilityGraphsTestCase(unittest.TestCase):

    def test_visualize(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.visualize(graphtype='surface')
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.visualize(graphtype='surface')
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.visualize(graphtype='surface')
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.visualize(graphtype='surface')


    def test_linegraph(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.linegraph()
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.linegraph()
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.linegraph()
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.linegraph()


    def test_scatter(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.scatter()
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.scatter()
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.scatter()
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.scatter()


    def test_surface(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.surface(scatter=True, smoothing=True, surfacetype='spline')
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.surface(scatter=True, smoothing=True, surfacetype='spline')
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.surface(scatter=True, smoothing=True, surfacetype='spline')
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.surface(scatter=True, smoothing=True, surfacetype='spline')

if __name__ == '__main__':
    unittest.main()
