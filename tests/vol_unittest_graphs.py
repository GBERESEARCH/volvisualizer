import unittest
import pickle
import volvisualizer.volatility as vol
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
        
    
    def test_line_graph(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.line_graph()
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.line_graph()
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.line_graph()
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.line_graph()
        
        
    def test_scatter_3D(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.scatter_3D()
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.scatter_3D()
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.scatter_3D()
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.scatter_3D()
        
        
    def test_surface_3D(self):
        spx_pickle = open('spx_data', 'rb')
        spx = pickle.load(spx_pickle)
        spx_pickle.close()
        spx.surface_3D(scatter=True, smoothing=True, surfacetype='spline')
        gld_pickle = open('gld_data', 'rb')
        gld = pickle.load(gld_pickle)
        gld_pickle.close()
        gld.surface_3D(scatter=True, smoothing=True, surfacetype='spline')
        tsla_pickle = open('tsla_data', 'rb')
        tsla = pickle.load(tsla_pickle)
        tsla_pickle.close()
        tsla.surface_3D(scatter=True, smoothing=True, surfacetype='spline')
        aapl_pickle = open('aapl_data', 'rb')
        aapl = pickle.load(aapl_pickle)
        aapl_pickle.close()
        aapl.surface_3D(scatter=True, smoothing=True, surfacetype='spline')


if __name__ == '__main__':
    unittest.main()
