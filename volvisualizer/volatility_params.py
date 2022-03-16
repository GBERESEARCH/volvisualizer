"""
Key parameters for visualizer

"""

# Dictionary of default parameters
vol_params_dict = {
    'ticker':'^SPX',
    'ticker_label':None,
    'start_date':None,
    'wait':5,
    'minopts':None,
    'mindays':None,
    'lastmins':None,
    'volume':None,
    'openint':None,
    'graphtype':'line',
    'surfacetype':'mesh',
    'smoothing':False,
    'scatter':True,
    'voltype':'last',
    'smoothopt':6,
    'notebook':False,
    'r':0.005,
    'q':0,
    'epsilon':0.001,
    'method':'nr',
    'order':3,
    'spacegrain':100,
    'azim':-50,
    'elev':20,
    'fig_size':(15, 12),
    'rbffunc':'thin_plate',
    'colorscale':'BlueRed',
    'monthlies':True,
    'divisor':None,
    'divisor_SPX':25,
    'spot':None,
    'strike_limits':(0.5, 2.0),
    'put_strikes':None,
    'call_strikes':None,
    'opacity':1,
    'surf':True,
    'save_image':False,
    'image_folder':'images',
    'image_filename':'impvol',
    'image_dpi':50,
    'skew_months':12,
    'skew_direction':'downside',

    # Dictionary of implied vol fields used in graph methods
    'vols_dict':{
        'bid':'Imp Vol - Bid',
        'mid':'Imp Vol - Mid',
        'ask':'Imp Vol - Ask',
        'last':'Imp Vol - Last'
        },

    # Dictionary of price fields used for filtering zero prices in
    # graph methods
    'prices_dict':{
        'bid':'Bid',
        'mid':'Mid',
        'ask':'Ask',
        'last':'Last Price'
        },

    # Dictionary of implied vol fields used in implied vol calculation
    'row_dict':{
        'Bid':'Imp Vol - Bid',
        'Mid':'Imp Vol - Mid',
        'Ask':'Imp Vol - Ask',
        'Last Price':'Imp Vol - Last'
        },

    # Dictionary of interpolation methods used in implied vol calculation
    'method_dict':{
        'nr':'implied_vol_newton_raphson',
        'bisection':'implied_vol_bisection',
        'naive':'implied_vol_naive'
        },

    # Dictionary mapping tenor buckets to number of days
    'ir_tenor_dict':{
        '1 Mo':30,
        '2 Mo':60,
        '3 Mo':90,
        '6 Mo':180,
        '1 Yr':365,
        '2 Yr':730,
        '3 Yr':1095,
        '5 Yr':1826,
        '7 Yr':2556,
        '10 Yr':3652,
        '20 Yr':7305,
        '30 Yr':10952
        },

    # Parameters to overwrite mpl_style defaults
    'mpl_line_params':{
        'axes.edgecolor':'black',
        'axes.titlepad':20,
        'axes.xmargin':0.05,
        'axes.ymargin':0.05,
        'axes.linewidth':2,
        'axes.facecolor':(0.8, 0.8, 0.9, 0.5),
        'xtick.major.pad':10,
        'ytick.major.pad':10,
        'lines.linewidth':3.0,
        'grid.color':'black',
        'grid.linestyle':':'
        },

    'mpl_3D_params':{
        'axes.facecolor':'w',
        'axes.labelcolor':'k',
        'axes.edgecolor':'w',
        'lines.linewidth':0.5,
        'xtick.labelbottom':True,
        'ytick.labelleft':True
        },

    }
