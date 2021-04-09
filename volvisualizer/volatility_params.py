# Dictionary of default parameters
vol_params_dict = {
    'df_ticker':'^SPX',
    'df_wait':5,
    'df_graphtype':'line', 
    'df_surfacetype':'mesh', 
    'df_smoothing':False, 
    'df_scatter':False, 
    'df_voltype':'last',
    'df_smoothopt':6,
    'df_notebook':False,
    'df_r':0.005, 
    'df_q':0, 
    'df_epsilon':0.001, 
    'df_method':'nr',
    'df_order':3,
    'df_spacegrain':100,
    'df_azim':-50,
    'df_elev':20,
    'df_fig_size':(15, 12),
    'df_rbffunc':'thin_plate',
    'df_colorscale':'BlueRed',
    'df_monthlies':True,
    'df_opacity':1,
    'df_surf':True,
    'df_save_image':False,
    'df_image_folder':'images',
    'df_image_filename':'impvol',
    'df_image_dpi':50,
    
    # Dictionary of implied vol fields used in graph methods
    'df_vols_dict':{
        'bid':'Imp Vol - Bid',
        'mid':'Imp Vol - Mid',
        'ask':'Imp Vol - Ask',
        'last':'Imp Vol - Last'
        },
    
    # Dictionary of price fields used for filtering zero prices in 
    # graph methods
    'df_prices_dict':{
        'bid':'Bid',
        'mid':'Mid',
        'ask':'Ask',
        'last':'Last Price'
        },
    
    # Dictionary of implied vol fields used in implied vol calculation
    'df_row_dict':{
        'Bid':'Imp Vol - Bid',
        'Mid':'Imp Vol - Mid',
        'Ask':'Imp Vol - Ask',
        'Last Price':'Imp Vol - Last'
        },
    
    # Dictionary of interpolation methods used in implied vol calculation
    'df_method_dict':{
        'nr':'implied_vol_newton_raphson',
        'bisection':'implied_vol_bisection',
        'naive':'implied_vol_naive'
        },
    
    # Parameters to overwrite mpl_style defaults
    'df_mpl_line_params':{
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
    
    'df_mpl_3D_params':{
        'axes.facecolor':'w',
        'axes.labelcolor':'k',
        'axes.edgecolor':'w',
        'lines.linewidth':0.5,
        'xtick.labelbottom':True,
        'ytick.labelleft':True
        },

    }