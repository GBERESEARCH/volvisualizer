import models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import time
import pytz
from datetime import date
from collections import Counter
import plotly.graph_objects as go
import scipy as sp
from scipy.interpolate import griddata
from plotly.offline import plot

df_dict = {'vols_dict':{'bid':'Imp Vol - Bid',
                        'mid':'Imp Vol - Mid',
                        'ask':'Imp Vol - Ask',
                        'last':'Imp Vol - Last'},
           'row_dict':{'Last Price':'Imp Vol - Last', 
                       'Mid':'Imp Vol - Mid', 
                       'Bid':'Imp Vol - Bid', 
                       'Ask':'Imp Vol - Ask'},
           'method_dict':{'NR':'newtonraphson',
                          'Bisection':'bisection',
                          'Naive':'iv_naive'},
           'graphtype':'line', 
           'surfacetype':'mesh', 
           'smoothing':False, 
           'scatter':False, 
           'voltype':'last', 
           'notebook':False,
           'r':0.005, 
           'q':0, 
           'epsilon':0.001, 
           'method':'NR',
           'order':3,
           'spacegrain':100}

class Volatility(models.ImpliedVol):
    
    def __init__(self, vols_dict=df_dict['vols_dict'], row_dict=df_dict['row_dict'], 
                 method_dict=df_dict['method_dict'], graphtype=df_dict['graphtype'], 
                 surfacetype=df_dict['surfacetype'], smoothing=df_dict['smoothing'], 
                 scatter=df_dict['scatter'], voltype=df_dict['voltype'], notebook=df_dict['notebook'], 
                 r=df_dict['r'], q=df_dict['q'], epsilon=df_dict['epsilon'], method=df_dict['method'], 
                 order=df_dict['order'], spacegrain=df_dict['spacegrain']):
        models.ImpliedVol.__init__(self)
        self.vols_dict = vols_dict
        self.row_dict = row_dict
        self.method_dict = method_dict
        self.surfacetype = surfacetype
        self.smoothing = smoothing
        self.scatter = scatter
        self.voltype = voltype
        self.notebook = notebook
        self.r = r
        self.q = q
        self.epsilon = epsilon
        self.method = method
        self.order = order
        self.spacegrain = spacegrain
        
        
    def extract(self, url_dict):
        """
        

        Parameters
        ----------
        url_dict : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """    
        df_dict = {}
        for input_date, url in url_dict.items():
            html = requests.get(url).content
            time.sleep(5)
            df = pd.read_html(html)
            df_dict[input_date] = df
        
        self.full_data = pd.DataFrame()
        
        date_list = list(df_dict.keys()) 
        
        for input_date in date_list:
            calls = df_dict[input_date][0]
            calls['Option Type'] = 'call'
            puts = df_dict[input_date][1]
            puts['Option Type'] = 'put'
            options = pd.concat([calls, puts])
            options['Expiry'] = pd.to_datetime(input_date).date()
            self.full_data = pd.concat([self.full_data, options])
        
        return self


    def transform(self, start_date):    
        """
        

        Parameters
        ----------
        start_date : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.start_date = start_date
        self.data = self.full_data.copy()
        est = pytz.timezone('US/Eastern')
        self.data['Last Trade Date'] = pd.to_datetime(self.data['Last Trade Date'], format='%Y-%m-%d %H:%M%p EDT')
        self.data['Last Trade Date'] = self.data['Last Trade Date'].apply(lambda x: x.replace(tzinfo=est))
        self.data = self.data[self.data['Last Trade Date']  >= str(pd.to_datetime(start_date))]
        self.data['Bid'] = self.data['Bid'].replace('-',0).astype(float)
        self.data['Mid'] = (self.data['Ask'] + self.data['Bid']) / 2
        self.data['TTM'] = (pd.to_datetime(self.data['Expiry']) - pd.to_datetime(date.today())) / (pd.Timedelta(days=1) * 365)
        self.data['Days'] = (self.data['TTM']*365).astype(int)
        
        return self
    

    def _imp_vol_by_row(self, row, S, K, r, q, epsilon, option, method):
        """
        

        Parameters
        ----------
        row : TYPE
            DESCRIPTION.
        S : TYPE
            DESCRIPTION.
        K : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.
        epsilon : TYPE
            DESCRIPTION.
        option : TYPE
            DESCRIPTION.
        method : TYPE
            DESCRIPTION.

        Returns
        -------
        row : TYPE
            DESCRIPTION.

        """

        for flag, func_name in self.method_dict.items():
            if method == flag:
                for input_row, output_row in self.row_dict.items():
                    row[output_row] = getattr(self, func_name)(S=S, K=K, T=row['TTM'], 
                                                               r=r, q=q, cm=row[input_row], 
                                                               epsilon=epsilon, option=option)
        
        return row
    

    def _imp_vol_apply(self, input_data, S, K, r, q, epsilon, option, method):
        """
        

        Parameters
        ----------
        input_data : TYPE
            DESCRIPTION.
        S : TYPE
            DESCRIPTION.
        K : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.
        q : TYPE
            DESCRIPTION.
        epsilon : TYPE
            DESCRIPTION.
        option : TYPE
            DESCRIPTION.
        method : TYPE
            DESCRIPTION.

        Returns
        -------
        input_data : TYPE
            DESCRIPTION.

        """
        input_data = input_data[(input_data['Strike'] == K) & (input_data['Option Type'] == option)]
        input_data = input_data.apply(lambda x: self._imp_vol_by_row(x, S, K, r, q, epsilon, option, method), axis=1)
        
        return input_data         
    
    
    def combine(self, ticker_label, put_strikes, call_strikes, spot, 
                r=None, q=None, epsilon=None, method=None):
        """
        

        Parameters
        ----------
        ticker_label : TYPE
            DESCRIPTION.
        put_strikes : TYPE
            DESCRIPTION.
        call_strikes : TYPE
            DESCRIPTION.
        spot : TYPE
            DESCRIPTION.
        r : TYPE, optional
            DESCRIPTION. The default is 0.005.
        q : TYPE, optional
            DESCRIPTION. The default is 0.
        epsilon : TYPE, optional
            DESCRIPTION. The default is 0.001.
        method : TYPE, optional
            DESCRIPTION. The default is 'NR'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        if r is None:
            r = self.r
        if q is None:
            q = self.q
        if epsilon is None:
            epsilon = self.epsilon
        if method is None:
            method = self.method
        
        input_data = self.data.copy()
        self.ticker_label = ticker_label
        self.opt_list = []
        self.opt_dict = {}
        for strike in put_strikes:
            opt_name = ticker_label+'_'+str(strike)
            self.opt_dict[opt_name] = self._imp_vol_apply(input_data=input_data, S=spot, K=strike, 
                                                          r=r, q=q, epsilon=epsilon, option='put', 
                                                          method=method)
            self.opt_list.append(self.opt_dict[opt_name])
            
        for strike in call_strikes:
            opt_name = ticker_label+'_'+str(strike)
            self.opt_dict[opt_name] = self._imp_vol_apply(input_data=input_data, S=spot, K=strike, 
                                                          r=r, q=q, epsilon=epsilon, 
                                                          option='call', method=method)
            self.opt_list.append(self.opt_dict[opt_name])    
        
        self.imp_vol_data = pd.concat(self.opt_list)
    
        return self
    

    def _vol_map(self, row):
        """
        

        Parameters
        ----------
        row : TYPE
            DESCRIPTION.

        Returns
        -------
        row : TYPE
            DESCRIPTION.

        """
        row['Smoothed Vol'] = self.smooth_surf.loc[row['Strike'], str(row['Days'])]
        
        return row
       
    
    def smooth(self, order=None, voltype='last'):
        """
        

        Parameters
        ----------
        order : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        if order is None:
            order = self.order
        
        self.voltype = voltype
        self.mat_dict = dict(Counter(self.imp_vol_data['Days']))
        self.maturities = sorted(list(set(self.imp_vol_data['Days'])))
        self.strikes_full = sorted(list(set((self.imp_vol_data['Strike'].astype(int)))))
        
        self.imp_vol_data_smoothed = self.imp_vol_data.copy()
        
        for ttm, count in self.mat_dict.items():
            if count < 6:
                self.maturities.remove(ttm)
                self.imp_vol_data_smoothed = self.imp_vol_data_smoothed[self.imp_vol_data_smoothed['Days'] != ttm]            
        
        self.smooth_surf = pd.DataFrame(index=self.strikes_full)

        for maturity in reversed(self.maturities):
            strikes = self.imp_vol_data[self.imp_vol_data['Days']==maturity]['Strike']
            vols = self.imp_vol_data[self.imp_vol_data['Days']==maturity][str(self.vols_dict[str(self.voltype)])]
            curve_fit = np.polyfit(strikes, vols, order)
            p = np.poly1d(curve_fit)
            iv_new = []
            for strike in self.strikes_full:
                iv_new.append(p(strike))
            self.smooth_surf.insert(0, str(maturity), iv_new) 
    
        self.imp_vol_data_smoothed = self.imp_vol_data_smoothed.apply(lambda x: self._vol_map(x), axis=1)

        return self

    
    def visualize(self, graphtype=None, surfacetype=None, smoothing=None, scatter=None, 
                  voltype=None, notebook=None):
        """
        

        Parameters
        ----------
        graphtype : TYPE, optional
            DESCRIPTION. The default is 'line'.
        surfacetype : TYPE, optional
            DESCRIPTION. The default is 'mesh'.
        smoothing : TYPE, optional
            DESCRIPTION. The default is False.
        scatter : TYPE, optional
            DESCRIPTION. The default is False.
        voltype : TYPE, optional
            DESCRIPTION. The default is 'last'.
        notebook : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        if graphtype is None:
            graphtype = self.graphtype
        else:
            self.graphtype = graphtype 
        if surfacetype is None:
            surfacetype = self.surfacetype
        else:
            self.surfacetype = surfacetype            
        if smoothing is None:
            smoothing = self.smoothing
        else:
            self.smoothing = smoothing
        if scatter is None:
            scatter = self.scatter
        else:
            self.scatter = scatter
        if voltype == None:
            voltype = self.voltype    
        else:
            self.voltype = voltype
        if notebook is None:
            notebook = self.notebook
        else:
            self.notebook = notebook
        
        if graphtype == 'line':
            self.line_graph(voltype=voltype)
        if graphtype == 'scatter':
            self.scatter_3D(voltype=voltype)
        if graphtype == 'surface':
            self.surface_3D(surfacetype=surfacetype, scatter=scatter, voltype=voltype, notebook=notebook)
            
    
    def line_graph(self, voltype=None):
        """
        

        Parameters
        ----------
        voltype : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if voltype == None:
            voltype = self.voltype
        
        dates = list(set(self.imp_vol_data['Expiry']))
        dates.sort()
        tenors = list(set(self.imp_vol_data['TTM']))
        tenors.sort()
        tenor_date_dict = dict(zip(dates, tenors))
                
        fig, ax = plt.subplots(figsize=(12,9))
        plt.style.use('seaborn-darkgrid')
        for exp_date, tenor in tenor_date_dict.items():
                ax.plot(self.imp_vol_data[self.imp_vol_data['TTM']==tenor]['Strike'], 
                        self.imp_vol_data[self.imp_vol_data['TTM']==tenor][str(self.vols_dict[str(voltype)])] * 100, 
                        label=str(exp_date)+' Expiry')
        plt.grid(True)
        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Implied Volatility %', fontsize=12)
        ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=14)
        ax.legend()
        plt.show()


    def scatter_3D(self, voltype=None):
        """
        

        Parameters
        ----------
        voltype : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if voltype == None:
            voltype = self.voltype
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        x = self.imp_vol_data['Strike']
        y = self.imp_vol_data['TTM'] * 365
        z = self.imp_vol_data[str(self.vols_dict[str(voltype)])] * 100
        
        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Time To Maturity - Days', fontsize=12)
        ax.set_zlabel('Implied Volatility %', fontsize=12)
        ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=14)       
        ax.scatter3D(x, y, z, c=z, cmap='viridis')
    

    def surface_3D(self, surfacetype=None, smoothing=None, scatter=None, voltype=None, 
                   order=None, spacegrain=None, notebook=None):
        """
        

        Parameters
        ----------
        surfacetype : TYPE, optional
            DESCRIPTION. The default is None.
        smoothing : TYPE, optional
            DESCRIPTION. The default is None.
        scatter : TYPE, optional
            DESCRIPTION. The default is None.
        voltype : TYPE, optional
            DESCRIPTION. The default is None.
        order : TYPE, optional
            DESCRIPTION. The default is None.
        spacegrain : TYPE, optional
            DESCRIPTION. The default is None.
        notebook : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if notebook is None:
            notebook = self.notebook
        if surfacetype is None:
            surfacetype = self.surfacetype
        if smoothing is None:
            smoothing = self.smoothing
        if scatter is None:
            scatter = self.scatter
        if voltype == None:
            voltype = self.voltype    
        if order == None:
            order = self.order 
        if spacegrain == None:
            spacegrain = self.spacegrain    
        
        
        if smoothing == False:
            self.data_3D = self.imp_vol_data.copy()
            self.data_3D['Graph Vol'] = self.data_3D[str(self.vols_dict[str(voltype)])]
        else:
            self.smooth(order=order, voltype=voltype)
            self.data_3D = self.imp_vol_data_smoothed.copy()
            self.data_3D['Graph Vol'] = self.data_3D['Smoothed Vol']
        
        x = self.data_3D['Strike']
        y = self.data_3D['TTM'] * 365
        z = self.data_3D['Graph Vol'] * 100
        
        
        if surfacetype == 'trisurf':
        
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('Strike', fontsize=12)
            ax.set_ylabel('Time To Maturity - Days', fontsize=12)
            ax.set_zlabel('Implied Volatility %', fontsize=12)
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=14) 
            ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')


        if surfacetype == 'mesh':
    
            x1,y1 = np.meshgrid(np.linspace(min(x), max(x), int(self.spacegrain)), 
                                np.linspace(min(y), max(y), int(self.spacegrain)))
            z1 = griddata(np.array([x,y]).T, np.array(z), (x1,y1), method='cubic')
            fig = plt.figure(figsize=(12, 9))
            ax = Axes3D(fig, azim=-60, elev=30)
            ax.plot_surface(x1,y1,z1)
            ax.contour(x1,y1,z1)
            ax.set_xlabel('Strike', fontsize=12)
            ax.set_ylabel('Time To Maturity - Days', fontsize=12)
            ax.set_zlabel('Implied Volatility %', fontsize=12)
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=14) 
            plt.show()


        if surfacetype == 'spline':
 
            x1 = np.linspace(min(x), max(x), int(self.spacegrain))
            y1 = np.linspace(min(y), max(y), int(self.spacegrain))
            x2, y2 = np.meshgrid(x1, y1, indexing='xy')
            z2 = np.zeros((x.size, z.size))
            
            spline = sp.interpolate.Rbf(x, y, z, function='thin_plate', smooth=5, episilon=5)
            
            z2 = spline(x2, y2)
            fig = plt.figure(figsize=(12,9))
            ax = Axes3D(fig)
            ax.set_xlabel('Strike', fontsize=12)
            ax.set_ylabel('Time To Maturity - Days', fontsize=12)
            ax.set_zlabel('Implied Volatility %', fontsize=12)
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=14) 
            ax.plot_wireframe(x2, y2, z2)
            ax.plot_surface(x2, y2, z2, alpha=0.2)
            if scatter == True:
                ax.scatter3D(x, y, z, c='r')


        if surfacetype in ['interactive_mesh', 'interactive_spline']:

            contour_x_start = 0
            contour_x_stop = 2 * 360
            contour_x_size = contour_x_stop / 18
            contour_y_start = self.data_3D['Strike'].min()
            contour_y_stop = self.data_3D['Strike'].max()
            if (self.data_3D['Strike'].max() - self.data_3D['Strike'].min()) > 2000:
                contour_y_size = 200
            elif (self.data_3D['Strike'].max() - self.data_3D['Strike'].min()) > 1000:
                contour_y_size = 100
            elif (self.data_3D['Strike'].max() - self.data_3D['Strike'].min()) > 250:
                contour_y_size = 50
            else:
                contour_y_size = 25
            contour_z_start = 0
            contour_z_stop = 0.5
            contour_z_size = 0.05
            
            x = self.data_3D['TTM'] * 365
            y = self.data_3D['Strike']
            z = self.data_3D['Graph Vol'] * 100
            
            x1 = np.linspace(x.min(), x.max(), int(self.spacegrain))
            y1 = np.linspace(y.min(), y.max(), int(self.spacegrain))
            x2, y2 = np.meshgrid(x1, y1, indexing='xy')
        
            if surfacetype == 'interactive_mesh':
                z2 = griddata((x, y), z, (x2, y2), method='cubic')
            
            if surfacetype == 'interactive_spline':
                z2 = np.zeros((x.size, z.size))
                spline = sp.interpolate.Rbf(x, y, z, function='thin_plate', smooth=5, episilon=5)
                z2 = spline(x2, y2)
            
            fig = go.Figure(data=[go.Surface(x=x2, 
                                             y=y2, 
                                             z=z2, 
                                             colorscale='BlueRed', 
                                             contours = {"x": {"show": True, "start": contour_x_start, 
                                                               "end": contour_x_stop, "size": contour_x_size, "color":"white"},            
                                                         "y": {"show": True, "start": contour_y_start, 
                                                               "end": contour_y_stop, "size": contour_y_size, "color":"white"},  
                                                         "z": {"show": True, "start": contour_z_start, 
                                                               "end": contour_z_stop, "size": contour_z_size}},)])
            
            camera = dict(
                eye=dict(x=2, y=1, z=1)
            )
            
            
            fig.update_scenes(xaxis_autorange="reversed")
            fig.update_layout(scene = dict(
                                xaxis = dict(
                                     backgroundcolor="rgb(200, 200, 230)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",),
                                yaxis = dict(
                                    backgroundcolor="rgb(230, 200,230)",
                                    gridcolor="white",
                                    showbackground=True,
                                    zerolinecolor="white"),
                                zaxis = dict(
                                    backgroundcolor="rgb(230, 230,200)",
                                    gridcolor="white",
                                    showbackground=True,
                                    zerolinecolor="white",),
                                xaxis_title='Time to Expiration (Days)',
                                yaxis_title='Underlying Value',
                                zaxis_title='Implied Vol',),
                              title=(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                                     ' Price '+str(self.start_date)), 
                              autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                              scene_camera=camera)
            
            if self.notebook == True:
                fig.show()
            else:
                plot(fig, auto_open=True)




        