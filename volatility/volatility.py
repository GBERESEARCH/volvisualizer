import models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import pytz
from datetime import date
from collections import Counter
import plotly.graph_objects as go
from scipy.interpolate import griddata
from plotly.offline import plot


class Volatility(models.ImpliedVol):
    
    def __init__(self):
        models.ImpliedVol.__init__(self)
        
        
    def extract(self, url_dict):
        
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
    
        row_dict = {'Last Price':'Imp Vol - Last', 
                    'Mid':'Imp Vol - Mid', 
                    'Bid':'Imp Vol - Bid', 
                    'Ask':'Imp Vol - Ask'}
        method_dict = {'NR':'newtonraphson',
                       'Bisection':'bisection',
                       'Naive':'iv_naive'}
        for flag, func_name in method_dict.items():
            if method == flag:
                for input_row, output_row in row_dict.items():
                    row[output_row] = getattr(self, func_name)(S=S, K=K, T=row['TTM'], 
                                                               r=r, q=q, cm=row[input_row], 
                                                               epsilon=epsilon, option=option)
        
        return row
    

    def _imp_vol_apply(self, input_data, S, K, r, q, epsilon, option, method):
        
        input_data = input_data[(input_data['Strike'] == K) & (input_data['Option Type'] == option)]
        input_data = input_data.apply(lambda x: self._imp_vol_by_row(x, S, K, r, q, epsilon, option, method), axis=1)
        
        return input_data         
    
    
    def combine(self, ticker_label, put_strikes, call_strikes, spot, 
                r=0.005, q=0, epsilon=0.001, method='NR'):
        
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
        
        row['Smoothed Vol'] = self.smooth_surf.loc[row['Strike'], str(row['Days'])]
        
        return row
       
    
    def smooth(self, order=3):
        
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
            vols = self.imp_vol_data[self.imp_vol_data['Days']==maturity]['Imp Vol - Last']
            curve_fit = np.polyfit(strikes, vols, order)
            p = np.poly1d(curve_fit)
            iv_new = []
            for strike in self.strikes_full:
                iv_new.append(p(strike))
            self.smooth_surf.insert(0, str(maturity), iv_new) 
    
        self.imp_vol_data_smoothed = self.imp_vol_data_smoothed.apply(lambda x: self._vol_map(x), axis=1)

        return self

    
    def visualize(self, type='line', smoothing=False, interactive=False, notebook=False):
        
        if type == 'line':
            self.line_graph()
        if type == 'scatter':
            self.scatter_3D()
        if type == 'surface':
            self.surface_3D(smoothing=smoothing, interactive=interactive, notebook=notebook)
            
    
    def line_graph(self):
        
        dates = list(set(self.imp_vol_data['Expiry']))
        dates.sort()
        tenors = list(set(self.imp_vol_data['TTM']))
        tenors.sort()
        tenor_date_dict = dict(zip(dates, tenors))
                
        fig, ax = plt.subplots()
        plt.style.use('seaborn-darkgrid')
        for exp_date, tenor in tenor_date_dict.items():
                ax.plot(self.imp_vol_data[self.imp_vol_data['TTM']==tenor]['Strike'], 
                        self.imp_vol_data[self.imp_vol_data['TTM']==tenor]['Imp Vol - Last'] * 100, 
                        label=str(exp_date)+' Expiry')
        plt.grid(True)
        ax.set(xlabel='Strike', ylabel='Implied Vol', title=str(self.ticker_label.upper())+' Implied Vol '+str(self.start_date))
        ax.legend()
        plt.show()


    def scatter_3D(self):
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        x = self.imp_vol_data['Strike']
        y = self.imp_vol_data['TTM'] * 365
        z = self.imp_vol_data['Imp Vol - Last'] * 100
        
        ax.set_xlabel('Strike', fontsize=12)
        ax.set_ylabel('Time To Maturity - Days', fontsize=12)
        ax.set_zlabel('Implied Volatility %', fontsize=12)
        ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(self.start_date), fontsize=14)
       
        ax.scatter3D(x, y, z, c=z, cmap='viridis')
    

    def surface_3D(self, smoothing=False, interactive=False, notebook=False):
        
        self.notebook = notebook
        self.interactive = interactive
        
        if smoothing == False:
            self.data_3D = self.imp_vol_data.copy()
        if smoothing == True:
            self.data_3D = self.imp_vol_data_smoothed.copy()
            self.data_3D = self.data_3D.drop(['Imp Vol - Last'], axis=1)
            self.data_3D = self.data_3D.rename(columns={'Smoothed Vol':'Imp Vol - Last'})
        
        if self.interactive == False:
        
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            x = self.data_3D['Strike']
            y = self.data_3D['TTM']
            z = self.data_3D['Imp Vol - Last']
            #ax.invert_xaxis()
            ax.set_xlabel('Strike', fontsize=12)
            ax.set_ylabel('Time To Maturity - Days', fontsize=12)
            ax.set_zlabel('Implied Volatility %', fontsize=12)
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(self.start_date), fontsize=14)

            ax.plot_trisurf(x, y*365, z*100, cmap='viridis', edgecolor='none')

        if self.interactive == True:

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
            
            x = self.data_3D['TTM']
            y = self.data_3D['Strike']
            z = self.data_3D['Imp Vol - Last']
            
            x1 = np.linspace(x.min(), x.max(), len(x.unique()))
            y1 = np.linspace(y.min(), y.max(), len(y.unique()))
            x2, y2 = np.meshgrid(x1, y1)
            z2 = griddata((x, y), z, (x2, y2), method='cubic')
            
            
            fig = go.Figure(data=[go.Surface(x=x2 * 365, 
                                             y=y2, 
                                             z=z2 * 100, 
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
                              title='Implied Vol '+str(self.ticker_label.upper())+' '+str(self.start_date), 
                              autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                              scene_camera=camera)
            
            if self.notebook == True:
                fig.show()
            else:
                plot(fig, auto_open=True)




        