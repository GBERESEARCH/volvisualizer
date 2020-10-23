import volvisualizer.models as models
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
from bs4 import BeautifulSoup
import datetime as dt
import calendar


# Dictionary of default parameters
df_dict = {'vols_dict':{'bid':'Imp Vol - Bid',
                        'mid':'Imp Vol - Mid',
                        'ask':'Imp Vol - Ask',
                        'last':'Imp Vol - Last'},
           'prices_dict':{'bid':'Bid',
                          'mid':'Mid',
                          'ask':'Ask',
                          'last':'Last Price'},
           'row_dict':{'Bid':'Imp Vol - Bid',
                       'Mid':'Imp Vol - Mid',
                       'Ask':'Imp Vol - Ask',
                       'Last Price':'Imp Vol - Last'},
           'method_dict':{'nr':'implied_vol_newton_raphson',
                          'bisection':'implied_vol_bisection',
                          'naive':'implied_vol_naive'},
           'wait':5,
           'graphtype':'line', 
           'surfacetype':'mesh', 
           'smoothing':False, 
           'scatter':False, 
           'voltype':'last',
           'smoothopt':6,
           'notebook':False,
           'r':0.005, 
           'q':0, 
           'epsilon':0.001, 
           'method':'nr',
           'order':3,
           'spacegrain':100,
           'rbffunc':'thin_plate',
           'colorscale':'BlueRed',
           'monthlies':False}


class Volatility(models.ImpliedVol):
    
    def __init__(self, vols_dict=df_dict['vols_dict'], prices_dict=df_dict['prices_dict'], 
                 row_dict=df_dict['row_dict'], method_dict=df_dict['method_dict'], 
                 wait=df_dict['wait'], graphtype=df_dict['graphtype'], surfacetype=df_dict['surfacetype'], 
                 smoothing=df_dict['smoothing'], scatter=df_dict['scatter'], voltype=df_dict['voltype'], 
                 smoothopt=df_dict['smoothopt'], notebook=df_dict['notebook'], r=df_dict['r'], 
                 q=df_dict['q'], epsilon=df_dict['epsilon'], method=df_dict['method'], 
                 order=df_dict['order'], spacegrain=df_dict['spacegrain'], rbffunc=df_dict['rbffunc'], 
                 colorscale=df_dict['colorscale'], monthlies=df_dict['monthlies']):
        
        models.ImpliedVol.__init__(self)
        self.vols_dict = vols_dict # Dictionary of implied vol fields used in graph methods
        self.prices_dict = prices_dict # Dictionary of price fields used for filtering zero prices in graph methods
        self.row_dict = row_dict # Dictionary of implied vol fields used in implied vol calculation
        self.method_dict = method_dict # Dictionary of implied vol calculation methods used in implied vol calculation
        self.wait = wait # Time to wait between each url query
        self.surfacetype = surfacetype # Type of 3D graph
        self.smoothing = smoothing # Whether to graph implied vols smoothed using polyfit
        self.scatter = scatter # Whether to include scatter points in 3D meshplot
        self.voltype = voltype # Vol to use - Bid, Mid, Ask or Last
        self.smoothopt = smoothopt # Min number of options to include per tenor when applying smoothing
        self.notebook = notebook # Whether interactive graph is run in Jupyter notebook or IDE
        self.r = r # Interest Rate
        self.q = q # Dividend Yield
        self.epsilon = epsilon # Degree of precision that implied vol calculated to
        self.method = method # Choice of implied vol method
        self.order = order # Polynomial order used in smoothing
        self.spacegrain = spacegrain # Number of points in each axis linspace argument for 3D graphs
        self.rbffunc = rbffunc # Radial basis function used in interpolation
        self.colorscale = colorscale # Colors used in plotly interactive graph
        self.monthlies = monthlies # Whether to filter expiry dates to just 3rd Friday of month
        
    
    def extracturls(self, ticker):
        """
        Extract the URL for each of the listed option on Yahoo Finance for the given ticker. 

        Parameters
        ----------
        ticker : Str
            Yahoo ticker (Reuters RIC) for the stock.

        Returns
        -------
        Dict
            Dictionary of dates and URLs.

        """
        
        # Define the stock root webpage
        url = 'https://finance.yahoo.com/quote/'+ticker+'/options?p='+ticker
        
        # Create a requests object to extract data from the url
        r = requests.get(url)
        
        # Collect the text fromthis object
        html_doc = r.text
        
        # Use Beautiful Soup to parse this
        soup = BeautifulSoup(html_doc, features="lxml")
        
        # Create a list of all the option dates 
        option_dates = [a.get_text() for a in soup.find_all('option')]
        
        # Convert this list from string to datetimes 
        dates_list = [dt.datetime.strptime(date, "%B %d, %Y").date() for date in option_dates]
        
        # Convert back to strings in the required format
        str_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in dates_list]
        
        # Create a list of all the unix dates used in the url for each of these dates
        option_pages = [a.attrs['value'] for a in soup.find_all('option')]
        
        # Combine the dates and unixdates in a dictionary
        optodict = dict(zip(str_dates, option_pages))
        
        # Create an empty dictionary
        self.url_dict = {}
        
        # For each date and unixdate in the first dictionary
        for date_val, page in optodict.items():
            
            # Create an entry with the date as key and the url plus unix date as value
            self.url_dict[date_val] = str('https://finance.yahoo.com/quote/'+ticker+'/options?date='+page)
        
        return self    
    
        
    def extractoptions(self, url_dict=None, wait=None):
        """
        Extract option data from Yahoo Finance

        Parameters
        ----------
        url_dict : Dict
            Dictionary of URLs to download option prices from.
        wait : Int
            Number of seconds to wait between each url query

        Returns
        -------
        DataFrame
            All option data from each of the supplied urls.

        """    
        
        # If inputs are not supplied, take existing values
        if url_dict is None:
            url_dict = self.url_dict
        if wait is None:
            wait = self.wait
        
        # Create an empty dictionary
        df_dict = {}
        self.url_except_dict = {}
        
        # each url needs to have an option expiry date associated with it in the url dict 
        for input_date, url in url_dict.items():
            
            # requests function downloads the data            
            html = requests.get(url).content
            
            # wait between each query so as not to overload server
            time.sleep(wait)
            
            # if data exists
            try:
                # read html data into a DataFrame 
                df = pd.read_html(html)
                
                # Add this DataFrame to the default dictionary, named with the expiry date it refers to
                df_dict[input_date] = df
            
            # otherwise collect dictionary of exceptions
            except:
                self.url_except_dict[input_date] = url
        
        # Create an empty DataFrame
        self.full_data = pd.DataFrame()
        
        # Make a list of all the dates of the DataFrames just stored in the default dictionary
        date_list = list(df_dict.keys()) 
        
        # Create list to store exceptions
        self.opt_except_list = []
        
        # For each of these dates
        for input_date in date_list:
            
            try:
                # The first entry is 'calls'
                calls = df_dict[input_date][0]
                
                # Create a column designating these as calls
                calls['Option Type'] = 'call'

                try:
                    # The second entry is 'puts'
                    puts = df_dict[input_date][1]
                    
                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'
                    
                    # Concatenate these two DataFrames
                    options = pd.concat([calls, puts])
    
                    # Add an 'Expiry' column with the expiry date
                    options['Expiry'] = pd.to_datetime(input_date).date()
                    
                    # Add this DataFrame to 'full_data'
                    self.full_data = pd.concat([self.full_data, options])

                except:
 
                    # Add an 'Expiry' column with the expiry date
                    calls['Expiry'] = pd.to_datetime(input_date).date()
                    
                    # Add this DataFrame to 'full_data'
                    self.full_data = pd.concat([self.full_data, calls])

            except:
                
                try:
                    # The second entry is 'puts'
                    puts = df_dict[input_date][1]
                    
                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'
                    
                    # Add an 'Expiry' column with the expiry date
                    puts['Expiry'] = pd.to_datetime(input_date).date()
                    
                    # Add this DataFrame to 'full_data'
                    self.full_data = pd.concat([self.full_data, puts])
            
                except:
                    self.opt_except_list.append(input_date)

        
        return self


    def transform(self, start_date, lastmins=None, mindays=None, minopts=None, volume=None, 
                  openint=None, monthlies=None):    
        """
        Perform some filtering / transforming of the option data

        Parameters
        ----------
        start_date : Str
            Date from when to include prices (some of the options won't have traded 
                                              for days / weeks and therefore will have stale prices).
        lastmins : Int, Optional 
            Restrict to trades within number of minutes since last trade time recorded
        mindays : Int, Optional    
            Restrict to options greater than certain option expiry
        minopts : Int, Optional
            Restrict to minimum number of options to include that option expiry
        volume : Int, Optional    
            Restrict to minimum Volume
        openint : Int, Optional
            Restrict to minimum Open Interest
        monthlies : Bool    
            Restrict expiries to only 3rd Friday of the month. Default is False.
            
        Returns
        -------
        DataFrame
            Creates a new DataFrame as a modification of 'full_data'.

        """
        
        if monthlies is None:
            monthlies = self.monthlies
        
        # Assign start date to the object
        self.start_date = start_date

        # Make a copy of 'full_data'
        self.data = self.full_data.copy()

        # Set timezone
        est = pytz.timezone('US/Eastern')

        # Convert 'Last Trade Date' to a DateTime variable
        self.data['Last Trade Date Raw'] = self.data['Last Trade Date']
        self.data['Last Trade Date'] = pd.to_datetime(self.data['Last Trade Date'], format='%Y-%m-%d %I:%M%p EDT')
        self.data['Last Trade Date'] = self.data['Last Trade Date'].apply(lambda x: x.replace(tzinfo=est))

        # Create columns of expiry date as datetime object and str
        self.data['Expiry_datetime'] = pd.to_datetime(self.data['Expiry'], format='%Y-%m-%d')
        self.data['Expiry_str'] = self.data['Expiry_datetime'].dt.strftime('%Y-%m-%d')

        # Filter data from start date
        self.data = self.data[self.data['Last Trade Date']  >= str(pd.to_datetime(start_date))]

        # Create a column of the Trade Day
        self.data['Last Trade Day'] = self.data['Last Trade Date'].dt.date

        # Create a column of the Trade Time of Day
        self.data['Last Trade Time'] = self.data['Last Trade Date'].dt.time

        # Create a column of the Trade Hour of Day
        self.data['Last Trade Hour'] = self.data['Last Trade Date'].dt.hour

        # Create a column of the Trade Date represented in unixtime
        self.data['Unixtime'] = self.data['Last Trade Date'].astype(np.int64) // 10**9

        # Clean Volume column
        self.data['Volume'] = self.data['Volume'].replace('-',0).astype(int)

        # Clean Open Interest column
        self.data['Open Interest'] = self.data['Open Interest'].replace('-',0).astype(int)

        # Clean Bid column 
        self.data['Bid'] = self.data['Bid'].replace('-',0).astype(float)

        # Create Mid column
        self.data['Mid'] = (self.data['Ask'] + self.data['Bid']) / 2

        # Create Time To Maturity (in years) column
        self.data['TTM'] = (pd.to_datetime(self.data['Expiry']) - pd.to_datetime(date.today())) / (pd.Timedelta(days=1) * 365)

        # Create Days to Maturity column
        self.data['Days'] = (self.data['TTM']*365).astype(int)
        
        # If a minutes parameter is supplied, filter for most recent minutes
        if lastmins is not None:
            self.data = self.data[self.data['Unixtime']  >= (max(self.data['Unixtime']) - lastmins * 60)]
        
        # If a mindays parameter is supplied, filter for option expiry greater than parameter 
        if mindays is not None:
            self.data = self.data[self.data['Days']  >= mindays]
        
        # If a minopts parameter is supplied, filter for volume greater than parameter 
        if minopts is not None:
            
            # Create a dictionary of the number of options for each maturity
            mat_dict = dict(Counter(self.data['Days']))
            for ttm, count in mat_dict.items():
                
                # if there are less than minopts options for a given maturity
                if count < minopts:
                    
                    # remove that maturity from the DataFrame
                    self.data = self.data[self.data['Days'] != ttm]  
        
        # If a volume parameter is supplied, filter for volume greater than parameter 
        if volume is not None:
            self.data = self.data[self.data['Volume']  >= volume]
        
        # If an openint parameter is supplied, filter for Open Interest greater than parameter   
        if openint is not None:
            self.data = self.data[self.data['Open Interest']  >= openint]                          
        
        # If the monthlies flag is set
        if monthlies == True:
            
            # Create an empty list
            date_list = []
            
            # For each date in the url_dict 
            for key in self.url_dict.keys():
                
                # Format that string as a datetime object
                key_date = dt.datetime.strptime(key, "%Y-%m-%d")
                
                # Store the year and month as a tuple in date_list
                date_list.append((key_date.year, key_date.month))
            
            # Create a sorted list from the unique dates in date_list
            sorted_dates = sorted(list(set(date_list)))
            
            # Create an empty list
            days_list = []
            
            # Create a calendar object
            c = calendar.Calendar(firstweekday=calendar.SATURDAY)
            
            # For each tuple of year, month in sorted_dates
            for tup in sorted_dates:
                
                # Create a list of lists of days in that month
                monthcal = c.monthdatescalendar(tup[0], tup[1])
                
                # Extract the date corresponding to the 3rd Friday 
                expiry = monthcal[2][-1]
                
                # Calculate the number of days until that expiry
                ttm = (expiry - dt.date.today()).days
                
                # Append this to the days_list
                days_list.append(ttm)
            
            # For each unique number of days to expiry
            for days_to_expiry in set(self.data['Days']):
            
                # if the expiry is not in the list of monthly expiries
                if days_to_expiry not in days_list:
                    
                    # Remove that expiry from the DataFrame
                    self.data = self.data[self.data['Days'] != days_to_expiry] 
         
        return self
    

    def _imp_vol_by_row(self, row, S, K, r, q, epsilon, option, method):
        """
        Private function used to calculate implied vol for one row of a DataFrame.

        Parameters
        ----------
        row : Array
            Each row in the DataFrame.
        S : Float
            Stock Price.
        K : Float
            Strike Price.
        r : Float
            Interest Rate.
        q : Float
            Dividend Yield.
        epsilon : Float
            Degree of precision to return implied vol. 
        option : Str
            Option type; 'put' or 'call'.
        method : Str
            Implied Vol method; 'newtonraphson', 'bisection' or 'iv_naive'.

        Returns
        -------
        row : Array
            Each row in the DataFrame.

        """
        
        # For the chosen implied vol method and its method name
        for flag, func_name in self.method_dict.items():
            
            # select the method from the dictionary
            if method == flag:
                
                # for each of the prices: bid, mid, ask, last
                for input_row, output_row in self.row_dict.items():
                    
                    # populate the column using the chosen implied vol method 
                    # (using getattr() to select dynamically)
                    row[output_row] = getattr(self, func_name)(S=S, K=K, T=row['TTM'], 
                                                               r=r, q=q, cm=row[input_row], 
                                                               epsilon=epsilon, option=option, 
                                                               timing=False)
        
        return row
    

    def _imp_vol_apply(self, input_data, S, K, r, q, epsilon, option, method):
        """
        Private function used to apply _implied_vol_by_row method to each row of a DataFrame.

        Parameters
        ----------
        input_data : DataFrame
            DataFrame of Option prices.
        S : Float
            Stock Price.
        K : Float
            Strike Price.
        r : Float
            Interest Rate. The default is 0.005.
        q : Float
            Dividend Yield. The default is 0.
        epsilon : Float
            Degree of precision to return implied vol. The default is 0.001.
        option : Str
            Option type; 'put' or 'call'.
        method : Str
            Implied Vol method; 'nr', 'bisection' or 'naive'. The default is 'nr'.

        Returns
        -------
        input_data : DataFrame
            DataFrame of Option prices.

        """
        
        # Filter data by strike and option type
        input_data = input_data[(input_data['Strike'] == K) & (input_data['Option Type'] == option)]
        
        # Apply implied vol method to each row
        input_data = input_data.apply(lambda x: self._imp_vol_by_row(x, S, K, r, q, epsilon, option, method), axis=1)
        
        return input_data         
    
    
    def combine(self, ticker_label, put_strikes, call_strikes, spot, 
                r=None, q=None, epsilon=None, method=None):
        """
        Calculate implied volatilities for specified put and call strikes and combine.

        Parameters
        ----------
        ticker_label : Str
            Stcok Ticker.
        put_strikes : List
            Range of put strikes to calculate implied volatility for.
        call_strikes : List
            Range of call strikes to calculate implied volatility for.
        spot : Float
            Underlying reference level.
        r : Float
            Interest Rate. The default is 0.005.
        q : Float
            Dividend Yield. The default is 0.
        epsilon : Float
            Degree of precision to return implied vol. The default is 0.001.
        method : Str
            Implied Vol method; 'nr', 'bisection' or 'naive'. The default is 'nr'.

        Returns
        -------
        DataFrame
            DataFrame of Option prices.

        """
        
        # If inputs are not supplied, take existing values
        if r is None:
            r = self.r
        if q is None:
            q = self.q
        if epsilon is None:
            epsilon = self.epsilon
        if method is None:
            method = self.method
        
        # create copy of filtered data
        input_data = self.data.copy()
        
        # Assign ticker label to the object
        self.ticker_label = ticker_label
        
        # Create empty list and dictionary for storing options
        self.opt_list = []
        self.opt_dict = {}
        
        # For each put strike price
        for strike in put_strikes:
            
            # Assign an option name of ticker plus strike 
            opt_name = ticker_label+'_'+str(strike)
            
            # store the implied vol results for that strike in the option dictionary 
            self.opt_dict[opt_name] = self._imp_vol_apply(input_data=input_data, S=spot, K=strike, 
                                                          r=r, q=q, epsilon=epsilon, option='put', 
                                                          method=method)
            
            # store the implied vol results for that strike in the option list
            self.opt_list.append(self.opt_dict[opt_name])
        
        # For each put strike price    
        for strike in call_strikes:
            
            # Assign an option name of ticker plus strike 
            opt_name = ticker_label+'_'+str(strike)
            
            # store the implied vol results DataFrame for that strike in the option dictionary 
            self.opt_dict[opt_name] = self._imp_vol_apply(input_data=input_data, S=spot, K=strike, 
                                                          r=r, q=q, epsilon=epsilon, 
                                                          option='call', method=method)
            
            # store the implied vol results DataFrame for that strike in the option list
            self.opt_list.append(self.opt_dict[opt_name])    
        
        # Concatenate all the option results into a single DataFrame
        self.imp_vol_data = pd.concat(self.opt_list)
    
        return self
    

    def _vol_map(self, row):
        """
        Map value calculated in smooth surface DataFrame to 'Smoothed Vol' column.

        Parameters
        ----------
        row : Array
            Each row in the DataFrame.

        Returns
        -------
        row : Array
            Each row in the DataFrame.

        """
        row['Smoothed Vol'] = self.smooth_surf.loc[row['Strike'], str(row['Days'])]
        
        return row
       
    
    def smooth(self, order=None, voltype=None, smoothopt=None):
        """
        Create a column of smoothed implied vols

        Parameters
        ----------
        order : Int
            Polynomial order used in numpy polyfit function. The default is 3.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The default is 'last'.
        smoothopt : Int    
            Minimum number of options to fit curve to. The default is 6.
        
        Returns
        -------
        DataFrame
            DataFrame of Option prices.

        """
        
        # If inputs are not supplied, take existing values
        if order is None:
            order = self.order
        if voltype is None:
            voltype = self.voltype
        if smoothopt is None:
            smoothopt = self.smoothopt
        
        # Assign voltype to the object
        self.voltype = voltype
        
        # Create a dictionary of the number of options for each maturity
        self.mat_dict = dict(Counter(self.imp_vol_data['Days']))
        
        # Create a sorted list of the different number of days to maturity
        self.maturities = sorted(list(set(self.imp_vol_data['Days'])))
        
        # Create a sorted list of the different number of strikes
        self.strikes_full = sorted(list(set((self.imp_vol_data['Strike'].astype(int)))))
        
        # create copy of implied vol data
        self.imp_vol_data_smoothed = self.imp_vol_data.copy()
        
        for ttm, count in self.mat_dict.items():
            
            # if there are less than smoothopt (default is 6) options for a given maturity
            if count < smoothopt:
                
                # remove that maturity from the maturities list
                self.maturities.remove(ttm)
                
                # and remove that maturity from the implied vol DataFrame
                self.imp_vol_data_smoothed = self.imp_vol_data_smoothed[self.imp_vol_data_smoothed['Days'] != ttm]            
        
        # Create empty DataFrame with the full range of strikes as index
        self.smooth_surf = pd.DataFrame(index=self.strikes_full)
        
        # going through the maturity list (in reverse so the columns created are in increasing order)
        for maturity in reversed(self.maturities):
            
            # Extract the strikes for this maturity
            strikes = self.imp_vol_data[self.imp_vol_data['Days']==maturity]['Strike']
            
            # And the vols (specifying the voltype)
            vols = self.imp_vol_data[self.imp_vol_data['Days']==maturity][str(self.vols_dict[str(self.voltype)])]
            
            # Fit a polynomial to this data
            curve_fit = np.polyfit(strikes, vols, order)
            p = np.poly1d(curve_fit)
            
            # Create empty list to store smoothed implied vols
            iv_new = []
            
            # For each strike
            for strike in self.strikes_full:
                
                # Add the smoothed value to the iv_new list 
                iv_new.append(p(strike))
            
            # Append this list as a new column in the smooth_surf DataFrame    
            self.smooth_surf.insert(0, str(maturity), iv_new) 
    
        # Apply the _vol_map function to add smoothed vol column to DataFrame
        self.imp_vol_data_smoothed = self.imp_vol_data_smoothed.apply(lambda x: self._vol_map(x), axis=1)

        return self

    
    def visualize(self, graphtype=None, surfacetype=None, smoothing=None, scatter=None, 
                  voltype=None, order=None, spacegrain=None, rbffunc=None, colorscale=None, 
                  notebook=None):
        """
        Visualize the implied volatility as 2D linegraph, 3D scatter or 3D surface

        Parameters
        ----------
        graphtype : Str
            Whether to display 'line', 'scatter' or 'surface'. The default is 'line'.
        surfacetype : Str
            The type of 3D surface to display from 'trisurf', 'mesh', spline', 
            'interactive_mesh' and 'interactive_spline'. The default is 'mesh'.
        smoothing : Bool
            Whether to apply polynomial smoothing. The default is False.
        scatter : Bool
            Whether to plot scatter points on 3D mesh grid. The default is False.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The default is 'last'.
        order : Int
            Polynomial order used in numpy polyfit function. The default is 3.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D graphs. The default is 100.
        rbffunc : Str
            Radial basis function used in interpolation chosen from 'multiquadric', 
            'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'. The 
            default is 'thin_plate'     
        colorscale : Str
            Colors used in plotly interactive graph. The default is 'BlueRed'
        notebook : Bool
            Whether interactive graph is run in Jupyter notebook or IDE. The default is False.

        Returns
        -------
        Displays the output of the chosen graphing method.

        """
              
        # If inputs are not supplied, take existing values
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
        if order == None:
            order = self.order    
        else:
            self.order = order
        if spacegrain == None:
            spacegrain = self.spacegrain    
        else:
            self.spacegrain = spacegrain
        if rbffunc == None:
            rbffunc = self.rbffunc
        else:
            self.rbffunc = rbffunc
        if colorscale == None:
            colorscale = self.colorscale
        else:
            self.colorscale = colorscale    
        if notebook is None:
            notebook = self.notebook
        else:
            self.notebook = notebook
        
        # Run method selected by graphtype
        if graphtype == 'line':
            self.line_graph(voltype=voltype)
        if graphtype == 'scatter':
            self.scatter_3D(voltype=voltype)
        if graphtype == 'surface':
            self.surface_3D(surfacetype=surfacetype, smoothing=smoothing, scatter=scatter, 
                            voltype=voltype, order=order, spacegrain=spacegrain, 
                            rbffunc=rbffunc, colorscale=colorscale, notebook=notebook)
            
    
    def line_graph(self, voltype=None):
        """
        Displays a linegraph of each option maturity plotted by strike and implied vol

        Parameters
        ----------
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The default is 'last'.

        Returns
        -------
        Linegraph.

        """
        
        # If inputs are not supplied, take existing values
        if voltype == None:
            voltype = self.voltype
        
        # Create a sorted list of the different number of option expiries
        dates = sorted(list(set(self.imp_vol_data['Expiry'])))
        
        # Create a sorted list of the different number of option time to maturity
        tenors = sorted(list(set(self.imp_vol_data['TTM'])))
        
        # Combine these in a dictionary
        tenor_date_dict = dict(zip(dates, tenors))
        
        # Create figure, axis objects        
        fig, ax = plt.subplots(figsize=(14,12))
        plt.style.use('seaborn-darkgrid')
        
        # Set fontsize of axis ticks
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # For each expiry date
        for exp_date, tenor in tenor_date_dict.items():
            
            # Plot the specified voltype against strike
            ax.plot(self.imp_vol_data[self.imp_vol_data['TTM']==tenor]['Strike'], 
                    self.imp_vol_data[self.imp_vol_data['TTM']==tenor][str(self.vols_dict[str(voltype)])] * 100, 
                    label=str(exp_date)+' Expiry')
        plt.grid(True)
        
        # Label axes 
        ax.set_xlabel('Strike', fontsize=14)
        ax.set_ylabel('Implied Volatility %', fontsize=14)
        
        # Set legend title and font sizes
        ax.legend(title="Option Expiry", fontsize=12, title_fontsize=15)
        
        # Specify title with ticker label, voltype and date and shift away from chart
        st = fig.suptitle(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=18, fontweight=0, color='black', style='italic', y=1.02)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.9)
        
        # Display graph
        plt.show()


    def scatter_3D(self, voltype=None):
        """
        Displays a 3D scatter plot of each option implied vol against strike and maturity

        Parameters
        ----------
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The default is 'last'.

        Returns
        -------
        3D Scatter plot.

        """
        
        # If inputs are not supplied, take existing values
        if voltype == None:
            voltype = self.voltype
        
        # Create figure and axis objects
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set fontsize of axis ticks
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Create copy of data
        self.data_3D = self.imp_vol_data.copy()
        
        # Filter out any zero prices
        self.data_3D = self.data_3D[self.data_3D[str(self.prices_dict[str(voltype)])] != 0]
        
        # Specify the 3 axis values
        x = self.data_3D['Strike']
        y = self.data_3D['TTM'] * 365
        z = self.data_3D[str(self.vols_dict[str(voltype)])] * 100
                
        # Label axes
        ax.set_xlabel('Strike', fontsize=14, labelpad=15)
        ax.set_ylabel('Time to Expiration (Days)', fontsize=14, labelpad=15)
        ax.set_zlabel('Implied Volatility %', fontsize=14, labelpad=15)
        
        # Specify title with ticker label, voltype and date
        ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=18)       
        
        # Display scatter, specifying colour to vary with z-axis and use colormap 'viridis'
        ax.scatter3D(x, y, z, c=z, cmap='viridis')
    

    def surface_3D(self, surfacetype=None, smoothing=None, scatter=None, voltype=None, 
                   order=None, spacegrain=None, rbffunc=None, colorscale=None, notebook=None):
        """
        Displays a 3D surface plot of the implied vol surface against strike and maturity

        Parameters
        ----------
        surfacetype : Str
            The type of 3D surface to display from 'trisurf', 'mesh', spline', 
            'interactive_mesh' and 'interactive_spline'. The default is 'mesh'.
        smoothing : Bool
            Whether to apply polynomial smoothing. The default is False.
        scatter : Bool
            Whether to plot scatter points on 3D mesh grid. The default is False.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The default is 'last'.
        order : Int
            Polynomial order used in numpy polyfit function. The default is 3.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D graphs. The default 
            is 100.
        rbffunc : Str
            Radial basis function used in interpolation chosen from 'multiquadric', 
            'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'. The 
            default is 'thin_plate'
        colorscale : Str
            Colors used in plotly interactive graph. The default is 'BlueRed'
        notebook : Bool
            Whether interactive graph is run in Jupyter notebook or IDE. The default 
            is False.

        Returns
        -------
        3D surface plot.

        """
        
        # If inputs are not supplied, take existing values
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
        if rbffunc == None:
            rbffunc = self.rbffunc
        if colorscale == None:
            colorscale = self.colorscale
        if notebook is None:
            notebook = self.notebook    
        
        # If smoothing is set to False
        if smoothing == False:
            
            # Create copy of implied vol data
            self.data_3D = self.imp_vol_data.copy()
            
            # Filter out any zero prices
            self.data_3D = self.data_3D[self.data_3D[str(self.prices_dict[str(voltype)])] != 0]
            
            # Set 'graph vol' to be the specified voltype
            self.data_3D['Graph Vol'] = self.data_3D[str(self.vols_dict[str(voltype)])]
        
        # Otherwise, if smoothing is set to True
        else:
            
            # Apply the smoothing function to the specified voltype
            self.smooth(order=order, voltype=voltype)
            
            # Create copy of implied vol data
            self.data_3D = self.imp_vol_data_smoothed.copy()
            
            # Filter out any zero prices
            self.data_3D = self.data_3D[self.data_3D[str(self.prices_dict[str(voltype)])] != 0]
            
            # Set 'graph vol' to be the smoothed vol
            self.data_3D['Graph Vol'] = self.data_3D['Smoothed Vol']
        
        # Specify the 3 axis values
        x = self.data_3D['Strike']
        y = self.data_3D['TTM'] * 365
        z = self.data_3D['Graph Vol'] * 100
        
        
        if surfacetype == 'trisurf':
            
            # Create figure and axis objects
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Set fontsize of axis ticks
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Label axes
            ax.set_xlabel('Strike', fontsize=14, labelpad=15)
            ax.set_ylabel('Time to Expiration (Days)', fontsize=14, labelpad=15)
            ax.set_zlabel('Implied Volatility %', fontsize=14, labelpad=15)
            
            # Specify title with ticker label, voltype and date
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=18) 
            
            # Display triangular surface plot, using colormap 'viridis'
            ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')


        if surfacetype == 'mesh':
    
            # Create arrays across x and y-axes of equally spaced points from min to max values
            x1, y1 = np.meshgrid(np.linspace(min(x), max(x), int(self.spacegrain)), 
                                np.linspace(min(y), max(y), int(self.spacegrain)))
            
            # Map the z-axis with the scipy griddata method, applying cubic spline interpolation
            z1 = griddata(np.array([x,y]).T, np.array(z), (x1,y1), method='cubic')
            
            # Create figure and axis objects
            fig = plt.figure(figsize=(12, 9))
            ax = Axes3D(fig, azim=-60, elev=30)
            
            # Set fontsize of axis ticks
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Plot the surface
            ax.plot_surface(x1, y1, z1)
            
            # Apply contour lines
            ax.contour(x1, y1, z1)
            
            # Label axes
            ax.set_xlabel('Strike', fontsize=14, labelpad=15)
            ax.set_ylabel('Time to Expiration (Days)', fontsize=14, labelpad=15)
            ax.set_zlabel('Implied Volatility %', fontsize=14, labelpad=15)
            
            # Specify title with ticker label, voltype and date
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=18) 
            plt.show()


        if surfacetype == 'spline':

            # Create arrays across x and y-axes of equally spaced points from min to max values
            x1 = np.linspace(min(x), max(x), int(self.spacegrain))
            y1 = np.linspace(min(y), max(y), int(self.spacegrain))
            x2, y2 = np.meshgrid(x1, y1, indexing='xy')
            
            # Initialize the z-axis as an array of zero values
            z2 = np.zeros((x.size, z.size))
            
            # Apply scipy interpolate radial basis function, choosing the rbffunc parameter 
            spline = sp.interpolate.Rbf(x, y, z, function=rbffunc, smooth=5, episilon=5)
            
            # Populate z-axis array using this function
            z2 = spline(x2, y2)
            
            # Create figure and axis objects
            fig = plt.figure(figsize=(12,9))
            ax = Axes3D(fig)
            
            # Set fontsize of axis ticks
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Label axes
            ax.set_xlabel('Strike', fontsize=14, labelpad=15)
            ax.set_ylabel('Time to Expiration (Days)', fontsize=14, labelpad=15)
            ax.set_zlabel('Implied Volatility %', fontsize=14, labelpad=15)
            
            # Specify title with ticker label, voltype and date
            ax.set_title(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                     ' Price '+str(self.start_date), fontsize=18) 
            
            # Plot the surface
            ax.plot_wireframe(x2, y2, z2)
            ax.plot_surface(x2, y2, z2, alpha=0.2)
            
            # If scatter is True, overlay the surface with the unsmoothed scatter points
            if scatter == True:
                z = self.data_3D[str(self.vols_dict[str(voltype)])] * 100
                ax.scatter3D(x, y, z, c='r')


        if surfacetype in ['interactive_mesh', 'interactive_spline']:
            
            # Set the range of x, y and z contours and interval
            contour_x_start = 0
            contour_x_stop = 2 * 360
            contour_x_size = contour_x_stop / 18
            contour_y_start = self.data_3D['Strike'].min()
            contour_y_stop = self.data_3D['Strike'].max()
            
            # Vary the strike interval based on spot level
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

            # Specify the 3 axis values            
            x = self.data_3D['TTM'] * 365
            y = self.data_3D['Strike']
            z = self.data_3D['Graph Vol'] * 100
            
            # Create arrays across x and y-axes of equally spaced points from min to max values
            x1 = np.linspace(x.min(), x.max(), int(self.spacegrain))
            y1 = np.linspace(y.min(), y.max(), int(self.spacegrain))
            x2, y2 = np.meshgrid(x1, y1, indexing='xy')
        
            # If surfacetype is 'interactive_mesh', map the z-axis with the scipy 
            # griddata method, applying cubic spline interpolation
            if surfacetype == 'interactive_mesh':
                z2 = griddata((x, y), z, (x2, y2), method='cubic')
            
            # If surfacetype is 'interactive_spline', apply scipy interpolate radial 
            # basis function, choosing the rbffunc parameter
            if surfacetype == 'interactive_spline':
                z2 = np.zeros((x.size, z.size))
                spline = sp.interpolate.Rbf(x, y, z, function=rbffunc, smooth=5, episilon=5)
                z2 = spline(x2, y2)
            
            # Initialize Figure object
            fig = go.Figure(data=[go.Surface(x=x2, 
                                             y=y2, 
                                             z=z2,
                                             
                                             # Specify the colors to be used
                                             colorscale=colorscale,
                                             
                                             # Define the contours
                                             contours = {"x": {"show": True, "start": contour_x_start, 
                                                               "end": contour_x_stop, "size": contour_x_size, "color":"white"},            
                                                         "y": {"show": True, "start": contour_y_start, 
                                                               "end": contour_y_stop, "size": contour_y_size, "color":"white"},  
                                                         "z": {"show": True, "start": contour_z_start, 
                                                               "end": contour_z_stop, "size": contour_z_size}},)])
            
            # Set initial camera angle
            camera = dict(
                eye=dict(x=2, y=1, z=1)
            )
            
            # Set Time To Expiration to increase left to right
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
                                # Label axes
                                xaxis_title='Time to Expiration (Days)',
                                yaxis_title='Underlying Value',
                                zaxis_title='Implied Volatility %',),
                              # Specify title with ticker label, voltype and date
                              title={'text':(str(self.ticker_label.upper())+' Implied Volatility '+str(voltype.title())+
                                     ' Price '+str(self.start_date)),
                                     'y':0.9,
                                     'x':0.5,
                                     'xanchor':'center',
                                     'yanchor':'top',
                                     'font':dict(#family="Courier New, monospace",
                                               size=20,
                                               color="black")},
                              autosize=False, 
                              width=800, height=800,
                              margin=dict(l=65, r=50, b=65, t=90),
                              scene_camera=camera)
            
            # If running within a Jupyter notebook, plot graph inline
            if self.notebook == True:
                fig.show()
            
            # Otherwise create a new HTML window to display    
            else:
                plot(fig, auto_open=True)




        