import os
import calendar
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import scipy as sp
import time
import optionmodels.models as models
import volvisualizer.volatility_params as vp
import warnings
from bs4 import BeautifulSoup
from lxml import html
from collections import Counter
from datetime import date
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from plotly.offline import plot
from scipy.interpolate import griddata

class Volatility(models.ImpliedVol):
    
    def __init__(self):
        
        # Inherit methods from models.ImpliedVol
        models.ImpliedVol.__init__(self)
        
        # Dictionary of parameter defaults
        self.vol_params_dict = vp.vol_params_dict 
        
        # Initialize fixed default parameters
        self._init_fixed_params()
        
        
    def _init_fixed_params(self):
        """
        Initialize fixed default parameters using values from parameters dict

        Returns
        -------
        Various parameters and dictionaries to the object.

        """
        # Dictionary of implied vol fields used in graph methods
        self.vols_dict = self.vol_params_dict['df_vols_dict'] 
        
        # Dictionary of price fields used for filtering zero prices in 
        # graph methods
        self.prices_dict = self.vol_params_dict['df_prices_dict'] 
        
        # Dictionary of implied vol fields used in implied vol 
        # calculation
        self.row_dict = self.vol_params_dict['df_row_dict'] 
        
        # Dictionary of implied vol calculation methods used in 
        # implied vol calculation
        self.method_dict = self.vol_params_dict['df_method_dict'] 
        
        # Parameters to overwrite mpl_style defaults for line graphs
        self.mpl_line_params = self.vol_params_dict['df_mpl_line_params'] 
        
        # Parameters to overwrite mpl_style defaults for 3D graphs
        self.mpl_3D_params = self.vol_params_dict['df_mpl_3D_params'] 
        
        
    def _refresh_params_current(self, **kwargs):
        """
        Set parameters for use in various pricing functions to the 
        current object values.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and set to current 
            object values if no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Try to set to the object value and assign to the object 
                # and to input dictionary otherwise use default value
                try:
                    v = self.__dict__[k]
                except:
                    v = self.vol_params_dict['df_'+str(k)]
                    self.__dict__[k] = v
                
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs        
    
    
    def _refresh_params_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions to the
        default values.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object 
                # and to input dictionary
                v = self.vol_params_dict['df_'+str(k)]
                self.__dict__[k] = v
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs        
    
    
    def create_option_data(
            self, start_date, ticker=None, ticker_label=None, wait=None, 
            lastmins=None, mindays=None, minopts=None, volume=None, 
            openint=None, monthlies=None, spot=None, put_strikes=None, 
            call_strikes=None, strike_limits=None, divisor=None, r=None, 
            q=None, epsilon=None, method=None):
        """
        Extract the URL for each of the listed option on Yahoo Finance 
        for the given ticker. Extract option data from each URL.
        
        Filter / transform the data and calculate implied volatilities for 
        specified put and call strikes.

        Parameters
        ----------
        start_date : Str
            Date from when to include prices (some of the options 
            won't have traded for days / weeks and therefore will 
            have stale prices).
        ticker : Str
            The ticker identifier used by Yahoo for the chosen stock. The 
            default is '^SPX'.
        ticker_label : TYPE, optional
            DESCRIPTION. The default is None.
        wait : Int
            Number of seconds to wait between each url query
        lastmins : Int, Optional 
            Restrict to trades within number of minutes since last 
            trade time recorded
        mindays : Int, Optional    
            Restrict to options greater than certain option expiry
        minopts : Int, Optional
            Restrict to minimum number of options to include that 
            option expiry
        volume : Int, Optional    
            Restrict to minimum Volume
        openint : Int, Optional
            Restrict to minimum Open Interest
        monthlies : Bool    
            Restrict expiries to only 3rd Friday of the month. Default 
            is False.
        spot : Float
            Underlying reference level.
        put_strikes : List
            Range of put strikes to calculate implied volatility for.
        call_strikes : List
            Range of call strikes to calculate implied volatility for.
        strike_limit : Tuple
            min and max strikes to use expressed as a decimal 
            percentage. The default is (0.5, 2.0).
        divisor : Int
            Distance between each strike in dollars. The default is 25 for SPX 
            and 10 otherwise.
        r : Float
            Interest Rate. The default is 0.005.
        q : Float
            Dividend Yield. The default is 0.
        epsilon : Float
            Degree of precision to return implied vol. The default 
            is 0.001.
        method : Str
            Implied Vol method; 'nr', 'bisection' or 'naive'. The 
            default is 'nr'.

        Returns
        -------
        DataFrame
            DataFrame of Option data.

        """
        
        # Extract URLs and option data
        self._extractoptions(ticker=ticker, wait=wait)
        print("Options data extracted")
        
        # Filter / transform data
        self._transform(
            start_date=start_date, lastmins=lastmins, mindays=mindays, 
            minopts=minopts, volume=volume, openint=openint, 
            monthlies=monthlies)
        print("Data transformed")
        
        # Calculate implied volatilities and combine
        self._combine(
            ticker_label=ticker_label, spot=spot, put_strikes=put_strikes, 
            call_strikes=call_strikes, divisor=divisor, r=r, q=q, 
            epsilon=epsilon, method=method)
        
        print("Data combined")
    
        
    def _extractoptions(self, ticker=None, wait=None):
        """
        Extract the URL for each of the listed option on Yahoo Finance 
        for the given ticker. Extract option data from each URL.
        

        Parameters
        ----------
        ticker : Str
            The ticker identifier used by Yahoo for the chosen stock. The 
            default is '^SPX'.
        wait : Int
            Number of seconds to wait between each url query

        Returns
        -------
        DataFrame
            All option data from each of the supplied urls.

        """    
        
        # If inputs are not supplied, take existing values
        ticker, wait = itemgetter(
            'ticker', 'wait')(self._refresh_params_current(
                ticker=ticker, wait=wait))
        
        # Extract dictionary of option dates and urls        
        self.url_dict = self._extracturls(ticker)
        print("URL's extracted")                
                
        # Create an empty dictionary
        option_dict = {}
        self.url_except_dict = {}
        
        # each url needs to have an option expiry date associated with
        # it in the url dict 
        for input_date, url in self.url_dict.items():
            
            # requests function downloads the data            
            html = requests.get(url).content
            
            # wait between each query so as not to overload server
            time.sleep(wait)
            
            # if data exists
            try:
                # read html data into a DataFrame 
                option_frame = pd.read_html(html)
                
                # Add this DataFrame to the default dictionary, named 
                # with the expiry date it refers to
                option_dict[input_date] = option_frame
            
            # otherwise collect dictionary of exceptions
            except:
                self.url_except_dict[input_date] = url
        
        # Create an empty DataFrame
        self.full_data = pd.DataFrame()
        
        # Make a list of all the dates of the DataFrames just stored 
        # in the default dictionary
        date_list = list(option_dict.keys()) 
        
        # Create list to store exceptions
        self.opt_except_list = []
        
        # For each of these dates
        for input_date in date_list:
            
            try:
                # The first entry is 'calls'
                calls = option_dict[input_date][0]
                
                # Create a column designating these as calls
                calls['Option Type'] = 'call'

                try:
                    # The second entry is 'puts'
                    puts = option_dict[input_date][1]
                    
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
                    puts = option_dict[input_date][1]
                    
                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'
                    
                    # Add an 'Expiry' column with the expiry date
                    puts['Expiry'] = pd.to_datetime(input_date).date()
                    
                    # Add this DataFrame to 'full_data'
                    self.full_data = pd.concat([self.full_data, puts])
            
                except:
                    self.opt_except_list.append(input_date)

        
        return self


    def _extracturls(self, ticker):
        """
        Extract the URL for each of the listed option on Yahoo Finance 
        for the given ticker. 

        Parameters
        ----------
        ticker : Str
            Yahoo ticker (Reuters RIC) for the stock.

        Returns
        -------
        Dict
            Dictionary of dates and URLs.

        """
        self.ticker = ticker
        
        # Define the stock root webpage
        url = 'https://finance.yahoo.com/quote/'+ticker+'/options?p='+ticker
        
        # Create a requests object to extract data from the url
        self.requestslink = requests.get(url)
        
        # Collect the text fromthis object
        html_doc = self.requestslink.text
        
        # Use Beautiful Soup to parse this
        soup = BeautifulSoup(html_doc, features="lxml")
        
        # Create a list of all the option dates 
        option_dates = [a.get_text() for a in soup.find_all('option')]
        
        # Convert this list from string to datetimes 
        dates_list = [dt.datetime.strptime(date, "%B %d, %Y").date() for date 
                      in option_dates]
        
        # Convert back to strings in the required format
        str_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in dates_list]
        
        # Create a list of all the unix dates used in the url for each 
        # of these dates
        option_pages = [a.attrs['value'] for a in soup.find_all('option')]
        
        # Combine the dates and unixdates in a dictionary
        optodict = dict(zip(str_dates, option_pages))
        
        # Create an empty dictionary
        url_dict = {}
        
        # For each date and unixdate in the first dictionary
        for date_val, page in optodict.items():
            
            # Create an entry with the date as key and the url plus 
            # unix date as value
            url_dict[date_val] = str('https://finance.yahoo.com/quote/'+
                                          ticker+'/options?date='+page)
        
        return url_dict    
    

    def _transform(self, start_date, lastmins=None, mindays=None, minopts=None, 
                  volume=None, openint=None, monthlies=None):    
        """
        Perform some filtering / transforming of the option data

        Parameters
        ----------
        start_date : Str
            Date from when to include prices (some of the options 
            won't have traded for days / weeks and therefore will 
            have stale prices).
        lastmins : Int, Optional 
            Restrict to trades within number of minutes since last 
            trade time recorded
        mindays : Int, Optional    
            Restrict to options greater than certain option expiry
        minopts : Int, Optional
            Restrict to minimum number of options to include that 
            option expiry
        volume : Int, Optional    
            Restrict to minimum Volume
        openint : Int, Optional
            Restrict to minimum Open Interest
        monthlies : Bool    
            Restrict expiries to only 3rd Friday of the month. Default 
            is False.
            
        Returns
        -------
        DataFrame
            Creates a new DataFrame as a modification of 'full_data'.

        """
        # If inputs are not supplied, take default values
        monthlies = itemgetter(
            'monthlies')(self._refresh_params_default(
                monthlies=monthlies))
        
        # Assign start date to the object
        self.start_date = start_date

        # Make a copy of 'full_data'
        self.data = self.full_data.copy()

        # Set timezone
        est = pytz.timezone('US/Eastern')

        # Convert 'Last Trade Date' to a DateTime variable
        self.data['Last Trade Date Raw'] = self.data['Last Trade Date']
        
        # Format date based on Eastern Daylight or Standard Time
        try:
            self.data['Last Trade Date'] = pd.to_datetime(
                self.data['Last Trade Date'], format='%Y-%m-%d %I:%M%p EDT')
        except:
            self.data['Last Trade Date'] = pd.to_datetime(
                self.data['Last Trade Date'], format='%Y-%m-%d %I:%M%p EST')
            
        self.data['Last Trade Date'] = self.data['Last Trade Date'].apply(
            lambda x: x.replace(tzinfo=est))

        # Create columns of expiry date as datetime object and str
        self.data['Expiry_datetime'] = pd.to_datetime(self.data['Expiry'], 
                                                      format='%Y-%m-%d')
        self.data['Expiry_str'] = self.data['Expiry_datetime'].dt.strftime(
            '%Y-%m-%d')

        # Filter data from start date
        self.data = self.data[self.data['Last Trade Date']  >= str(
            pd.to_datetime(start_date))]

        # Create a column of the Trade Day
        self.data['Last Trade Day'] = self.data['Last Trade Date'].dt.date

        # Create a column of the Trade Time of Day
        self.data['Last Trade Time'] = self.data['Last Trade Date'].dt.time

        # Create a column of the Trade Hour of Day
        self.data['Last Trade Hour'] = self.data['Last Trade Date'].dt.hour

        # Create a column of the Trade Date represented in unixtime
        self.data['Unixtime'] = self.data['Last Trade Date'].astype(
            np.int64) // 10**9

        # Clean Volume column
        self.data['Volume'] = self.data['Volume'].replace('-',0).astype(int)

        # Clean Open Interest column
        self.data['Open Interest'] = self.data['Open Interest'].replace(
            '-',0).astype(int)

        # Clean Ask column 
        self.data['Ask'] = self.data['Ask'].replace('-',0).astype(float)

        # Clean Bid column 
        self.data['Bid'] = self.data['Bid'].replace('-',0).astype(float)

        # Create Mid column
        self.data['Mid'] = (self.data['Ask'] + self.data['Bid']) / 2

        # Create Time To Maturity (in years) column
        self.data['TTM'] = (pd.to_datetime(self.data['Expiry']) - 
                            pd.to_datetime(date.today())) / (
                                pd.Timedelta(days=1) * 365)

        # Create Days to Maturity column
        self.data['Days'] = (self.data['TTM']*365).astype(int)
        
        # If a minutes parameter is supplied, filter for most recent 
        # minutes
        if lastmins is not None:
            self.data = self.data[self.data['Unixtime']  >= (max(
                self.data['Unixtime']) - lastmins * 60)]
        
        # If a mindays parameter is supplied, filter for option expiry
        # greater than parameter 
        if mindays is not None:
            self.data = self.data[self.data['Days']  >= mindays]
        
        # If a minopts parameter is supplied, filter for volume greater
        # than parameter 
        if minopts is not None:
            
            # Create a dictionary of the number of options for each 
            # maturity
            mat_dict = dict(Counter(self.data['Days']))
            for ttm, count in mat_dict.items():
                
                # if there are less than minopts options for a given 
                # maturity
                if count < minopts:
                    
                    # remove that maturity from the DataFrame
                    self.data = self.data[self.data['Days'] != ttm]  
        
        # If a volume parameter is supplied, filter for volume greater 
        # than parameter 
        if volume is not None:
            self.data = self.data[self.data['Volume']  >= volume]
        
        # If an openint parameter is supplied, filter for Open Interest 
        # greater than parameter   
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


    def _combine(self, ticker_label=None, spot=None, put_strikes=None, 
                call_strikes=None, strike_limits=None, divisor=None, r=None, 
                q=None, epsilon=None, method=None):
        """
        Calculate implied volatilities for specified put and call 
        strikes and combine.

        Parameters
        ----------
        ticker_label : Str
            Stock Ticker.
        spot : Float
            Underlying reference level.
        put_strikes : List
            Range of put strikes to calculate implied volatility for.
        call_strikes : List
            Range of call strikes to calculate implied volatility for.
        strike_limit : Tuple
            min and max strikes to use expressed as a decimal 
            percentage. The default is (0.5, 2.0).
        divisor : Int
            Distance between each strike in dollars. The default is 25 for SPX 
            and 10 otherwise.
        r : Float
            Interest Rate. The default is 0.005.
        q : Float
            Dividend Yield. The default is 0.
        epsilon : Float
            Degree of precision to return implied vol. The default 
            is 0.001.
        method : Str
            Implied Vol method; 'nr', 'bisection' or 'naive'. The 
            default is 'nr'.

        Returns
        -------
        DataFrame
            DataFrame of Option prices.

        """
        
        # If inputs are not supplied, take default values
        r, q, epsilon, method = itemgetter(
            'r', 'q', 'epsilon', 'method')(self._refresh_params_default(
                r=r, q=q, epsilon=epsilon, method=method))
        
        # Calculate strikes if strikes and spot price are not supplied.          
        spot, put_strikes, call_strikes = self._create_strike_range(
            spot=spot, put_strikes=put_strikes, call_strikes=call_strikes, 
            strike_limits=strike_limits, divisor=divisor)        
                
        # create copy of filtered data
        input_data = self.data.copy()
        
        # Assign ticker label to the object
        if ticker_label is None:
            ticker_label = self.ticker.lstrip('^')
        
        self.ticker_label = ticker_label
        
        # Create empty list and dictionary for storing options
        opt_list = []
        opt_dict = {}
        
        # For each put strike price
        for strike in put_strikes:
            
            # Assign an option name of ticker plus strike 
            opt_name = ticker_label+'_'+str(strike)
            
            # store the implied vol results for that strike in the 
            # option dictionary 
            opt_dict[opt_name] = self._imp_vol_apply(
                input_data=input_data, S=spot, K=strike, r=r, q=q, 
                epsilon=epsilon, option='put', method=method)
            
            # store the implied vol results for that strike in the 
            # option list
            opt_list.append(opt_dict[opt_name])
            
            print('Put option: ', opt_name)
        
        # For each put strike price    
        for strike in call_strikes:
            
            # Assign an option name of ticker plus strike 
            opt_name = ticker_label+'_'+str(strike)
            
            # store the implied vol results DataFrame for that strike 
            # in the option dictionary 
            opt_dict[opt_name] = self._imp_vol_apply(
                input_data=input_data, S=spot, K=strike, r=r, q=q, 
                epsilon=epsilon, option='call', method=method)
            
            # store the implied vol results DataFrame for that strike 
            # in the option list
            opt_list.append(opt_dict[opt_name])    
        
            print('Call option: ', opt_name)
        
        # Concatenate all the option results into a single DataFrame and drop 
        # any null values
        self.imp_vol_data = pd.concat(opt_list).dropna()
    
        return self


    def _create_strike_range(self, spot, put_strikes, call_strikes, 
                             strike_limits, divisor):
        
        # Set the distance between put strikes as 25 for SPX or 10 otherwise 
        # if not provided
        if divisor is None:
            if self.ticker == '^SPX':
                divisor = self.vol_params_dict['df_divisor_SPX']
            else:
                divisor = self.vol_params_dict['df_divisor']
        
        # Extract the spot level from the html data    
        if spot is None:
            tree = html.fromstring(self.requestslink.content)
            priceparse = tree.xpath(
                '//span[@class="Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)"]/text()')
            spot = float([str(p) for p in priceparse][0].replace(',',''))

        # Calculate the point to switch from put to call options
        roundspot = round(spot / divisor) * divisor
        
        if strike_limits is None:
            strike_limits = self.vol_params_dict['df_strike_limits'] 
        
        # Calculate put options (default is 1/2 spot level)
        if put_strikes is None:
            put_min = round(spot * strike_limits[0] / divisor) * divisor
            put_strikes = list(range(put_min, roundspot, divisor))

        # Calculate call options (default is twice the spot level)
        if call_strikes is None:
            call_max = round(spot * strike_limits[1] / divisor) * divisor
            call_strikes = list(range(roundspot, call_max, divisor))

        return spot, put_strikes, call_strikes
        

    def _imp_vol_apply(self, input_data, S, K, r, q, epsilon, option, method):
        """
        Apply _implied_vol_by_row method to each row of a DataFrame.

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
            Degree of precision to return implied vol. The default is 
            0.001.
        option : Str
            Option type; 'put' or 'call'.
        method : Str
            Implied Vol method; 'nr', 'bisection' or 'naive'. The 
            default is 'nr'.

        Returns
        -------
        input_data : DataFrame
            DataFrame of Option prices.

        """
        
        # Filter data by strike and option type
        input_data = input_data[(input_data['Strike'] == K) & (
            input_data['Option Type'] == option)]
        
        # Apply implied vol method to each row
        input_data = input_data.apply(lambda x: self._imp_vol_by_row(
            x, S, K, r, q, epsilon, option, method), axis=1)
        
        return input_data         
        

    def _imp_vol_by_row(self, row, S, K, r, q, epsilon, option, method):
        """
        Calculate implied vol for one row of a DataFrame.

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
            Implied Vol method; 'newtonraphson', 'bisection' or 
            'iv_naive'.

        Returns
        -------
        row : Array
            Each row in the DataFrame.

        """
        
        # Suppress runtime warnings caused by bad vol data
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # For the chosen implied vol method and its method name
        for flag, func_name in self.method_dict.items():
            
            # select the method from the dictionary
            if method == flag:
                
                # for each of the prices: bid, mid, ask, last
                for input_row, output_row in self.row_dict.items():
                    
                    try:
                        # populate the column using the chosen implied 
                        # vol method (using getattr() to select 
                        # dynamically)
                        # check if n/a value is returned and print error 
                        # message if so
                        output = getattr(self, func_name)(
                            S=S, K=K, T=row['TTM'], r=r, q=q, 
                            cm=row[input_row], epsilon=epsilon, option=option, 
                            timing=False)
                        
                        output = float(output)
                        row[output_row] = output
                    
                    except:
                        print("Error with option: Strike="+str(K)+
                                  " TTM="+str(round(row['TTM'], 3))+
                                  " vol="+str(row[input_row])+
                                  " option="+option)                            
        
        # Return warnings to default setting
        warnings.filterwarnings("default", category=RuntimeWarning)
        
        return row
    

    def visualize(self, graphtype=None, surfacetype=None, smoothing=None, 
                  scatter=None, voltype=None, order=None, spacegrain=None, 
                  azim=None, elev=None, fig_size=None, rbffunc=None, 
                  colorscale=None, opacity=None, surf=None, notebook=None, 
                  save_image=None, image_folder=None, image_dpi=None):
        """
        Visualize the implied volatility as 2D linegraph, 3D scatter 
        or 3D surface

        Parameters
        ----------
        graphtype : Str
            Whether to display 'line', 'scatter' or 'surface'. The 
            default is 'line'.
        surfacetype : Str
            The type of 3D surface to display from 'trisurf', 'mesh', 
            spline', 'interactive_mesh' and 'interactive_spline'. The 
            default is 'mesh'.
        smoothing : Bool
            Whether to apply polynomial smoothing. The default is False.
        scatter : Bool
            Whether to plot scatter points on 3D mesh grid. The default 
            is False.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The 
            default is 'last'.
        order : Int
            Polynomial order used in numpy polyfit function. The 
            default is 3.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. The default is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.    
        fig_size : Tuple
            3D graph size
        rbffunc : Str
            Radial basis function used in interpolation chosen from 
            'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 
            'quintic', 'thin_plate'. The default is 'thin_plate'     
        colorscale : Str
            Colors used in plotly interactive graph. The default is 
            'BlueRed'
        opacity : Float
            opacity of 3D interactive graph
        surf : Bool
            Plot surface in interactive graph    
        notebook : Bool
            Whether interactive graph is run in Jupyter notebook or 
            IDE. The default is False.
        save_image : Bool
            Whether to save a copy of the image as a png file. The default is 
            False
        image_folder : Str
            Location to save any images. The default is 'images'
        image_dpi : Int
            Resolution to save images. The default is 50.

        Returns
        -------
        Displays the output of the chosen graphing method.

        """
        
        if graphtype is None:
            graphtype = self.vol_params_dict['df_graphtype']

        # Run method selected by graphtype
        if graphtype == 'line':
            self.line_graph(voltype=voltype)
        if graphtype == 'scatter':
            self.scatter_3D(voltype=voltype, azim=azim, elev=elev, 
                            fig_size=fig_size)
        if graphtype == 'surface':
            self.surface_3D(
                surfacetype=surfacetype, smoothing=smoothing, scatter=scatter, 
                voltype=voltype, order=order, spacegrain=spacegrain, azim=azim, 
                elev=elev, fig_size=fig_size, rbffunc=rbffunc, 
                colorscale=colorscale, opacity=opacity, surf=surf, 
                notebook=notebook, save_image=save_image, 
                image_folder=image_folder, image_dpi=image_dpi)
            
    
    def line_graph(self, voltype=None, save_image=None, image_folder=None, 
                   image_dpi=None):
        """
        Displays a linegraph of each option maturity plotted by strike 
        and implied vol

        Parameters
        ----------
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The 
            default is 'last'.
        save_image : Bool
            Whether to save a copy of the image as a png file. The default is 
            False
        image_folder : Str
            Location to save any images. The default is 'images'
        image_dpi : Int
            Resolution to save images. The default is 50.    

        Returns
        -------
        Linegraph.

        """
        
        # If inputs are not supplied, take existing values
        voltype, save_image, image_folder, image_dpi = itemgetter(
            'voltype', 'save_image', 'image_folder', 
             'image_dpi')(self._refresh_params_default(
                voltype=voltype, save_image=save_image, 
                 image_folder=image_folder, image_dpi=image_dpi))
        
        # Create a sorted list of the different number of option 
        # expiries
        dates = sorted(list(set(self.imp_vol_data['Expiry'])))
        
        # Create a sorted list of the different number of option time 
        # to maturity
        tenors = sorted(list(set(self.imp_vol_data['TTM'])))
        
        # Combine these in a dictionary
        tenor_date_dict = dict(zip(dates, tenors))
        
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update(self.mpl_line_params)
        fig_size = (12, 9)
        
        # Create figure, axis objects        
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Create values that scale fonts with fig_size
        ax_font_scale = int(round(fig_size[0] * 1.5))
        title_font_scale = int(round(fig_size[0] * 2))
        
        # Set fontsize of axis ticks
        ax.tick_params(axis='both', which='major', labelsize=ax_font_scale)
        
        # For each expiry date
        for exp_date, tenor in tenor_date_dict.items():
            
            # Plot the specified voltype against strike
            ax.plot(
                np.array(self.imp_vol_data[self.imp_vol_data['TTM']==tenor][
                    'Strike']), 
                np.array(self.imp_vol_data[self.imp_vol_data['TTM']==tenor][
                    str(self.vols_dict[str(voltype)])] * 100), 
                label=str(exp_date)+' Expiry')
        
        plt.grid(True)
        
        # Label axes 
        ax.set_xlabel('Strike', fontsize=ax_font_scale)
        ax.set_ylabel('Implied Volatility %', fontsize=ax_font_scale)
        
        # Set legend title and font sizes
        ax.legend(title="Option Expiry", 
                  fontsize=ax_font_scale*0.6, 
                  title_fontsize=ax_font_scale*0.8)
        
        # Specify title with ticker label, voltype and date and shift 
        # away from chart
        st = fig.suptitle(str(self.ticker_label)+
                          ' Implied Volatility '+str(voltype.title())+
                          ' Price '+str(self.start_date), 
                          fontsize=title_font_scale, 
                          fontweight=0, 
                          color='black', 
                          style='italic', 
                          y=1.02)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.9)
        
        # Display graph
        plt.show()

        if save_image:
            # save the image as a png file
            self._image_save(fig=fig, image_folder=image_folder, 
                             image_dpi=image_dpi)
            

    def scatter_3D(self, voltype=None, azim=None, elev=None, fig_size=None, 
                   save_image=None, image_folder=None, image_dpi=None):
        """
        Displays a 3D scatter plot of each option implied vol against 
        strike and maturity

        Parameters
        ----------
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The 
            default is 'last'.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        fig_size : Tuple
            3D graph size    
        save_image : Bool
            Whether to save a copy of the image as a png file. The default is 
            False
        image_folder : Str
            Location to save any images. The default is 'images'
        image_dpi : Int
            Resolution to save images. The default is 50.    

        Returns
        -------
        3D Scatter plot.

        """
        
        # Refresh inputs if not supplied
        # For some we want them to persist between queries so take 
        # existing object values
        azim, elev, save_image, image_folder, image_dpi = itemgetter(
            'azim', 'elev', 'save_image', 'image_folder', 
             'image_dpi')(self._refresh_params_current(
                azim=azim, elev=elev, save_image=save_image, 
                 image_folder=image_folder, image_dpi=image_dpi))
        
        # For others we reset to default each time
        voltype, fig_size = itemgetter(
            'voltype', 'fig_size')(self._refresh_params_default(
                voltype=voltype, fig_size=fig_size))                                                                                            
                    
        # Create figure and axis objects and format
        fig, ax = self._graph_format(fig_size=fig_size, azim=azim, elev=elev, 
                                     voltype=voltype)
        
        # Create copy of data
        self.data_3D = self.imp_vol_data.copy()
        
        # Filter out any zero prices
        self.data_3D = self.data_3D[
            self.data_3D[str(self.prices_dict[str(voltype)])] != 0]
        
        # Specify the 3 axis values
        x = self.data_3D['Strike']
        y = self.data_3D['TTM'] * 365
        z = self.data_3D[str(self.vols_dict[str(voltype)])] * 100
        
        # Display scatter, specifying colour to vary with z-axis and use 
        # colormap 'viridis'
        ax.scatter3D(x, y, z, c=z, cmap='viridis')

        if save_image:
            # save the image as a png file
            self._image_save(fig=fig, image_folder=image_folder, 
                             image_dpi=image_dpi)
            

    def surface_3D(self, surfacetype=None, smoothing=None, scatter=None, 
                   voltype=None, order=None, spacegrain=None, azim=None, 
                   elev=None, fig_size=None, rbffunc=None, colorscale=None, 
                   opacity=None, surf=None, notebook=None, save_image=None, 
                   image_folder=None, image_dpi=None):
        """
        Displays a 3D surface plot of the implied vol surface against 
        strike and maturity

        Parameters
        ----------
        surfacetype : Str
            The type of 3D surface to display from 'trisurf', 'mesh', 
            'spline', 'interactive_mesh' and 'interactive_spline'. 
            The default is 'mesh'.
        smoothing : Bool
            Whether to apply polynomial smoothing. The default is False.
        scatter : Bool
            Whether to plot scatter points on 3D mesh grid. The default 
            is False.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The 
            default is 'last'.
        order : Int
            Polynomial order used in numpy polyfit function. The 
            default is 3.
        spacegrain : Int
            Number of points in each axis linspace argument for 3D 
            graphs. The default 
            is 100.
        azim : Float
            L-R view angle for 3D graphs. The default is -50.
        elev : Float
            Elevation view angle for 3D graphs. The default is 20.
        fig_size : Tuple
            3D graph size    
        rbffunc : Str
            Radial basis function used in interpolation chosen from 
            'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 
            'quintic', 'thin_plate'. The default is 'thin_plate'
        colorscale : Str
            Colors used in plotly interactive graph. The default is 
            'BlueRed'
        opacity : Float
            opacity of 3D interactive graph
        surf : Bool
            Plot surface in interactive graph
        notebook : Bool
            Whether interactive graph is run in Jupyter notebook or IDE. 
            The default is False.
        save_image : Bool
            Whether to save a copy of the image as a png file. The default is 
            False
        image_folder : Str
            Location to save any images. The default is 'images'
        image_dpi : Int
            Resolution to save images. The default is 50.    

        Returns
        -------
        3D surface plot.

        """

        # Refresh inputs if not supplied
        # For some we want them to persist between queries so take existing 
        # object values
        (surfacetype, smoothing, colorscale, azim, elev, 
         notebook, save_image, image_folder, image_dpi) = itemgetter(
             'surfacetype', 'smoothing', 'colorscale', 'azim', 'elev', 
             'notebook', 'save_image', 'image_folder', 
             'image_dpi')(self._refresh_params_current(
                 surfacetype=surfacetype, smoothing=smoothing, 
                 colorscale=colorscale, azim=azim, elev=elev,
                 notebook=notebook, save_image=save_image, 
                 image_folder=image_folder, image_dpi=image_dpi))
 

        # For others we reset to default each time
        (voltype, scatter, order, spacegrain, fig_size, rbffunc, 
         opacity, surf) = itemgetter(
                'voltype', 'scatter', 'order', 'spacegrain', 'fig_size', 
                'rbffunc', 'opacity', 'surf')(self._refresh_params_default(
                    voltype=voltype, scatter=scatter, order=order, 
                    spacegrain=spacegrain, fig_size=fig_size, rbffunc=rbffunc, 
                    opacity=opacity, surf=surf))
        
        
        # Suppress mpl user warning about data containing nan values
        warnings.filterwarnings("ignore", category=UserWarning, 
                                message='Z contains NaN values. This '
                                +'may result in rendering artifacts.')
        
        # If smoothing is set to False
        if smoothing == False:
            
            # Create copy of implied vol data
            self.data_3D = self.imp_vol_data.copy()
            
            # Filter out any zero prices
            self.data_3D = self.data_3D[self.data_3D[str(
                self.prices_dict[str(voltype)])] != 0]
            
            # Set 'graph vol' to be the specified voltype
            self.data_3D['Graph Vol'] = self.data_3D[str(
                self.vols_dict[str(voltype)])]
        
        # Otherwise, if smoothing is set to True
        else:
            
            # Apply the smoothing function to the specified voltype
            self._smooth(order=order, voltype=voltype)
            
            # Create copy of implied vol data
            self.data_3D = self.imp_vol_data_smoothed.copy()
            
            # Filter out any zero prices
            self.data_3D = self.data_3D[self.data_3D[str(
                self.prices_dict[str(voltype)])] != 0]
            
            # Set 'graph vol' to be the smoothed vol
            self.data_3D['Graph Vol'] = self.data_3D['Smoothed Vol']
        
        # Specify the 3 axis values
        x = self.data_3D['Strike']
        y = self.data_3D['TTM'] * 365
        z = self.data_3D['Graph Vol'] * 100
        
        
        if surfacetype == 'trisurf':
            
            # Create figure and axis objects and format
            fig, ax = self._graph_format(
                fig_size=fig_size, azim=azim, elev=elev, voltype=voltype)
           
            # Display triangular surface plot, using colormap 'viridis'
            ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')


        if surfacetype == 'mesh':
    
            # Create arrays across x and y-axes of equally spaced points 
            # from min to max values
            x1, y1 = np.meshgrid(
                np.linspace(min(x), max(x), int(self.spacegrain)), 
                np.linspace(min(y), max(y), int(self.spacegrain)))
            
            # Map the z-axis with the scipy griddata method, applying 
            # cubic spline interpolation
            z1 = griddata(np.array([x,y]).T, np.array(z), (x1,y1), 
                          method='cubic')
            
            # Create figure and axis objects and format
            fig, ax = self._graph_format(
                fig_size=fig_size, azim=azim, elev=elev, voltype=voltype)
                       
            # Plot the surface
            ax.plot_surface(x1, y1, z1)
            
            # Apply contour lines
            ax.contour(x1, y1, z1)
            
            plt.show()


        if surfacetype == 'spline':

            # Create arrays across x and y-axes of equally spaced points 
            # from min to max values
            x1 = np.linspace(min(x), max(x), int(self.spacegrain))
            y1 = np.linspace(min(y), max(y), int(self.spacegrain))
            x2, y2 = np.meshgrid(x1, y1, indexing='xy')
            
            # Initialize the z-axis as an array of zero values
            z2 = np.zeros((x.size, z.size))
            
            # Apply scipy interpolate radial basis function, choosing 
            # the rbffunc parameter 
            spline = sp.interpolate.Rbf(x, y, z, function=rbffunc, 
                                        smooth=5, epsilon=5)
            
            # Populate z-axis array using this function
            z2 = spline(x2, y2)
            
            # Create figure and axis objects and format
            fig, ax = self._graph_format(
                fig_size=fig_size, azim=azim, elev=elev, voltype=voltype)
            
            # Plot the surface
            ax.plot_wireframe(x2, y2, z2)
            ax.plot_surface(x2, y2, z2, alpha=0.2)
            
            # If scatter is True, overlay the surface with the 
            # unsmoothed scatter points
            if scatter:
                z = self.data_3D[str(self.vols_dict[str(self.voltype)])] * 100
                ax.scatter3D(x, y, z, c='r')
           

        if surfacetype in ['interactive_mesh', 'interactive_spline']:
            
            # Set the range of x, y and z contours and interval
            contour_x_start = 0
            contour_x_stop = 2 * 360
            contour_x_size = contour_x_stop / 18
            contour_y_start = self.data_3D['Strike'].min()
            contour_y_stop = self.data_3D['Strike'].max()
            
            # Vary the strike interval based on spot level
            if (self.data_3D['Strike'].max() - self.data_3D[
                    'Strike'].min()) > 2000: contour_y_size = 200
            elif (self.data_3D['Strike'].max() - self.data_3D[
                    'Strike'].min()) > 1000: contour_y_size = 100
            elif (self.data_3D['Strike'].max() - self.data_3D[
                    'Strike'].min()) > 250: contour_y_size = 50
            elif (self.data_3D['Strike'].max() - self.data_3D[
                    'Strike'].min()) > 50: contour_y_size = 10
            else:
                contour_y_size = 5
            
            # Set z contours    
            contour_z_start = 0
            contour_z_stop = 100
            contour_z_size = 5

            # Specify the 3 axis values            
            x = self.data_3D['TTM'] * 365
            y = self.data_3D['Strike']
            z = self.data_3D['Graph Vol'] * 100
            
            # Create arrays across x and y-axes of equally spaced 
            # points from min to max values
            x1 = np.linspace(x.min(), x.max(), int(self.spacegrain))
            y1 = np.linspace(y.min(), y.max(), int(self.spacegrain))
            x2, y2 = np.meshgrid(x1, y1, indexing='xy')
        
            # If surfacetype is 'interactive_mesh', map the z-axis with 
            # the scipy griddata method, applying cubic spline 
            # interpolation
            if surfacetype == 'interactive_mesh':
                z2 = griddata((x, y), z, (x2, y2), method='cubic')
            
            # If surfacetype is 'interactive_spline', apply scipy 
            # interpolate radial basis function, choosing the rbffunc 
            # parameter
            if surfacetype == 'interactive_spline':
                z2 = np.zeros((x.size, z.size))
                spline = sp.interpolate.Rbf(x, y, z, function=rbffunc, 
                                            smooth=5, epsilon=5)
                z2 = spline(x2, y2)
            
            # Initialize Figure object
            if scatter:
                
                # Set z to raw data points
                z = self.data_3D[str(self.vols_dict[str(voltype)])] * 100
                
                # Create figure object with fitted surface and scatter 
                # points
                if surf:
                    fig = go.Figure(data=[
                        go.Surface(x=x2, 
                                   y=y2, 
                                   z=z2,
                                   
                                   # Specify the colors to be used
                                   colorscale=colorscale,
                                   
                                   # Define the contours
                                   contours = {"x": {"show": True, "start": 
                                                     contour_x_start, 
                                                     "end": contour_x_stop, 
                                                     "size": contour_x_size, 
                                                     "color":"white"},            
                                               "y": {"show": True, "start": 
                                                     contour_y_start, 
                                                     "end": contour_y_stop, 
                                                     "size": contour_y_size, 
                                                     "color":"white"},  
                                               "z": {"show": True, "start": 
                                                     contour_z_start, 
                                                     "end": contour_z_stop, 
                                                     "size": contour_z_size}},
                                   
                                   # Set the surface opacity
                                   opacity=opacity),
                        
                        # Plot scatter of unsmoothed data 
                        go.Scatter3d(x=x,
                                     y=y,
                                     z=z,
                                     mode='markers',
                                     marker=dict(
                                         
                                         # Set size, color and opacity of 
                                         # each data point
                                         size=2,
                                         color='red', 
                                         opacity=0.9)
                                     )])
                
                # Plot just the scatter points
                else:
                    fig = go.Figure(data=[
                        # Plot scatter of unsmoothed data 
                        go.Scatter3d(x=x,
                                     y=y,
                                     z=z,
                                     mode='markers',
                                     marker=dict(
                                         
                                         # Set size, color and opacity of 
                                         # each data point
                                         size=2,
                                         color='red', 
                                         opacity=0.9)
                                     )])

            # Plot just the surface
            else:
                # Create figure object with fitted surface
                fig = go.Figure(data=[
                    go.Surface(x=x2, 
                               y=y2, 
                               z=z2,
                                
                               # Specify the colors to be used
                               colorscale=colorscale,
                                
                               # Define the contours
                               contours = {"x": {"show": True, "start": 
                                                 contour_x_start, 
                                                 "end": contour_x_stop, 
                                                 "size": contour_x_size, 
                                                 "color":"white"},            
                                           "y": {"show": True, "start": 
                                                 contour_y_start, 
                                                 "end": contour_y_stop, 
                                                 "size": contour_y_size, 
                                                 "color":"white"},  
                                           "z": {"show": True, "start": 
                                                 contour_z_start, 
                                                 "end": contour_z_stop, 
                                                 "size": contour_z_size}},
                                
                               # Set the surface opacity
                               opacity=opacity)])

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
                                yaxis_title='Strike',
                                zaxis_title='Implied Volatility %',),
                              # Specify title with ticker label, voltype 
                              # and date
                              title={'text':(str(self.ticker_label)+
                                             ' Implied Volatility '+
                                             str(voltype.title())+
                                             ' Price '+str(self.start_date)),
                                     'y':0.9,
                                     'x':0.5,
                                     'xanchor':'center',
                                     'yanchor':'top',
                                     'font':dict(size=20, 
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
        
        if save_image:
            # save the image as a png file
            self._image_save(fig=fig, image_folder=image_folder, 
                             image_dpi=image_dpi)

        # Set warnings back to default
        warnings.filterwarnings("default", category=UserWarning)
        

    def _graph_format(self, fig_size, azim, elev, voltype):
        
        # Update chart parameters        
        plt.rcParams.update(self.mpl_3D_params)
        
        # Create fig object
        fig = plt.figure(figsize=fig_size)
        
        # Create axes object
        ax = fig.add_subplot(111, projection='3d', azim=azim, 
                                  elev=elev)
              
        # Set background color to white
        ax.set_facecolor('w')

        # Create values that scale fonts with fig_size 
        ax_font_scale = int(round(fig_size[0] * 1.1))
        title_font_scale = int(round(fig_size[0] * 1.5))

        # Tint the axis panes, RGB values from 0-1 and alpha denoting 
        # color intensity
        ax.w_xaxis.set_pane_color((0.9, 0.8, 0.9, 0.8))
        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.9, 0.8))
        ax.w_zaxis.set_pane_color((0.9, 0.9, 0.8, 0.8))
        
        # Set z-axis to left hand side
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        
        # Set fontsize of axis ticks
        ax.tick_params(axis='both', which='major', labelsize=ax_font_scale)
        
        # Label axes
        ax.set_xlabel('Strike', fontsize=ax_font_scale, 
                      labelpad=ax_font_scale*1.2)
        ax.set_ylabel('Time to Expiration (Days)', fontsize=ax_font_scale, 
                      labelpad=ax_font_scale*1.2)
        ax.set_zlabel('Implied Volatility %', fontsize=ax_font_scale, 
                      labelpad=ax_font_scale*1.2)
        
        # Specify title with ticker label, voltype and date
        st = fig.suptitle(str(self.ticker_label)+
                          ' Implied Volatility '+
                          str(voltype.title())+
                          ' Price '+str(self.start_date), 
                          fontsize=title_font_scale, 
                          fontweight=0, 
                          color='black', 
                          style='italic', 
                          y=1.02)

        st.set_y(0.95)
        fig.subplots_adjust(top=1)
                
        return fig, ax
    
            
    def _smooth(self, order=None, voltype=None, smoothopt=None):
        """
        Create a column of smoothed implied vols

        Parameters
        ----------
        order : Int
            Polynomial order used in numpy polyfit function. The 
            default is 3.
        voltype : Str
            Whether to use 'bid', 'mid', 'ask' or 'last' price. The 
            default is 'last'.
        smoothopt : Int    
            Minimum number of options to fit curve to. The default 
            is 6.
        
        Returns
        -------
        DataFrame
            DataFrame of Option prices.

        """
        
        # If inputs are not supplied, take existing values
        order, voltype, smoothopt = itemgetter(
            'order', 'voltype', 'smoothopt')(self._refresh_params_default(
                order=order, voltype=voltype, smoothopt=smoothopt))
               
        # Create a dictionary of the number of options for each 
        # maturity
        mat_dict = dict(Counter(self.imp_vol_data['Days']))
        
        # Create a sorted list of the different number of days to 
        # maturity
        maturities = sorted(list(set(self.imp_vol_data['Days'])))
        
        # Create a sorted list of the different number of strikes
        strikes_full = sorted(list(set((self.imp_vol_data[
            'Strike'].astype(float)))))
        
        # create copy of implied vol data
        self.imp_vol_data_smoothed = self.imp_vol_data.copy()
        
        for ttm, count in mat_dict.items():
            
            # if there are less than smoothopt (default is 6) options 
            # for a given maturity
            if count < smoothopt:
                
                # remove that maturity from the maturities list
                maturities.remove(ttm)
                
                # and remove that maturity from the implied vol 
                # DataFrame
                self.imp_vol_data_smoothed = self.imp_vol_data_smoothed[
                    self.imp_vol_data_smoothed['Days'] != ttm]            
        
        # Create empty DataFrame with the full range of strikes as 
        # index
        self.smooth_surf = pd.DataFrame(index=strikes_full)
        
        # going through the maturity list (in reverse so the columns 
        # created are in increasing order)
        for maturity in reversed(maturities):
            
            # Extract the strikes for this maturity
            strikes = self.imp_vol_data[self.imp_vol_data[
                'Days']==maturity]['Strike']
            
            # And the vols (specifying the voltype)
            vols = self.imp_vol_data[self.imp_vol_data[
                'Days']==maturity][str(self.vols_dict[str(self.voltype)])]
            
            # Fit a polynomial to this data
            curve_fit = np.polyfit(strikes, vols, order)
            p = np.poly1d(curve_fit)
            
            # Create empty list to store smoothed implied vols
            iv_new = []
            
            # For each strike
            for strike in strikes_full:
                
                # Add the smoothed value to the iv_new list 
                iv_new.append(p(strike))
            
            # Append this list as a new column in the smooth_surf 
            # DataFrame    
            self.smooth_surf.insert(0, str(maturity), iv_new) 
    
        # Apply the _vol_map function to add smoothed vol column to 
        # DataFrame
        self.imp_vol_data_smoothed = self.imp_vol_data_smoothed.apply(
            lambda x: self._vol_map(x), axis=1)

        return self


    def _vol_map(self, row):
        """
        Map value calculated in smooth surface DataFrame to 
        'Smoothed Vol' column.

        Parameters
        ----------
        row : Array
            Each row in the DataFrame.

        Returns
        -------
        row : Array
            Each row in the DataFrame.

        """
        row['Smoothed Vol'] = self.smooth_surf.loc[row['Strike'], 
                                                   str(row['Days'])]
        
        return row
       
        
    def _image_save(self, fig, image_folder, image_dpi):
        
        # Create image folder if it does not already exist
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
            
        # save the image as a png file    
        plt.savefig('{}/{}{}.png'.format(
            image_folder, self.ticker_label, self.start_date), 
            dpi=image_dpi)