"""
Market data import and transformation functions

"""
import calendar
from collections import Counter
import copy
from datetime import datetime, date, timedelta
import time
from urllib.request import FancyURLopener
import warnings
import datetime as dt
from bs4 import BeautifulSoup
from lxml import html
import numpy as np
import pandas as pd
from pandas.tseries.holiday import get_calendar, HolidayCalendarFactory, GoodFriday
import pytz
from volvisualizer.utils import ImpliedVol
# pylint: disable=invalid-name


# Class used to open urls for financial data
class UrlOpener(FancyURLopener):
    """
    Extract data from Yahoo Finance URL

    """
    version = 'w3m/0.5.3+git20180125'


class Data():
    """
    Market data import and transformation functions

    """
    @classmethod
    def create_option_data(cls, params, tables):
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
        ticker_label : Str
            The ticker label used in charts.
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
        strike_limits : Tuple
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
        params, tables = cls.extractoptions(params=params, tables=tables)
        print("Options data extracted")

        # Filter / transform data
        params, tables = cls.transform(params=params, tables=tables)
        print("Data transformed")

        # Calculate implied volatilities and combine
        params, tables = cls.combine(params=params, tables=tables)
        print("Data combined")

        return params, tables


    @classmethod
    def extractoptions(cls, params, tables):
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

        # Extract dictionary of option dates and urls
        params = cls._extracturls(params=params)
        print("URL's extracted")

        params['raw_web_data'] = cls._extract_web_data(params=params)

        params = cls._read_web_data(params=params)

        # Create an empty DataFrame
        tables['full_data'] = pd.DataFrame()

        # Make a list of all the dates of the DataFrames just stored
        # in the default dictionary
        params['date_list'] = list(params['option_dict'].keys())

        params, tables = cls._process_options(params=params, tables=tables)

        return params, tables


    @staticmethod
    def _extracturls(params):
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

        # Define the stock root webpage
        url = 'https://finance.yahoo.com/quote/'+params['ticker']\
            +'/options?p='+params['ticker']

        # Create a UrlOpener object to extract data from the url
        urlopener = UrlOpener()
        response = urlopener.open(url)

        # Collect the text from this object
        params['html_doc'] = response.read()

        # Use Beautiful Soup to parse this
        soup = BeautifulSoup(params['html_doc'], features="lxml")

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
        params['url_dict'] = {}

        # For each date and unixdate in the first dictionary
        for date_val, page in optodict.items():

            # Create an entry with the date as key and the url plus
            # unix date as value
            params['url_dict'][date_val] = str(
                'https://finance.yahoo.com/quote/'
                +params['ticker']+'/options?date='+page)

        return params



    @staticmethod
    def _extract_web_data(params):

        raw_web_data = {}

        # each url needs to have an option expiry date associated with
        # it in the url dict
        for input_date, url in params['url_dict'].items():

            # UrlOpener function downloads the data
            urlopener = UrlOpener()
            weburl = urlopener.open(url)
            raw_web_data[input_date] = weburl.read()

            # wait between each query so as not to overload server
            time.sleep(params['wait'])

        return raw_web_data


    @staticmethod
    def _read_web_data(params):

        # Create an empty dictionary
        params['option_dict'] = {}
        params['url_except_dict'] = {}

        for input_date, url in params['url_dict'].items():
            # if data exists
            try:
                # read html data into a DataFrame
                option_frame = pd.read_html(params['raw_web_data'][input_date])

                # Add this DataFrame to the default dictionary, named
                # with the expiry date it refers to
                params['option_dict'][input_date] = option_frame

            # otherwise collect dictionary of exceptions
            except ValueError:
                params['url_except_dict'][input_date] = url

        return params


    @staticmethod
    def _process_options(params, tables):

        # Create list to store exceptions
        params['opt_except_list'] = []

        # For each of these dates
        for input_date in params['date_list']:

            try:
                # The first entry is 'calls'
                calls = params['option_dict'][input_date][0]

                # Create a column designating these as calls
                calls['Option Type'] = 'call'

                try:
                    # The second entry is 'puts'
                    puts = params['option_dict'][input_date][1]

                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'

                    # Concatenate these two DataFrames
                    options = pd.concat([calls, puts])

                    # Add an 'Expiry' column with the expiry date
                    options['Expiry'] = pd.to_datetime(input_date).date()

                    # Add this DataFrame to 'full_data'
                    tables['full_data'] = pd.concat(
                        [tables['full_data'], options])

                except IndexError:

                    # Add an 'Expiry' column with the expiry date
                    calls['Expiry'] = pd.to_datetime(input_date).date()

                    # Add this DataFrame to 'full_data'
                    tables['full_data'] = pd.concat(
                        [tables['full_data'], calls])

            except IndexError:

                try:
                    # The second entry is 'puts'
                    puts = params['option_dict'][input_date][1]

                    # Create a column designating these as puts
                    puts['Option Type'] = 'put'

                    # Add an 'Expiry' column with the expiry date
                    puts['Expiry'] = pd.to_datetime(input_date).date()

                    # Add this DataFrame to 'full_data'
                    tables['full_data'] = pd.concat(
                        [tables['full_data'], puts])

                except IndexError:
                    params['opt_except_list'].append(input_date)

        return params, tables


    @classmethod
    def transform(cls, params, tables):
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

        # Make a copy of 'full_data'
        tables['data'] = copy.deepcopy(tables['full_data'])

        # Set timezone
        est = pytz.timezone('US/Eastern')

        # Convert 'Last Trade Date' to a DateTime variable
        tables['data']['Last Trade Date Raw'] = (
            tables['data']['Last Trade Date'])

        # Format date based on Eastern Daylight or Standard Time
        try:
            tables['data']['Last Trade Date'] = pd.to_datetime(
                tables['data']['Last Trade Date'],
                format='%Y-%m-%d %I:%M%p EDT')

        except ValueError:
            tables['data']['Last Trade Date'] = pd.to_datetime(
                tables['data']['Last Trade Date'],
                format='%Y-%m-%d %I:%M%p EST')

        tables['data']['Last Trade Date'] = (
            tables['data']['Last Trade Date'].apply(
                lambda x: x.replace(tzinfo=est)))

        # Create columns of expiry date as datetime object and str
        tables['data']['Expiry_datetime'] = pd.to_datetime(
            tables['data']['Expiry'], format='%Y-%m-%d')
        tables['data']['Expiry_str'] = (
            tables['data']['Expiry_datetime'].dt.strftime('%Y-%m-%d'))

        # Filter data from start date
        tables['data'] = (
            tables['data'][tables['data']['Last Trade Date']>=str(
                pd.to_datetime(params['start_date']))])

        tables = cls._trade_columns(tables=tables)

        # Create Time To Maturity (in years) column
        tables['data']['TTM'] = (
            pd.to_datetime(tables['data']['Expiry'])
            - pd.to_datetime(date.today())) / (pd.Timedelta(days=1) * 365)

        # Create Days to Maturity column
        tables['data']['Days'] = np.round(tables['data']['TTM']*365, 0)

        params, tables = cls._filters(params=params, tables=tables)

        return params, tables


    @staticmethod
    def _trade_columns(tables):

        # Create a column of the Trade Day
        tables['data']['Last Trade Day'] = (
            tables['data']['Last Trade Date'].dt.date)

        # Create a column of the Trade Time of Day
        tables['data']['Last Trade Time'] = (
            tables['data']['Last Trade Date'].dt.time)

        # Create a column of the Trade Hour of Day
        tables['data']['Last Trade Hour'] = (
            tables['data']['Last Trade Date'].dt.hour)

        # Create a column of the Trade Date represented in unixtime
        tables['data']['Unixtime'] = (
            tables['data']['Last Trade Date'].view(np.int64) // 10**9)

        # Clean Volume column
        tables['data']['Volume'] = (
            tables['data']['Volume'].replace('-',0).astype(int))

        # Clean Open Interest column
        tables['data']['Open Interest'] = (
            tables['data']['Open Interest'].replace('-',0).astype(int))

        # Clean Ask column
        tables['data']['Ask'] = (
            tables['data']['Ask'].replace('-',0).astype(float))

        # Clean Bid column
        tables['data']['Bid'] = (
            tables['data']['Bid'].replace('-',0).astype(float))

        # Create Mid column
        tables['data']['Mid'] = (
            tables['data']['Ask'] + tables['data']['Bid']) / 2

        return tables


    @classmethod
    def _filters(cls, params, tables):

        # If a minutes parameter is supplied, filter for most recent
        # minutes
        if params['lastmins'] is not None:
            tables['data'] = (tables['data'][tables['data']['Unixtime']  >= (
                max(tables['data']['Unixtime']) - params['lastmins'] * 60)])

        # If a mindays parameter is supplied, filter for option expiry
        # greater than parameter
        if params['mindays'] is not None:
            tables['data'] = (
                tables['data'][tables['data']['Days']  >= params['mindays']])

        # If a minopts parameter is supplied, filter for volume greater
        # than parameter
        if params['minopts'] is not None:

            # Create a dictionary of the number of options for each
            # maturity
            mat_dict = dict(Counter(tables['data']['Days']))
            for ttm, count in mat_dict.items():

                # if there are less than minopts options for a given
                # maturity
                if count < params['minopts']:

                    # remove that maturity from the DataFrame
                    tables['data'] = (
                        tables['data'][tables['data']['Days'] != ttm])

        # If a volume parameter is supplied, filter for volume greater
        # than parameter
        if params['volume'] is not None:
            tables['data'] = (
                tables['data'][tables['data']['Volume']  >= params['volume']])

        # If an openint parameter is supplied, filter for Open Interest
        # greater than parameter
        if params['openint'] is not None:
            tables['data'] = (
                tables['data'][tables['data']['Open Interest']
                               >= params['openint']])

        params, tables = cls._monthlies(params=params, tables=tables)

        return params, tables


    @staticmethod
    def _monthlies(params, tables):

        # If the monthlies flag is set
        if params['monthlies'] is True:

            # Create an empty list
            date_list = []

            # For each date in the url_dict
            for key in params['url_dict'].keys():

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

                if expiry in params['trade_holidays']:
                    expiry = expiry - timedelta(days=1)

                # Calculate the number of days until that expiry
                ttm = (expiry - dt.date.today()).days

                # Append this to the days_list
                days_list.append(ttm)

            # For each unique number of days to expiry
            for days_to_expiry in set(tables['data']['Days']):

                # if the expiry is not in the list of monthly expiries
                if days_to_expiry not in days_list:

                    # Remove that expiry from the DataFrame
                    tables['data'] = (
                        tables['data'][tables['data']['Days']
                                       != days_to_expiry])

        return params, tables


    @classmethod
    def combine(cls, params, tables):
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

        # create copy of filtered data
        input_data = copy.deepcopy(tables['data'])

        # Calculate strikes if strikes and spot price are not supplied.
        params = cls._create_strike_range(params=params, tables=tables)

        # Assign ticker label to the object
        if params['ticker_label'] is None:
            params['ticker_label'] = params['ticker'].lstrip('^')

        # Create empty list and dictionary for storing options
        params['opt_list'] = []
        opt_dict = {}

        # For each put strike price
        for strike in params['put_strikes']:

            # Assign an option name of ticker plus strike
            opt_name = params['ticker_label']+'_'+str(strike)

            # store the implied vol results for that strike in the
            # option dictionary
            opt_dict[opt_name] = cls._imp_vol_apply(
                params=params, input_data=input_data, strike=strike,
                option='put')

            # store the implied vol results for that strike in the
            # option list
            params['opt_list'].append(opt_dict[opt_name])

            print('Put option: ', opt_name)

        # For each put strike price
        for strike in params['call_strikes']:

            # Assign an option name of ticker plus strike
            opt_name = params['ticker_label']+'_'+str(strike)

            # store the implied vol results DataFrame for that strike
            # in the option dictionary
            opt_dict[opt_name] = cls._imp_vol_apply(
                params=params, input_data=input_data, strike=strike,
                option='call')

            # store the implied vol results DataFrame for that strike
            # in the option list
            params['opt_list'].append(opt_dict[opt_name])

            print('Call option: ', opt_name)

        # Concatenate all the option results into a single DataFrame and drop
        # any null values
        tables['imp_vol_data'] = pd.concat(params['opt_list']).dropna()

        return params, tables


    @classmethod
    def _create_strike_range(cls, params, tables):

        # Extract the spot level from the html data
        if params['spot'] is None:
            tree = html.fromstring(params['html_doc'])
            priceparse = tree.xpath(
                '//fin-streamer[@class="Fw(b) Fz(36px) Mb(-4px) D(ib)"]/text()')
            params['spot'] = float(
                [str(p) for p in priceparse][0].replace(',',''))

        # Calculate initial spot, min and max strikes to use in divisor
        # calculation
        params['init_roundspot'], params['init_put_min'], \
            params['init_call_max'] = cls._strike_filters(
                params=params, divisor=10)

        # If a divisor value is not provided
        if params['divisor'] is None:
            params = cls._create_divisor(params=params, tables=tables)

        # Calculate final spot, min and max strikes using chosen divisor
        params['roundspot'], params['put_min'], \
            params['call_max'] = cls._strike_filters(
                params=params, divisor=params['divisor'])

        # Calculate put options (default is 1/2 spot level)
        if params['put_strikes'] is None:
            params['put_strikes'] = list(
                np.linspace(
                    params['put_min'],
                    params['roundspot'],
                    int((params['roundspot'] - params['put_min'])
                        / params['divisor']) + 1))

        # Calculate call options (default is twice the spot level)
        if params['call_strikes'] is None:
            params['call_strikes'] = list(
                np.linspace(
                    params['roundspot'],
                    params['call_max'],
                    int((params['call_max'] - params['roundspot'])
                        / params['divisor']) + 1))

        return params


    @staticmethod
    def _strike_filters(params, divisor):

        # Calculate the point to switch from put to call options
        roundspot = (
            round(params['spot'] / divisor) * divisor)

        put_min = (
            round(params['spot'] * params['strike_limits'][0] / divisor)
            * divisor)

        call_max = (
            round(params['spot'] * params['strike_limits'][1] / divisor)
            * divisor)

        return roundspot, put_min, call_max


    @staticmethod
    def _create_divisor(params, tables):

        # Take the set of all the option strikes in the data
        strikes = set(tables['data']['Strike'])

        # Find the number of options with a remainder of zero for each of the
        # listed potential divisors
        avail_strikes = {}
        for div in [0.5, 1, 1.25, 2.5, 5, 10, 25, 50, 100]:
            avail_strikes[div] = len(
                {x for x in strikes if (
                    x%(div) == 0
                    and params['init_put_min'] < x < params['init_call_max'])})

        # Find the maximum divisor of those with the highest number of options
        params['divisor'] = max(
            [divisor for max_strike_count in [max(avail_strikes.values())]
             for divisor, strike_count in avail_strikes.items()
             if strike_count == max_strike_count])

        return params


    @classmethod
    def _imp_vol_apply(cls, params, input_data, strike, option):
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
        input_data = (input_data[(input_data['Strike'] == strike)
                & (input_data['Option Type'] == option)])

        # Apply implied vol method to each row
        input_data = input_data.apply(
            lambda x: cls._imp_vol_by_row(x, params, strike, option), axis=1)

        return input_data


    @staticmethod
    def _imp_vol_by_row(row, params, strike, option):
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


        # Select the chosen implied vol method from the dictionary
        func_name = params['method_dict'][params['method']]

        # for each of the prices: bid, mid, ask, last
        for input_row, output_row in params['row_dict'].items():

            opt_params = {
                'S':params['spot'],
                'K':strike,
                'T':row['TTM'],
                'r':params['r'],
                'q':params['q'],
                'cm':row[input_row],
                'epsilon':params['epsilon'],
                'option':option
                }
            try:
                # populate the column using the chosen implied
                # vol method (using getattr() to select
                # dynamically)
                # check if n/a value is returned and print error
                # message if so
                output = getattr(
                    ImpliedVol, func_name)(opt_params=opt_params)

                output = float(output)
                row[output_row] = output

            except KeyError:
                print("Key Error with option: Strike="+str(strike)+
                          " TTM="+str(round(row['TTM'], 3))+
                          " vol="+str(row[input_row])+
                          " option="+option)

            except ValueError:
                print("Value Error with option: Strike="+str(strike)+
                          " TTM="+str(round(row['TTM'], 3))+
                          " vol="+str(row[input_row])+
                          " option="+option)

        # Return warnings to default setting
        warnings.filterwarnings("default", category=RuntimeWarning)

        return row


    @staticmethod
    def trading_calendar(params):
        """
        Generate list of trading holidays

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.

        """
        # Create calendar instance
        cal = get_calendar('USFederalHolidayCalendar')
        cal_mod = copy.deepcopy(cal)

        start = date.today()
        end = start + timedelta(days=2500)

        # Remove Columbus Day rule and Veteran's Day rule
        cal_mod.rules.pop(7)
        cal_mod.rules.pop(6)

        # Create new calendar generator
        tradingCal = HolidayCalendarFactory(
            'TradingCalendar', cal_mod, GoodFriday)

        tcal = tradingCal()

        holiday_array = tcal.holidays(start=start, end=end).to_pydatetime()

        params['trade_holidays'] = []
        for hol in holiday_array:
            params['trade_holidays'].append(hol.date())

        return params
