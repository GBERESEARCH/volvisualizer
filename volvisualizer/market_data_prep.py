"""
Market data transformation and combination functions

"""
import calendar
from collections import Counter
import copy
import datetime as dt
from datetime import date, timedelta
import random
import warnings
from lxml import html
import numpy as np
import pandas as pd
import pytz
import requests
from scipy import interpolate
from volvisualizer.vol_methods import ImpliedVol
from volvisualizer.volatility_params import USER_AGENTS

warnings.filterwarnings("ignore", category=DeprecationWarning)
# pylint: disable=invalid-name


class UrlOpener:
    """
    Extract data from Yahoo Finance URL

    """
    request_headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-US,en;q=0.9",
        "origin": "https://finance.yahoo.com",
        "referer": "https://finance.yahoo.com",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
    }
    user_agent = random.choice(USER_AGENTS)
    request_headers["User-Agent"] = user_agent

    def __init__(self):
        self._session = requests

    def open(self, url):
        """
        Extract data from Yahoo Finance URL

        Parameters
        ----------
        url : Str
            The URL to extract data from.

        Returns
        -------
        response : Response object
            Response object of requests module.

        """
        response = self._session.get(url=url, headers=self.request_headers)
        
        return response


class DataPrep():
    """
    Market data transformation and combination functions

    """
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

        # If a minopts parameter is supplied, filter for number of options
        # greater than parameter
        if params['minopts'] is not None:

            tables['data'] = cls._minopts(params=params, data=tables['data'])

        params, tables = cls._monthlies(params=params, tables=tables)

        return params, tables


    @staticmethod
    def _minopts(params, data):
        # Create a dictionary of the number of options for each
        # maturity
        mat_dict = dict(Counter(data['Days']))
        for ttm, count in mat_dict.items():

            # if there are less than minopts options for a given
            # maturity
            if count < params['minopts']:

                # remove that maturity from the DataFrame
                data = (
                    data[data['Days'] != ttm])

        return data


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

        # If a minopts parameter is supplied, filter for number of options
        # greater than parameter
        if params['minopts'] is not None:
            tables['imp_vol_data'] = cls._minopts(
                params=params, data=tables['imp_vol_data'])

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


    @classmethod
    def _imp_vol_by_row(cls, row, params, strike, option):
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
                'r':cls.interest_rate(row['Days'], params['yield_curve']),
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


    @classmethod
    def interest_rate(cls, ttm, yield_curve=None):
        """
        Returns the interest rate for a given number of days to maturity

        Parameters
        ----------
        ttm : Int
            Number of days to maturity.
        yield_curve : Scipy Interpolate object, optional
            The yield curve interpolation function. The default is None in which
            case it will be generated.

        Returns
        -------
        Float
            The interest rate for the given TTM.

        """
        if yield_curve is None:
            yield_curve = cls.generate_yield_curve()

        return np.round(float(yield_curve(ttm))/100, 5)


    @staticmethod
    def generate_yield_curve(r=None):
        """
        Returns a yield curve interpolation function

        Parameters
        ----------
        r : Float, optional
            If a single interest rate is supplied, the method will calculate a
            flat curve. The default is None.

        Returns
        -------
        yield_curve : Scipy Interpolate object
            The yield curve interpolation function.

        """

        if r is None:

            year = dt.date.today().strftime("%Y")

            # Extract Daily Treasury Par Yield Curve Rates
            url = 'https://home.treasury.gov/resource-center/data-chart-center'+\
                '/interest-rates/TextView?type=daily_treasury_yield_curve'+\
                    '&field_tdr_date_value='+year
            data = pd.read_html(url)[0]

            # Dictionary mapping tenors to days
            ir_tenor_dict = {
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
                }

            # Take the last row as most recent data and convert to Pandas Series
            current_rate = data.iloc[-1]
            current_rate = current_rate[list(ir_tenor_dict.keys())].squeeze()

            # Create lists of tenors and interest rates, adding a point at 1 day equal
            # to the 30 day point to allow interpolation of those dates
            tenors = []
            tenors.append(1)
            for bucket in current_rate.index:
                tenors.append(ir_tenor_dict[bucket])

            rates = []
            for rate in current_rate:
                if not rates:
                    rates.append(rate)
                    rates.append(rate)
                else:
                    rates.append(rate)

        else:
            tenors = [1, 10952]
            rates = [r, r]

        # Create a curve using cubic spline interpolation
        yield_curve = interpolate.interp1d(tenors, rates, kind='cubic')

        return yield_curve


    @classmethod
    def dividend_yield(cls, ticker):
        """
        Returns the dividend yield for a given ticker

        Parameters
        ----------
        ticker : Str
            The underlying to calculate dividend yield for.

        Returns
        -------
        Float
            The dividend yield for the given ticker.

        """

        # If the ticker is not valid return zero
        if ticker == '^SPX':
            try:
                div_yield = cls._spx_div_yield()
            except ValueError:
                print("No dividend data for SPX")
                div_yield = '0.0%'
        else:
            try:
                div_yield = cls._stock_dividend_yield(ticker)
            except ValueError:
                print("No dividend data for "+ticker)
                div_yield = '0.0%'

        try: 
            result = np.round(float(div_yield.rstrip('%'))/100, 5)
        except ValueError:
            print("No valid dividend data for "+ticker)
            result = 0.0

        return result


    @staticmethod
    def _stock_dividend_yield(ticker):

        url = 'https://stockanalysis.com/stocks/'+ticker+'/dividend/'

        r = requests.get(url)

        html_doc = r.text

        tree = html.fromstring(html_doc)
        
        parse = tree.xpath("//*[contains(text(), 'Dividend Yield')]/div/text()")

        return [str(p) for p in parse][0].replace('\n','')


    @staticmethod
    def _spx_div_yield():

        url = 'https://www.multpl.com/s-p-500-dividend-yield'

        urlopener = UrlOpener()
        response = urlopener.open(url)

        html_doc = response.text

        tree = html.fromstring(html_doc)

        parse = tree.xpath(
            '//div[@id="current"]/text()')

        return [str(p) for p in parse][1].replace('\n','')
