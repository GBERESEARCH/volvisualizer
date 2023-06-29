"""
Methods for extracting Implied Vol and producing skew reports

"""
from collections import Counter
import copy
from decimal import Decimal
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import scipy as sp
import scipy.interpolate as inter
import scipy.stats as si
# pylint: disable=invalid-name, consider-using-f-string

class ImpliedVol():
    """
    Implied Volatility Extraction methods

    """
    @classmethod
    def implied_vol_newton_raphson(cls, opt_params: dict) -> float | str:
        """
        Finds implied volatility using Newton-Raphson method - needs
        knowledge of partial derivative of option pricing formula
        with respect to volatility (vega)

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        cm : Float
            # Option price used to solve for vol. The default is 5.
        epsilon : Float
            Degree of precision. The default is 0.0001
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        default : Bool
            Whether the function is being called directly (in which
            case values that are not supplied are set to default
            values) or called from another function where they have
            already been updated.

        Returns
        -------
        result : Float
            Implied Volatility.

        """

        # Manaster and Koehler seed value
        opt_params['vi'] = np.sqrt(
            abs(np.log(opt_params['S'] / opt_params['K'])
                + opt_params['r'] * opt_params['T']) * (2 / opt_params['T']))

        opt_params['ci'] = cls.black_scholes_merton(
            opt_params=opt_params, sigma=opt_params['vi'])

        opt_params['vegai'] = cls.black_scholes_merton_vega(
            opt_params=opt_params, sigma=opt_params['vi'])

        opt_params['mindiff'] = abs(opt_params['cm'] - opt_params['ci'])

        while (abs(opt_params['cm'] - opt_params['ci'])
               >= opt_params['epsilon']
               and abs(opt_params['cm'] - opt_params['ci'])
               <= opt_params['mindiff']):

            opt_params['vi'] = (
                opt_params['vi']
                - (opt_params['ci'] - opt_params['cm']) / opt_params['vegai'])

            opt_params['ci'] = cls.black_scholes_merton(
                opt_params=opt_params, sigma=opt_params['vi'])

            opt_params['vegai'] = cls.black_scholes_merton_vega(
                opt_params=opt_params, sigma=opt_params['vi'])

            opt_params['mindiff'] = abs(opt_params['cm'] - opt_params['ci'])

        if abs(opt_params['cm'] - opt_params['ci']) < opt_params['epsilon']:
            result = opt_params['vi']
        else:
            result = 'NA'

        return result


    @classmethod
    def implied_vol_bisection(cls, opt_params: dict) -> float | str:
        """
        Finds implied volatility using bisection method.

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        cm : Float
            # Option price used to solve for vol. The default is 5.
        epsilon : Float
            Degree of precision. The default is 0.0001
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        default : Bool
            Whether the function is being called directly (in which
            case values that are not supplied are set to default
            values) or called from another function where they have
            already been updated.

        Returns
        -------
        result : Float
            Implied Volatility.

        """

        opt_params['vLow'] = 0.005
        opt_params['vHigh'] = 4
        opt_params['cLow'] = cls.black_scholes_merton(
            opt_params=opt_params, sigma=opt_params['vLow'])

        opt_params['cHigh'] = cls.black_scholes_merton(
            opt_params=opt_params, sigma=opt_params['vHigh'])

        counter = 0

        opt_params['vi'] = (
            opt_params['vLow']
            + (opt_params['cm'] - opt_params['cLow'])
            * (opt_params['vHigh'] - opt_params['vLow'])
            / (opt_params['cHigh'] - opt_params['cLow']))

        while abs(opt_params['cm'] - cls.black_scholes_merton(
                opt_params=opt_params,
                sigma=opt_params['vi'])) > opt_params['epsilon']:

            counter = counter + 1
            if counter == 100:
                result = 'NA'

            if cls.black_scholes_merton(
                    opt_params=opt_params,
                    sigma=opt_params['vi']) < opt_params['cm']:
                opt_params['vLow'] = opt_params['vi']

            else:
                opt_params['vHigh'] = opt_params['vi']

            opt_params['cLow'] = cls.black_scholes_merton(
                opt_params=opt_params, sigma=opt_params['vLow'])

            opt_params['cHigh'] = cls.black_scholes_merton(
                opt_params=opt_params, sigma=opt_params['vHigh'])

            opt_params['vi'] = (
                opt_params['vLow']
                + (opt_params['cm'] - opt_params['cLow'])
                * (opt_params['vHigh'] - opt_params['vLow'])
                / (opt_params['cHigh'] - opt_params['cLow']))

        result = opt_params['vi']

        return result


    @classmethod
    def implied_vol_naive(cls, opt_params: dict) -> float:
        """
        Finds implied volatility using simple naive iteration,
        increasing precision each time the difference changes sign.

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        cm : Float
            # Option price used to solve for vol. The default is 5.
        epsilon : Float
            Degree of precision. The default is 0.0001
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        result : Float
            Implied Volatility.

        """

        # Seed vol
        opt_params['vi'] = 0.2

        # Calculate starting option price using this vol
        opt_params['ci'] = cls.black_scholes_merton(
            opt_params=opt_params, sigma=opt_params['vi'])

        # Initial price difference
        opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']

        if opt_params['price_diff'] > 0:
            opt_params['flag'] = 1

        else:
            opt_params['flag'] = -1

        # Starting vol shift size
        opt_params['shift'] = 0.01

        opt_params['price_diff_start'] = opt_params['price_diff']

        while abs(opt_params['price_diff']) > opt_params['epsilon']:

            # If the price difference changes sign after the vol shift,
            # reduce the decimal by one and reverse the sign
            if (np.sign(opt_params['price_diff'])
                != np.sign(opt_params['price_diff_start'])):
                opt_params['shift'] = opt_params['shift'] * -0.1

            # Calculate new vol
            opt_params['vi'] += (opt_params['shift'] * opt_params['flag'])

            # Set initial price difference
            opt_params['price_diff_start'] = opt_params['price_diff']

            # Calculate the option price with new vol
            opt_params['ci'] = cls.black_scholes_merton(
                opt_params=opt_params, sigma=opt_params['vi'])

            # Price difference after shifting vol
            opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']

            # If values are diverging reverse the shift sign
            if (abs(opt_params['price_diff'])
                > abs(opt_params['price_diff_start'])):
                opt_params['shift'] = -opt_params['shift']

        result = opt_params['vi']

        return result


    @classmethod
    def implied_vol_naive_verbose(cls, opt_params: dict) -> float:
        """
        Finds implied volatility using simple naive iteration,
        increasing precision each time the difference changes sign.

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        cm : Float
            # Option price used to solve for vol. The default is 5.
        epsilon : Float
            Degree of precision. The default is 0.0001
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        result : Float
            Implied Volatility.

        """

        opt_params['vi'] = 0.2
        opt_params['ci'] = cls.black_scholes_merton(
            opt_params=opt_params, sigma=opt_params['vi'])

        opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
        if opt_params['price_diff'] > 0:
            opt_params['flag'] = 1
        else:
            opt_params['flag'] = -1
        while abs(opt_params['price_diff']) > opt_params['epsilon']:
            while opt_params['price_diff'] * opt_params['flag'] > 0:
                opt_params['ci'] = cls.black_scholes_merton(
                    opt_params=opt_params, sigma=opt_params['vi'])

                opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
                opt_params['vi'] += (0.01 * opt_params['flag'])

            while opt_params['price_diff'] * opt_params['flag'] < 0:
                opt_params['ci'] = cls.black_scholes_merton(
                    opt_params=opt_params, sigma=opt_params['vi'])

                opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
                opt_params['vi'] -= (0.001 * opt_params['flag'])

            while opt_params['price_diff'] * opt_params['flag'] > 0:
                opt_params['ci'] = cls.black_scholes_merton(
                    opt_params=opt_params, sigma=opt_params['vi'])

                opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
                opt_params['vi'] += (0.0001 * opt_params['flag'])

            while opt_params['price_diff'] * opt_params['flag'] < 0:
                opt_params['ci'] = cls.black_scholes_merton(
                    opt_params=opt_params, sigma=opt_params['vi'])

                opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
                opt_params['vi'] -= (0.00001 * opt_params['flag'])

            while opt_params['price_diff'] * opt_params['flag'] > 0:
                opt_params['ci'] = cls.black_scholes_merton(
                    opt_params=opt_params, sigma=opt_params['vi'])

                opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
                opt_params['vi'] += (0.000001 * opt_params['flag'])

            while opt_params['price_diff'] * opt_params['flag'] < 0:
                opt_params['ci'] = cls.black_scholes_merton(
                    opt_params=opt_params, sigma=opt_params['vi'])

                opt_params['price_diff'] = opt_params['cm'] - opt_params['ci']
                opt_params['vi'] -= (0.0000001 * opt_params['flag'])

        result = opt_params['vi']

        return result


    @staticmethod
    def black_scholes_merton(
        opt_params: dict,
        sigma: float) -> float:
        """
        Black-Scholes-Merton Option price

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        sigma : Float
            Implied Volatility.  The default is 0.2 (20%).
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        opt_price : Float
            Option Price.

        """

        opt_params['b'] = opt_params['r'] - opt_params['q']
        opt_params['carry'] = np.exp(
            (opt_params['b'] - opt_params['r']) * opt_params['T'])
        opt_params['d1'] = (
            (np.log(opt_params['S'] / opt_params['K'])
             + (opt_params['b'] + (0.5 * sigma ** 2)) * opt_params['T'])
              / (sigma * np.sqrt(opt_params['T'])))
        opt_params['d2'] = (
            (np.log(opt_params['S'] / opt_params['K'])
             + (opt_params['b'] - (0.5 * sigma ** 2)) * opt_params['T'])
              / (sigma * np.sqrt(opt_params['T'])))

        # Cumulative normal distribution function
        opt_params['Nd1'] = si.norm.cdf(opt_params['d1'], 0.0, 1.0)
        opt_params['minusNd1'] = si.norm.cdf(-opt_params['d1'], 0.0, 1.0)
        opt_params['Nd2'] = si.norm.cdf(opt_params['d2'], 0.0, 1.0)
        opt_params['minusNd2'] = si.norm.cdf(-opt_params['d2'], 0.0, 1.0)

        if opt_params['option'] == "call":
            opt_price = (
                (opt_params['S'] * opt_params['carry'] * opt_params['Nd1'])
                - (opt_params['K']
                   * np.exp(-opt_params['r'] * opt_params['T'])
                   * opt_params['Nd2']))

        elif opt_params['option'] == 'put':
            opt_price = (
                (opt_params['K']
                 * np.exp(-opt_params['r'] * opt_params['T'])
                 * opt_params['minusNd2'])
                - (opt_params['S']
                   * opt_params['carry']
                   * opt_params['minusNd1']))

        else:
            print("Please supply a value for option - 'put' or 'call'")

        return opt_price


    @staticmethod
    def black_scholes_merton_vega(
        opt_params: dict,
        sigma: float) -> float:
        """
        Black-Scholes-Merton Option Vega

        Parameters
        ----------
        S : Float
            Stock Price. The default is 100.
        K : Float
            Strike Price. The default is 100.
        T : Float
            Time to Maturity.  The default is 0.25 (3 Months).
        r : Float
            Interest Rate. The default is 0.005 (50bps)
        q : Float
            Dividend Yield.  The default is 0.
        sigma : Float
            Implied Volatility.  The default is 0.2 (20%).

        Returns
        -------
        opt_vega : Float
            Option Vega.

        """

        opt_params['b'] = opt_params['r'] - opt_params['q']
        opt_params['carry'] = np.exp(
            (opt_params['b'] - opt_params['r']) * opt_params['T'])
        opt_params['d1'] = (
            (np.log(opt_params['S'] / opt_params['K'])
             + (opt_params['b'] + (0.5 * sigma ** 2)) * opt_params['T'])
              / (sigma * np.sqrt(opt_params['T'])))
        opt_params['nd1'] = (
            1 / np.sqrt(2 * np.pi)) * (np.exp(-opt_params['d1'] ** 2 * 0.5))

        opt_vega = (opt_params['S']
                    * opt_params['carry']
                    * opt_params['nd1']
                    * np.sqrt(opt_params['T']))

        return opt_vega


class VolMethods():
    """
    Methods for extracting Implied Vol and producing skew reports

    """
    @classmethod
    def smooth(
        cls,
        params: dict,
        tables: dict) -> tuple[dict, dict]:
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

        # Create a dictionary of the number of options for each
        # maturity
        mat_dict = dict(Counter(tables['imp_vol_data']['Days']))

        # Create a sorted list of the different number of days to
        # maturity
        maturities = sorted(list(set(tables['imp_vol_data']['Days'])))

        # Create a sorted list of the different number of strikes
        strikes_full = sorted(list(set((tables['imp_vol_data'][
            'Strike'].astype(float)))))

        # create copy of implied vol data
        tables['imp_vol_data_smoothed'] = copy.deepcopy(tables['imp_vol_data'])

        for ttm, count in mat_dict.items():

            # if there are less than smoothopt (default is 6) options
            # for a given maturity
            if count < params['smoothopt']:

                # remove that maturity from the maturities list
                maturities.remove(ttm)

                # and remove that maturity from the implied vol
                # DataFrame
                tables['imp_vol_data_smoothed'] = tables[
                    'imp_vol_data_smoothed'][
                        tables['imp_vol_data_smoothed']['Days'] != ttm]

        # Create empty DataFrame with the full range of strikes as
        # index
        tables['smooth_surf'] = pd.DataFrame(index=strikes_full)

        # going through the maturity list (in reverse so the columns
        # created are in increasing order)
        for maturity in reversed(maturities):

            # Extract the strikes for this maturity
            strikes = tables['imp_vol_data'][tables['imp_vol_data'][
                'Days']==maturity]['Strike']

            # And the vols (specifying the voltype)
            vols = tables['imp_vol_data'][tables['imp_vol_data'][
                'Days']==maturity][str(
                    params['vols_dict'][str(params['voltype'])])]

            # Fit a polynomial to this data
            curve_fit = np.polyfit(strikes, vols, params['order'])
            p = np.poly1d(curve_fit)

            # Create empty list to store smoothed implied vols
            iv_new = []

            # For each strike
            for strike in strikes_full:

                # Add the smoothed value to the iv_new list
                iv_new.append(p(strike))

            # Append this list as a new column in the smooth_surf
            # DataFrame
            tables['smooth_surf'].insert(0, str(maturity), iv_new)

        # Apply the _vol_map function to add smoothed vol column to
        # DataFrame
        tables['imp_vol_data_smoothed'] = (
            tables['imp_vol_data_smoothed'].apply(
                lambda x: cls._vol_map(x, tables), axis=1))

        return params, tables


    @staticmethod
    def _vol_map(
        row: pd.Series,
        tables: dict) -> pd.Series:
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
        row['Smoothed Vol'] = (
            tables['smooth_surf'].loc[row['Strike'], str(row['Days'])])

        return row


    @classmethod
    def map_vols(
        cls,
        params: dict,
        tables: dict) -> tuple[inter._rbf.Rbf, inter._rbf.Rbf]:
        """
        Create vol surface mapping function

        Parameters
        ----------
        tables : Dict
            Dictionary containing the market data tables.

        Returns
        -------
        vol_surface : scipy.interpolate.rbf.Rbf
            Vol surface interpolation function.

        """
        params, tables = cls.smooth(params=params, tables=tables)
        data = tables['imp_vol_data_smoothed']
        t_vols_smooth = data['Smoothed Vol'] * 100
        t_vols = data['Imp Vol - Last'] * 100
        t_strikes = data['Strike']
        t_ttm = data['TTM'] * 365
        vol_surface = sp.interpolate.Rbf(
            t_strikes,
            t_ttm,
            t_vols,
            function=params['rbffunc'],
            smooth=5,
            epsilon=5)

        vol_surface_smoothed = sp.interpolate.Rbf(
            t_strikes,
            t_ttm,
            t_vols_smooth,
            function=params['rbffunc'],
            smooth=5,
            epsilon=5)

        return vol_surface, vol_surface_smoothed


    @staticmethod
    def get_vol(
        maturity: str,
        strike: int,
        params: dict,
        surface_models: dict) -> float:
        """
        Return implied vol for a given maturity and strike

        Parameters
        ----------
        maturity : Str
            The date for the option maturity, expressed as 'YYYY-MM-DD'.
        strike : Int
            The strike expressed as a percent, where ATM = 100.

        Returns
        -------
        imp_vol : Float
            The implied volatility.

        """
        strike_level = params['spot'] * strike / 100
        maturity_date = dt.datetime.strptime(maturity, '%Y-%m-%d')
        start_date = dt.datetime.strptime(params['start_date'], '%Y-%m-%d')
        ttm = (maturity_date - start_date).days
        if params['smoothing']:
            imp_vol = surface_models[
                'vol_surface_smoothed'](strike_level, ttm)
        else:
            imp_vol = surface_models['vol_surface'](strike_level, ttm)

        return np.round(imp_vol, 2)


    @classmethod
    def create_vol_dict(
        cls,
        params: dict,
        surface_models: dict) -> dict:
        """
        Create dictionary of implied vols by tenor and strike to use in skew
        report

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        surface_models : Dict
            Dictionary of vol surfaces.

        Returns
        -------
        vol_dict : Dict
            Dictionary of implied vols.

        """
        vol_dict = {}
        start_date = dt.datetime.strptime(params['start_date'], '%Y-%m-%d')
        for month in range(1, params['skew_months']+1):
            for strike in [80, 90, 100, 110, 120]:
                maturity = dt.datetime.strftime(
                    start_date + relativedelta(months=month), '%Y-%m-%d')
                vol_dict[(month, strike)] = cls.get_vol(
                    maturity=maturity, strike=strike, params=params,
                    surface_models=surface_models)

        return vol_dict


    @classmethod
    def print_skew_report(
        cls,
        vol_dict: dict,
        params: dict) -> None:
        """
        Print a report showing implied vols for 80%, 90% and ATM strikes and
        selected tenor length

        Parameters
        ----------
        vol_dict : Dict
            Dictionary of implied vols.
        params : Dict
            Dictionary of key parameters.

        Returns
        -------
        Prints the report to the console.

        """
        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')

        if params['skew_direction'] == 'full':
            cls._full_skew(vol_dict=vol_dict, params=params, dp2=dp2)
        else:
            cls._header(params=params)

            if params['skew_direction'] == 'up':
                cls._upside_skew(vol_dict=vol_dict, params=params, dp2=dp2)

            else:
                cls._downside_skew(vol_dict=vol_dict, params=params, dp2=dp2)


    @staticmethod
    def _header(params: dict) -> None:

        print('='*78)
        print(': {:^74} :'.format('Skew Summary'))
        print('-'*78)

        # Contract traded on left and period covered on right
        print(': Underlying Ticker : {:<19}{} : {} :'.format(
            params['ticker_label'],
            'Close of Business Date',
            params['start_date']))
        print('-'*78)

        # Strike and skew headers
        print(': {:^12} :{:^34} : {:^23} :'.format(
            'Maturity',
            'Strike',
            'Skew'))
        print('-'*78)

        if params['skew_direction'] == 'up':

            print(': {:>15}{:>7}   : {:>7}   : {:>7}   : {:>10}'\
                  ' : {:>10} :'.format(
                ': ',
                'ATM',
                '110%',
                '120%',
                '+10% Skew',
                '+20% Skew'))

        if params['skew_direction'] == 'down':
            print(': {:>15}{:>7}   : {:>7}   : {:>7}   : {:>10}'\
                  ' : {:>10} :'.format(
                ': ',
                '80%',
                '90%',
                'ATM',
                '-10% Skew',
                '-20% Skew'))


    @staticmethod
    def _downside_skew(
        vol_dict: dict,
        params: dict,
        dp2: Decimal) -> None:

        # Monthly skew summary for selected number of months
        for month in range(1, params['skew_months'] + 1):
            if month < 10:
                month_label = ' '+str(month)
            else:
                month_label = str(month)
            print(': {} Month Vol : {:>7}   : {:>7}   : {:>7}   : {:>7}'\
                  '    : {:>7}    :'.format(
                month_label,
                Decimal(vol_dict[(month, 80)]).quantize(dp2),
                Decimal(vol_dict[(month, 90)]).quantize(dp2),
                Decimal(vol_dict[(month, 100)]).quantize(dp2),
                Decimal((vol_dict[(month, 90)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 80)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2)))

        print('-'*78)
        print('='*78)


    @staticmethod
    def _upside_skew(
        vol_dict: dict,
        params: dict,
        dp2: Decimal) -> None:

        # Monthly skew summary for selected number of months
        for month in range(1, params['skew_months'] + 1):
            if month < 10:
                month_label = ' '+str(month)
            else:
                month_label = str(month)
            print(': {} Month Vol : {:>7}   : {:>7}   : {:>7}   : {:>7}'\
                  '    : {:>7}    :'.format(
                month_label,
                Decimal(vol_dict[(month, 100)]).quantize(dp2),
                Decimal(vol_dict[(month, 110)]).quantize(dp2),
                Decimal(vol_dict[(month, 120)]).quantize(dp2),
                Decimal((vol_dict[(month, 110)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 120)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2)))

        print('-'*78)
        print('='*78)


    @staticmethod
    def _full_skew(
        vol_dict: dict,
        params: dict,
        dp2: Decimal) -> None:

        print('='*115)
        print(': {:^111} :'.format('Skew Summary'))
        print('-'*115)

        # Contract traded on left and period covered on right
        print(': Underlying Ticker : {:<56}{} : {} :'.format(
            params['ticker_label'],
            'Close of Business Date',
            params['start_date']))
        print('-'*115)

        # Strike and skew headers
        print(': {:^13} : {:^47} : {:^45} :'.format(
            'Maturity',
            'Strike',
            'Skew'))
        print('-'*115)

        # Header rows
        print(': {:>16}{:>6}  : {:>6}  : {:>6}  : {:>6}  : {:>6}  : {:>9}'\
              ' : {:>9} : {:>9} : {:>9} :'.format(
            ': ',
            '80%',
            '90%',
            'ATM',
            '110%',
            '120%',
            '-20% Skew',
            '-10% Skew',
            '+10% Skew',
            '+20% Skew'))

        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')

        # Monthly skew summary for selected number of months
        for month in range(1, params['skew_months'] + 1):
            if month < 10:
                month_label = ' '+str(month)
            else:
                month_label = str(month)
            print(': {} Month Vol  : {:>6}  : {:>6}  : {:>6}  : {:>6}  : '\
                  '{:>6}  : {:>7}   : {:>7}   : {:>7}   : {:>7}   :'.format(
                month_label,
                Decimal(vol_dict[(month, 80)]).quantize(dp2),
                Decimal(vol_dict[(month, 90)]).quantize(dp2),
                Decimal(vol_dict[(month, 100)]).quantize(dp2),
                Decimal(vol_dict[(month, 110)]).quantize(dp2),
                Decimal(vol_dict[(month, 120)]).quantize(dp2),
                Decimal((vol_dict[(month, 80)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2),
                Decimal((vol_dict[(month, 90)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 110)]
                         - vol_dict[(month, 100)]) / 10).quantize(dp2),
                Decimal((vol_dict[(month, 120)]
                         - vol_dict[(month, 100)]) / 20).quantize(dp2)))

        print('-'*115)
        print('='*115)
