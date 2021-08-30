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
# pylint: disable=invalid-name

class VolMethods():
    """
    Methods for extracting Implied Vol and producing skew reports

    """
    @classmethod
    def smooth(cls, params, tables):
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
    def _vol_map(row, tables):
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
    def map_vols(cls, params, tables):
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
    def get_vol(maturity, strike, params, surface_models):
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
    def create_vol_dict(cls, params, surface_models):
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
    def print_skew_report(cls, vol_dict, params):
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
    def _header(params):

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
    def _downside_skew(vol_dict, params, dp2):

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
    def _upside_skew(vol_dict, params, dp2):

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
    def _full_skew(vol_dict, params, dp2):

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
