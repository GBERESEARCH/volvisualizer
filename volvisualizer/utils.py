"""
Utility functions for refreshing parameters and getting implied volatility
from prices

"""
import copy
import datetime as dt
import numpy as np
import scipy.stats as si
from pandas.tseries.offsets import BDay
from volvisualizer.volatility_params import vol_params_dict
# pylint: disable=invalid-name

class Utils():
    """
    Utility functions for refreshing parameters

    """
    @classmethod
    def init_params(cls, inputs):
        """
        Initialise parameter dictionary
        Parameters
        ----------
        inputs : Dict
            Dictionary of parameters supplied to the function.
        Returns
        -------
        params : Dict
            Dictionary of parameters.
        """
        # Copy the default parameters
        params = copy.deepcopy(vol_params_dict)

        # For all the supplied arguments
        for key, value in inputs.items():

            # Replace the default parameter with that provided
            params[key] = value

        params = cls.set_start_date(params=params)

        return params


    @staticmethod
    def set_start_date(params):
        """
        Set start date to previous working day if not provided

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.

        """
        if params['start_date'] is None:
            start_date_as_dt = (dt.datetime.today() - BDay(1)).date()
            params['start_date'] = str(start_date_as_dt)

        return params


class ImpliedVol():
    """
    Implied Volatility Extraction methods

    """
    @classmethod
    def implied_vol_newton_raphson(cls, opt_params):
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
    def implied_vol_bisection(cls, opt_params):
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
    def implied_vol_naive(cls, opt_params):
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
    def implied_vol_naive_verbose(cls, opt_params):
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
    def black_scholes_merton(opt_params, sigma):
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
    def black_scholes_merton_vega(opt_params, sigma):
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
