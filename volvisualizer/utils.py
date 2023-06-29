"""
Utility functions for refreshing parameters and getting implied volatility
from prices

"""
import copy
import datetime as dt
from pandas.tseries.offsets import BDay
from volvisualizer.volatility_params import vol_params_dict
from volvisualizer.market_data_prep import DataPrep
# pylint: disable=invalid-name

class Utils():
    """
    Utility functions for refreshing parameters

    """
    @classmethod
    def init_params(cls, inputs: dict) -> dict:
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

        if 'r' not in inputs.keys():
            cls.set_interest_rate(params=params)
        else:
            cls.set_interest_rate(params=params, r=params['r'])

        if 'q' not in inputs.keys():
            cls.set_dividend_yield(params=params)

        params = cls.set_start_date(params=params)

        return params


    @staticmethod
    def set_start_date(params: dict) -> dict:
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


    @staticmethod
    def set_interest_rate(
        params: dict,
        r: float | None = None) -> dict:
        """
        Returns the yield curve interpolation function

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        r : Float, Optional
            Interest Rate.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.

        """
        params['yield_curve'] = DataPrep.generate_yield_curve(r=r)

        return params


    @staticmethod
    def set_dividend_yield(params: dict) -> dict:
        """
        Returns the dividend yield

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.

        """
        params['q'] = DataPrep.dividend_yield(params['ticker'])

        return params
