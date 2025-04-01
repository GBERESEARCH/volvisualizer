"""
Extract option prices from Yahoo Finance and visualize volatility

"""

import copy
from volvisualizer.graph import Graph
from volvisdata.market_data import Data
from volvisdata.utils import Utils
from volvisdata.volatility_params import vol_params_dict
from volvisdata.vol_methods import VolMethods
from volvisdata.skew_report import SkewReport

class Volatility():
    """
    Extract option prices from Yahoo Finance and visualize volatility

    """
    def __init__(self, **kwargs):
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

        # Dictionary of parameter defaults
        self.df_dict = copy.deepcopy(vol_params_dict)

        # Store initial inputs
        inputs = {}
        for key, value in kwargs.items():
            inputs[key] = value

        # Initialise system parameters
        params = Utils.init_params(inputs)
        tables = {}

        # Update holiday calendar
        params = Data.trading_calendar(params=params)

        # Generate option data
        params, tables = Data.create_option_data(params=params, tables=tables)

        # Map volatilities
        surface_models = {}
        surface_models['vol_surface'], \
            surface_models['vol_surface_smoothed'] = VolMethods.map_vols(
                params=params, tables=tables)

        self.data_dict = {}
        self.params = params
        self.tables = tables
        self.surface_models = surface_models


    def visualize(self, **kwargs):
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

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        # Run method selected by graphtype
        if self.params['graphtype'] == 'line':
            if self.params['data_output']:
                self.data_dict = Graph.line_graph(
                params=self.params, tables=self.tables)

            self.params, self.tables = Graph.line_graph(
                params=self.params, tables=self.tables)

        elif self.params['graphtype'] == 'scatter':
            if self.params['data_output']:
                self.data_dict = Graph.scatter_3d(
                params=self.params, tables=self.tables)

            self.params, self.tables = Graph.scatter_3d(
                params=self.params, tables=self.tables)

        elif self.params['graphtype'] == 'surface':
            if self.params['data_output']:
                self.data_dict = Graph.surface_3d(
                params=self.params, tables=self.tables)

            self.params, self.tables = Graph.surface_3d(
                params=self.params, tables=self.tables)

        else:
            print ("Please select a graphtype from 'line', "\
                   "'scatter' and 'surface'")

        if self.params['data_output']:
            return self.data_dict

        return None


    def linegraph(self, **kwargs):
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

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        self.params, self.tables = Graph.line_graph(
            params=self.params, tables=self.tables)


    def scatter(self, **kwargs):
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

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        self.params, self.tables = Graph.scatter_3d(
            params=self.params, tables=self.tables)


    def surface(self, **kwargs):
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

        # Update params with the specified parameters
        for key, value in kwargs.items():

            # Replace the default parameter with that provided
            self.params[key] = value

        self.params, self.tables = Graph.surface_3d(
            params=self.params, tables=self.tables)


    def vol(
        self,
        maturity: str,
        strike: int,
        smoothing: bool | None = None):
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
        if smoothing is not None:
            self.params['smoothing'] = smoothing

        return VolMethods.get_vol(
            maturity=maturity, strike=strike, params=self.params,
            surface_models=self.surface_models)


    def skewreport(
        self,
        months: int | None = None,
        direction: str | None = None):
        """
        Print a report showing implied vols for 80%, 90% and ATM strikes and
        selected tenor length

        Parameters
        ----------
        months : Int
            Number of months to display in report. The default is 12.
        direction : Str
            The direction of skew to show. The options are 'up', 'down' and
            'full'. The default is 'down'
        Returns
        -------
        Prints the report to the console.

        """
        if months is not None:
            self.params['skew_months'] = months

        if direction is not None:
            self.params['skew_direction'] = direction

        vol_dict = SkewReport.create_vol_dict(
            params=self.params, surface_models=self.surface_models)

        if self.params['data_output']:
            data_dict = {
                'vol_dict': vol_dict,
                'params': self.params
            }
            return data_dict

        return SkewReport.print_skew_report(
            vol_dict=vol_dict, params=self.params)
