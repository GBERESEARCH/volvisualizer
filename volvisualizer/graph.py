"""
Methods for graphing volatility data

"""
import copy
import os
import warnings
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy as sp
from matplotlib import axes
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from plotly.offline import plot
from scipy.interpolate import griddata
from volvisualizer.vol_methods import VolMethods
# pylint: disable=invalid-name, consider-using-f-string

class Graph():
    """
    Methods for graphing volatility data

    """
    @classmethod
    def line_graph(
        cls,
        params: dict,
        tables: dict) -> tuple[dict, dict]:
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
        # Create a sorted list of the different number of option
        # expiries
        dates = sorted(list(set(tables['imp_vol_data']['Expiry'])))

        # Create a sorted list of the different number of option time
        # to maturity
        tenors = sorted(list(set(tables['imp_vol_data']['TTM'])))

        # Combine these in a dictionary
        tenor_date_dict = dict(zip(dates, tenors))

        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update(params['mpl_line_params'])
        fig_size = (12, 9)

        # Create figure, axis objects
        fig, ax = plt.subplots(figsize=fig_size)

        # Create values that scale fonts with fig_size
        ax_font_scale = int(round(fig_size[0] * 1.5))
        title_font_scale = int(round(fig_size[0] * 2))

        # Set fontsize of axis ticks
        ax.tick_params(axis='both', which='major', labelsize=ax_font_scale)

        opt_dict = {}
        # For each expiry date
        for exp_date, tenor in tenor_date_dict.items():
            opt_dict[str(exp_date)] = {}
            opt_dict[str(exp_date)]['strikes'] = np.array(tables['imp_vol_data'][
                tables['imp_vol_data']['TTM']==tenor]['Strike'])
            opt_dict[str(exp_date)]['vols'] = np.array(tables['imp_vol_data'][
                tables['imp_vol_data']['TTM']==tenor][str(
                    params['vols_dict'][str(params['voltype'])])] * 100)
            opt_dict[str(exp_date)]['label'] = str(exp_date)+' Expiry'

            # Plot the specified voltype against strike
            ax.plot(
                np.array(tables['imp_vol_data'][
                    tables['imp_vol_data']['TTM']==tenor]['Strike']),
                np.array(tables['imp_vol_data'][
                    tables['imp_vol_data']['TTM']==tenor][str(
                        params['vols_dict'][str(params['voltype'])])] * 100),
                label=str(exp_date)+' Expiry')


        opt_dict['x_label'] = 'Strike'
        opt_dict['y_label'] = 'Implied Volatility %'
        opt_dict['legend_title'] = 'Option Expiry'

        opt_dict['title'] = (
            str(params['ticker_label'])
            +' Implied Volatility '
            +str(params['voltype'].title())
            +' Price '
            +str(params['start_date'])
            )

        plt.grid(True)

        # Label axes
        ax.set_xlabel(opt_dict['x_label'], fontsize=ax_font_scale)
        ax.set_ylabel(opt_dict['y_label'], fontsize=ax_font_scale)

        # Set legend title and font sizes
        ax.legend(title=opt_dict['legend_title'],
                fontsize=ax_font_scale*0.6,
                title_fontsize=ax_font_scale*0.8)

        # Specify title with ticker label, voltype and date and shift
        # away from chart

        st = fig.suptitle(
            opt_dict['title'],
            fontsize=title_font_scale,
            fontweight=0,
            color='black',
            style='italic',
            y=1.02)
        st.set_y(0.95)
        fig.subplots_adjust(top=0.9)

        # Display graph
        plt.show()

        if params['save_image']:
            # save the image as a png file
            fig = cls._image_save(params=params, fig=fig)

        if params['data_output']:
            data_dict = {
                'params': params,
                'tables': tables,
                'opt_dict': opt_dict
            }

        return params, tables


    @classmethod
    def scatter_3d(
        cls,
        params: dict,
        tables: dict) -> tuple[dict, dict]:
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

        # Create figure and axis objects and format
        fig, ax = cls._graph_format(params=params)

        # Create copy of data
        tables['data_3D'] = copy.deepcopy(tables['imp_vol_data'])

        # Filter out any zero prices
        tables['data_3D'] = tables['data_3D'][tables['data_3D'][str(
            params['prices_dict'][str(params['voltype'])])] != 0]

        # Specify the 3 axis values
        x = tables['data_3D']['Strike']
        y = tables['data_3D']['TTM'] * 365
        z = tables['data_3D'][str(params['vols_dict'][str(
            params['voltype'])])] * 100

        # Display scatter, specifying colour to vary with z-axis and use
        # colormap 'viridis'
        ax.scatter3D(x, y, z, c=z, cmap='viridis')

        if params['save_image']:
            # save the image as a png file
            fig = cls._image_save(params=params, fig=fig)

        return params, tables


    @classmethod
    def surface_3d(
        cls,
        params: dict,
        tables: dict) -> tuple[dict, dict]:
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

        # Suppress mpl user warning about data containing nan values
        warnings.filterwarnings(
            "ignore", category=UserWarning, message='Z contains NaN values. '\
                'This may result in rendering artifacts.')

        # If smoothing is set to False
        if params['smoothing'] is False:

            # Create copy of implied vol data
            tables['data_3D'] = copy.deepcopy(tables['imp_vol_data'])

            # Filter out any zero prices
            tables['data_3D'] = tables['data_3D'][tables['data_3D'][str(
                params['prices_dict'][str(params['voltype'])])] != 0]

            # Set 'graph vol' to be the specified voltype
            tables['data_3D']['Graph Vol'] = tables['data_3D'][str(
                params['vols_dict'][str(params['voltype'])])]

        # Otherwise, if smoothing is set to True
        else:
            # Apply the smoothing function to the specified voltype
            params, tables = VolMethods.smooth(params=params, tables=tables)

            # Create copy of implied vol data
            tables['data_3D'] = copy.deepcopy(tables['imp_vol_data_smoothed'])

            # Filter out any zero prices
            tables['data_3D'] = tables['data_3D'][tables['data_3D'][str(
                params['prices_dict'][str(params['voltype'])])] != 0]

            # Set 'graph vol' to be the smoothed vol
            tables['data_3D']['Graph Vol'] = tables['data_3D']['Smoothed Vol']

        # Specify the 3 axis values
        params['x'] = tables['data_3D']['Strike']
        params['y'] = tables['data_3D']['TTM'] * 365
        params['z'] = tables['data_3D']['Graph Vol'] * 100

        if params['surfacetype'] == 'trisurf':
            fig = cls._trisurf_graph(params=params)

        elif params['surfacetype'] == 'mesh':
            fig = cls._mesh_graph(params=params)

        elif params['surfacetype'] == 'spline':
            fig = cls._spline_graph(params=params, tables=tables)

        elif params['surfacetype'] in ['interactive_mesh',
                                       'interactive_spline']:
            fig = cls._interactive_graph(params=params, tables=tables)

        else:
            print("Enter a valid surfacetype from 'trisurf', 'mesh', "\
                  "'spline', 'interactive_mesh', 'interactive_spline'")

        if params['save_image']:
            # save the image as a png file
            fig = cls._image_save(params=params, fig=fig)

        # Set warnings back to default
        warnings.filterwarnings("default", category=UserWarning)

        return params, tables


    @classmethod
    def _trisurf_graph(cls, params: dict) -> mplfig.Figure:

        # Create figure and axis objects and format
        fig, ax = cls._graph_format(params=params)

        # Display triangular surface plot, using colormap 'viridis'
        ax.plot_trisurf(params['x'],
                        params['y'],
                        params['z'],
                        cmap='viridis',
                        edgecolor='none')

        return fig


    @classmethod
    def _mesh_graph(cls, params: dict) -> mplfig.Figure:

        # Create arrays across x and y-axes of equally spaced points
        # from min to max values
        x1, y1 = np.meshgrid(
            np.linspace(min(params['x']),
                        max(params['x']),
                        int(params['spacegrain'])),
            np.linspace(min(params['y']),
                        max(params['y']),
                        int(params['spacegrain'])))

        # Map the z-axis with the scipy griddata method, applying
        # cubic spline interpolation
        z1 = griddata(np.array([params['x'],params['y']]).T,
                      np.array(params['z']),
                      (x1,y1),
                      method='cubic')

        # Create figure and axis objects and format
        fig, ax = cls._graph_format(params=params)

        # Plot the surface
        ax.plot_surface(x1, y1, z1)

        # Apply contour lines
        ax.contour(x1, y1, z1)

        plt.show()

        return fig


    @classmethod
    def _spline_graph(
        cls,
        params: dict,
        tables: dict) -> mplfig.Figure:

        # Create arrays across x and y-axes of equally spaced points
        # from min to max values
        x1 = np.linspace(min(params['x']),
                         max(params['x']),
                         int(params['spacegrain']))
        y1 = np.linspace(min(params['y']),
                         max(params['y']),
                         int(params['spacegrain']))
        x2, y2 = np.meshgrid(x1, y1, indexing='xy')

        # Initialize the z-axis as an array of zero values
        z2 = np.zeros((params['x'].size, params['z'].size))

        # Apply scipy interpolate radial basis function, choosing
        # the rbffunc parameter
        spline = sp.interpolate.Rbf(
            params['x'],
            params['y'],
            params['z'],
            function=params['rbffunc'],
            smooth=5,
            epsilon=5)

        # Populate z-axis array using this function
        z2 = spline(x2, y2)

        # Create figure and axis objects and format
        fig, ax = cls._graph_format(params=params)

        # Plot the surface
        ax.plot_wireframe(x2, y2, z2)
        ax.plot_surface(x2, y2, z2, alpha=0.2)

        # If scatter is True, overlay the surface with the
        # unsmoothed scatter points
        if params['scatter']:
            params['z'] = tables['data_3D'][str(
                params['vols_dict'][str(params['voltype'])])] * 100
            ax.scatter3D(params['x'], params['y'], params['z'], c='r')

        return fig

    @classmethod
    def _interactive_graph(
        cls,
        params: dict,
        tables: dict) -> go.Figure:

        params = cls._set_contours(params=params, tables=tables)

        # Specify the 3 axis values
        params['x'] = tables['data_3D']['TTM'] * 365
        params['y'] = tables['data_3D']['Strike']
        params['z'] = tables['data_3D']['Graph Vol'] * 100

        # Create arrays across x and y-axes of equally spaced
        # points from min to max values
        x1 = np.linspace(
            params['x'].min(),
            params['x'].max(),
            int(params['spacegrain'])
            )
        y1 = np.linspace(
            params['y'].min(),
            params['y'].max(),
            int(params['spacegrain'])
            )
        params['x2'], params['y2'] = np.meshgrid(x1, y1, indexing='xy')

        # If surfacetype is 'interactive_mesh', map the z-axis with
        # the scipy griddata method, applying cubic spline
        # interpolation
        if params['surfacetype'] == 'interactive_mesh':
            params['z2'] = griddata((params['x'], params['y']),
                                    params['z'],
                                    (params['x2'], params['y2']),
                                    method='cubic')

        # If surfacetype is 'interactive_spline', apply scipy
        # interpolate radial basis function, choosing the rbffunc
        # parameter
        if params['surfacetype'] == 'interactive_spline':
            params['z2'] = np.zeros((params['x'].size, params['z'].size))
            spline = sp.interpolate.Rbf(
                params['x'],
                params['y'],
                params['z'],
                function=params['rbffunc'],
                smooth=5,
                epsilon=5)
            params['z2'] = spline(params['x2'], params['y2'])

        # Initialize Figure object
        if params['scatter']:
            fig = cls._int_scatter(params=params, tables=tables)

        # Plot just the surface
        else:
            fig = cls._int_surf(params=params)

        # Format the chart layout
        fig = cls._int_layout(params=params, fig=fig)

        # If running within a Jupyter notebook, plot graph inline
        if params['notebook'] is True:
            fig.show()

        # Otherwise create a new HTML window to display
        else:
            plot(fig, auto_open=True)

        return fig


    @staticmethod
    def _set_contours(
        params: dict,
        tables: dict) -> dict:

        # Set the range of x, y and z contours and interval
        params['contour_x_start'] = 0
        params['contour_x_stop'] = 2 * 360
        params['contour_x_size'] = params['contour_x_stop'] / 18
        params['contour_y_start'] = tables['data_3D']['Strike'].min()
        params['contour_y_stop'] = tables['data_3D']['Strike'].max()

        # Vary the strike interval based on spot level
        if ((tables['data_3D']['Strike'].max()
             - tables['data_3D']['Strike'].min()) > 2000):
            params['contour_y_size'] = 200
        elif ((tables['data_3D']['Strike'].max()
               - tables['data_3D']['Strike'].min()) > 1000):
            params['contour_y_size'] = 100
        elif ((tables['data_3D']['Strike'].max()
               - tables['data_3D']['Strike'].min()) > 250):
            params['contour_y_size'] = 50
        elif ((tables['data_3D']['Strike'].max()
               - tables['data_3D']['Strike'].min()) > 50):
            params['contour_y_size'] = 10
        else:
            params['contour_y_size'] = 5

        # Set z contours
        params['contour_z_start'] = 0
        params['contour_z_stop'] = 100
        params['contour_z_size'] = 5

        return params


    @staticmethod
    def _int_scatter(
        params: dict,
        tables: dict) -> go.Figure:

        # Set z to raw data points
        params['z'] = (tables['data_3D'][str(
            params['vols_dict'][str(params['voltype'])])] * 100)

        # Create figure object with fitted surface and scatter
        # points
        if params['surf']:
            fig = go.Figure(
                data=[go.Surface(
                    x=params['x2'],
                    y=params['y2'],
                    z=params['z2'],

                    # Specify the colors to be used
                    colorscale=params['colorscale'],

                    # Define the contours
                    contours = {
                        "x": {
                            "show": True,
                            "start":params['contour_x_start'],
                            "end":params['contour_x_stop'],
                            "size":params['contour_x_size'],
                            "color":"white"
                            },
                        "y": {
                            "show": True,
                            "start":params['contour_y_start'],
                            "end":params['contour_y_stop'],
                            "size":params['contour_y_size'],
                            "color":"white"
                            },
                        "z": {
                            "show": True,
                            "start":params['contour_z_start'],
                            "end": params['contour_z_stop'],
                            "size": params['contour_z_size']
                            }
                        },

                    # Set the surface opacity
                    opacity=params['opacity']),

                # Plot scatter of unsmoothed data
                go.Scatter3d(
                    x=params['x'],
                    y=params['y'],
                    z=params['z'],
                    mode='markers',
                    # Set size, color and opacity of each data point
                    marker={
                        'size':2,
                        'color':'red',
                        'opacity':0.9
                        }
                    )
                ])

        # Plot just the scatter points
        else:
            fig = go.Figure(data=[
                # Plot scatter of unsmoothed data
                go.Scatter3d(
                    x=params['x'],
                    y=params['y'],
                    z=params['z'],
                    mode='markers',
                    # Set size, color and opacity of each data point
                    marker={
                        'size':2,
                        'color':'red',
                        'opacity':0.9
                        }
                    )
                ])

        return fig


    @staticmethod
    def _int_surf(params: dict) -> go.Figure:

        # Create figure object with fitted surface
        fig = go.Figure(
            data=[go.Surface(
                x=params['x2'],
                y=params['y2'],
                z=params['z2'],

                # Specify the colors to be used
                colorscale=params['colorscale'],

                # Define the contours
                contours = {
                    "x": {
                        "show": True,
                        "start":params['contour_x_start'],
                        "end":params['contour_x_stop'],
                        "size":params['contour_x_size'],
                        "color":"white"
                        },
                    "y": {
                        "show": True,
                        "start":params['contour_y_start'],
                        "end":params['contour_y_stop'],
                        "size":params['contour_y_size'],
                        "color":"white"
                        },
                    "z": {
                        "show": True,
                        "start":params['contour_z_start'],
                        "end":params['contour_z_stop'],
                        "size":params['contour_z_size']
                        }
                    },

                   # Set the surface opacity
                   opacity=params['opacity'])])

        return fig


    @staticmethod
    def _int_layout(
        params: dict,
        fig: go.Figure) -> go.Figure:

        # Set initial camera angle
        params['camera'] = {
            'eye': {
                'x':2,
                'y':1,
                'z':1
                }
            }

        # Set Time To Expiration to increase left to right
        fig.update_scenes(xaxis_autorange="reversed")
        fig.update_layout(
            scene={
                'xaxis': {
                    'backgroundcolor': "rgb(200, 200, 230)",
                    'gridcolor': "white",
                    'showbackground': True,
                    'zerolinecolor': "white"
                    },
                'yaxis': {
                    'backgroundcolor': "rgb(230, 200,230)",
                    'gridcolor': "white",
                    'showbackground': True,
                    'zerolinecolor': "white"
                    },
                'zaxis': {
                    'backgroundcolor': "rgb(230, 230,200)",
                    'gridcolor': "white",
                    'showbackground': True,
                    'zerolinecolor': "white"
                    },
                'aspectmode': 'cube',
                # Label axes
                'xaxis_title': 'Time to Expiration (Days)',
                'yaxis_title': 'Strike',
                'zaxis_title': 'Implied Volatility %'
                },
            # Specify title with ticker label, voltype
            # and date
            title={
                'text':(
                    str(params['ticker_label'])
                    +' Implied Volatility '
                    +str(params['voltype'].title())
                    +' Price '
                    +str(params['start_date'])
                    ),
                'y':0.9,
                'x':0.5,
                'xanchor':'center',
                'yanchor':'top',
                'font': {
                    'size': 20,
                    'color': "black"
                    }
                },
            autosize=False,
            width=800,
            height=800,
            margin={
                'l':65,
                'r':50,
                'b':65,
                't':90
                },
            scene_camera=params['camera'])

        return fig


    @staticmethod
    def _graph_format(params: dict) -> tuple[mplfig.Figure, axes.Axes]:

        # Update chart parameters
        plt.rcParams.update(params['mpl_3D_params'])

        # Create fig object
        fig = plt.figure(figsize=params['fig_size'])

        # Create axes object
        ax = fig.add_subplot(
            111,
            projection='3d',
            azim=params['azim'],
            elev=params['elev'])

        # Set background color to white
        ax.set_facecolor('w')

        # Create values that scale fonts with fig_size
        ax_font_scale = int(round(params['fig_size'][0] * 1.1))
        title_font_scale = int(round(params['fig_size'][0] * 1.5))

        # Tint the axis panes, RGB values from 0-1 and alpha denoting
        # color intensity
        ax.w_xaxis.set_pane_color((0.9, 0.8, 0.9, 0.8))
        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.9, 0.8))
        ax.w_zaxis.set_pane_color((0.9, 0.9, 0.8, 0.8))

        # Set z-axis to left hand side
        ax.zaxis._axinfo['juggled'] = (1, 2, 0) # pylint: disable=protected-access

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
        st = fig.suptitle(str(params['ticker_label'])
                          +' Implied Volatility '
                          +str(params['voltype'].title())
                          +' Price '
                          +str(params['start_date']),
                          fontsize=title_font_scale,
                          fontweight=0,
                          color='black',
                          style='italic',
                          y=1.02)

        st.set_y(0.95)
        fig.subplots_adjust(top=1)

        return fig, ax


    @staticmethod
    def _image_save(
        params: dict,
        fig: mplfig.Figure | go.Figure) -> mplfig.Figure | go.Figure:

        # Create image folder if it does not already exist
        if not os.path.exists(params['image_folder']):
            os.makedirs(params['image_folder'])

        # save the image as a png file
        plt.savefig('{}/{}{}.png'.format(
            params['image_folder'], params['ticker_label'],
            params['start_date']), dpi=params['image_dpi'])

        return fig
