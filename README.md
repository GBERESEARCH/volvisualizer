# volvisualizer
## Extract and visualize implied volatility from option chain data.

A tool to extract option data from Yahoo Finance and provide simple visualization and smoothing to get a general sense of the shape of the volatility surface.

#### Line graph of individual option expiries.
![aapl_line](images/aapl_line.png)

#### 3D Scatter plot of each option implied volatility by strike and expiry.
![spx_scatter](images/spx_scatter.png)

#### 3D Wireframe plot with scatter of each option implied volatility by strike and expiry.
![spx_wire_scatter](images/spx_wire_scatter.png)

#### 3D Meshgrid plot of each option implied volatility by strike and expiry.
![tsla_mesh](images/tsla_mesh.png)

#### 3D Interactive plot of each option implied volatility by strike and expiry that can be rotated and zoomed.
![aapl_int_spline](images/aapl_int_spline.png)

#### 3D Interactive plot of each option implied volatility by strike and expiry using radial basis function interpolation.
![tsla_int_rbf](images/tsla_int_rbf.png)

Some simplifying assumptions have been made:
  - interest rates are constant; for greater accuracy a term structure should be employed.
  - the prices are taken to be valid at the snap time; if the last trade is some time ago and / or the market is volatile then this will be less accurate.

There are parameters to filter the data based on time since last trade, volume, open interest and select only the monthly options expiring on the 3rd Friday. 
Additional work is required to calibrate to an arbitrage free surface.