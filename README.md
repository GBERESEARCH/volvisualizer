# volvisualizer
## Extract and visualize implied volatility from option chain data.

&nbsp;

A tool to extract option data from Yahoo Finance and provide simple visualization and smoothing to get a general sense of the shape of the volatility surface.

&nbsp;

### Installation
Install from PyPI:
```
$ pip install volvisualizer
```

&nbsp;

To install in new environment using anaconda:
```
$ conda create --name volvis
```
Activate new environment
```
$ activate volvis
```
Install Python
```
(volvis) $ conda install python
```
Install Spyder
```
(volvis) $ conda install spyder=4
```
Install package
```
(volvis) $ pip install volvisualizer
```

&nbsp;

### Setup
Import volatility module and initialise a Volatility object

```
import volvisualizer.volatility as vol
imp = vol.Volatility()
```

&nbsp;

Extract URLs and the option data for each, specifying a start date used in refining the dataset
```
imp.extracturls('^SPX').extractoptions().transform(start_date, monthlies=True)
```

&nbsp;

Select a range of call and put strikes and spot reference
```
imp.combine(ticker, put_strikes, call_strikes, spot, r, q, epsilon, method='NR')
```

&nbsp;

#### Line graph of individual option expiries.
```
imp.visualize(graphtype='line')
```
![tsla_line](images/tsla_line.png)

&nbsp;

#### 3D Scatter plot of each option implied volatility by strike and expiry.
```
imp.visualize(graphtype='scatter', voltype='ask')
```
![aapl_scatter](images/aapl_scatter.png)

&nbsp;

#### 3D Wireframe plot with scatter of each option implied volatility by strike and expiry.
```
imp.visualize(graphtype='surface', surfacetype='spline', scatter=True, smoothing=True)
```
![gld_wire_scatter](images/gld_wire_scatter.png)

&nbsp;

#### 3D Meshgrid plot of each option implied volatility by strike and expiry.
```
imp.visualize(graphtype='surface', surfacetype='mesh', smoothing=True)
```
![spx_mesh](images/spx_mesh.png)

&nbsp;

#### 3D Interactive plot of each option implied volatility by strike and expiry that can be rotated and zoomed.
```
imp.visualize(graphtype='surface', surfacetype='interactive_spline', smoothing=True, notebook=True)
```
![aapl_int_spline](images/aapl_int_spline.png)

&nbsp;

#### 3D Interactive plot of each option implied volatility by strike and expiry using radial basis function interpolation.
```
imp.visualize(graphtype='surface', surfacetype='interactive_spline', rbffunc='cubic', colorscale='Jet', smoothing=True)
```

![tsla_int_rbf](images/tsla_int_rbf.png)

&nbsp;

Some simplifying assumptions have been made:
  - interest rates are constant; for greater accuracy a term structure should be employed.
  - the prices are taken to be valid at the snap time; if the last trade is some time ago and / or the market is volatile then this will be less accurate.

&nbsp;

There are parameters to filter the data based on time since last trade, volume, open interest and select only the monthly options expiring on the 3rd Friday. 

Some of the smoothing techniques are very sensitive to the quality (and quantity) of input data. Overfitting becomes a problem if there aren't enough data points and the more illiquid tickers often generate results that are not to be relied upon.

Additional work is required to calibrate to an arbitrage free surface.