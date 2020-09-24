from setuptools import setup

setup(name='volatility',
      version='0.0.1',
      description='Extract and visualize implied volatility from option chain data',
      author='...',
      author_email='...',
      packages=['volatility', 'models'],
      install_requires=['matplotlib',
			'numpy',
                        'plotly',
			'scipy',
			'pandas',
			'requests',
			'pytz'])

