import math
import random
import time
import numpy as np
import operator as op
import scipy.stats as si
from functools import reduce, wraps
from scipy.special import comb
from scipy.stats import norm


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        for k, v in kwargs.items():
            if k == 'timing' and v == True:
                print('{}.{} : {} milliseconds'.format(func.__module__, func.__name__, 
                                                       round((end - start)*1e3, 2)))
        return r
    return wrapper


df_dict = {'df_S':100,
           'df_F':100,
           'df_K':100,
           'df_T':0.25,
           'df_r':0.005,
           'df_q':0,
           'df_sigma':0.2,
           'df_option':'call',
           'df_steps':1000,
           'df_steps_itt':10,
           'df_nodes':100,
           'df_vvol':0.5,
           'df_simulations':10000,
           'df_output_flag':'price',
           'df_american':False,
           'df_step':5,
           'df_state':5,
           'df_skew':0.0004,
           'df_sig0':0.09,
           'df_sigLR':0.0625, 
           'df_halflife':0.1,
           'df_rho':0,
           'df_cm':5.0,
           'df_epsilon':0.0001,
           'df_refresh':True,
           'df_timing':False,
           'df_params_list':['S', 'K', 'T', 'r', 'q', 'sigma', 'option', 'steps', 
                             'steps_itt', 'nodes', 'vvol', 'simulations', 'output_flag', 
                             'american', 'step', 'state', 'skew', 'sig0', 'sigLR', 
                             'halflife', 'rho', 'cm', 'epsilon', 'timing']}


sabr_df_dict = {'df_F':100,
                'df_X':70,
                'df_T':0.5,
                'df_r':0.05,
                'df_atmvol':0.3, 
                'df_beta':0.9999, 
                'df_volvol':0.5, 
                'df_rho':-0.4,
                'df_option':'put'}


class Pricer():
    
    def __init__(self, S=df_dict['df_S'], F=df_dict['df_F'], K=df_dict['df_K'], 
                 T=df_dict['df_T'], r=df_dict['df_r'], q=df_dict['df_q'], sigma=df_dict['df_sigma'], 
                 option=df_dict['df_option'], steps=df_dict['df_steps'], steps_itt=df_dict['df_steps_itt'], 
                 nodes=df_dict['df_nodes'], vvol=df_dict['df_vvol'], simulations=df_dict['df_simulations'], 
                 output_flag=df_dict['df_output_flag'], american=df_dict['df_american'], 
                 step=df_dict['df_step'], state=df_dict['df_state'], skew=df_dict['df_skew'], 
                 sig0=df_dict['df_sig0'], sigLR=df_dict['df_sigLR'], halflife=df_dict['df_halflife'], 
                 rho=df_dict['df_rho'], cm=df_dict['df_cm'], epsilon=df_dict['df_epsilon'], 
                 refresh=df_dict['df_refresh'], timing=df_dict['df_timing'], 
                 df_params_list=df_dict['df_params_list'], df_dict=df_dict):
        
        self.S = S # Spot price
        self.F = F # Forward price
        self.K = K # Strike price
        self.T = T # Time to maturity
        self.r = r # Interest rate
        self.q = q # Dividend Yield 
        self.b = self.r - self.q # Cost of carry
        self.sigma = sigma # Volatility
        self.option = option # Option type, call or put
        self.steps = steps # Number of time steps.
        self.steps_itt = steps_itt # Number of time steps for Implied Trinomial Tree.
        self.nodes = nodes # Number of price steps.
        self.vvol = vvol # Vol of vol.
        self.simulations = simulations # Number of Monte Carlo runs.
        self.output_flag = output_flag # Output to return from method
        self.american = american # Whether the option is American.
        self.step = step # Time step used for Arrow Debreu price at single node.
        self.state = state # State position used for Arrow Debreu price at single node.
        self.skew = skew # Rate at which volatility increases (decreases) for every one point decrease 
                         # (increase) in the strike price.
        self.sig0 = sig0 # Initial Volatility.
        self.sigLR = sigLR # Long run mean reversion level of volatility.
        self.halflife = halflife # Half-life of volatility deviation 
        self.rho = rho # Correlation between asset price and volatility
        self.cm = cm # Option price used to solve for vol.
        self.epsilon = epsilon # Degree of precision for implied vol
        self.refresh = refresh # Whether to refresh parameters, set to False if called from another function
        self.df_params_list = df_params_list # List of default parameters
        self.df_dict = df_dict # Dictionary of default parameters
        self.timing = timing
    
        
    def _refresh_params(self, **kwargs):
        """
        Set parameters for use in various pricing functions

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object
                v = df_dict['df_'+str(k)]
                self.__dict__[k] = v
            
            # If the value has been provided as an input, assign this to the object
            else:
                self.__dict__[k] = v
                      
        return self        
        
    
    @timethis
    def black_scholes_merton(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                             option=None, timing=None):
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
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                             timing=timing)
        
        self.b = self.r - self.q
        carry = np.exp((self.b - self.r) * self.T)
        d1 = ((np.log(self.S / self.K) + (self.b + (0.5 * self.sigma ** 2)) * self.T) / 
              (self.sigma * np.sqrt(self.T)))
        d2 = ((np.log(self.S / self.K) + (self.b - (0.5 * self.sigma ** 2)) * self.T) / 
              (self.sigma * np.sqrt(self.T)))
          
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        minusNd1 = si.norm.cdf(-d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        minusNd2 = si.norm.cdf(-d2, 0.0, 1.0)
               
        if self.option == "call":
            opt_price = ((self.S * carry * Nd1) - 
                              (self.K * np.exp(-self.r * self.T) * Nd2))  
        if self.option == 'put':
            opt_price = ((self.K * np.exp(-self.r * self.T) * minusNd2) - 
                              (self.S * carry * minusNd1))
               
        return opt_price
    
    
    @timethis
    def black_scholes_merton_vega(self, S=None, K=None, T=None, r=None, q=None, 
                                  sigma=None, option=None, timing=None):
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
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        opt_vega : Float
            Option Vega.

        """  
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                             timing=timing)

        self.b = self.r - self.q
        carry = np.exp((self.b - self.r) * self.T)
        d1 = ((np.log(self.S / self.K) + (self.b + (0.5 * self.sigma ** 2)) * self.T) / 
              (self.sigma * np.sqrt(self.T)))
        nd1 = (1 / np.sqrt(2 * np.pi)) * (np.exp(-d1 ** 2 * 0.5))
        
        opt_vega = self.S * carry * nd1 * np.sqrt(self.T)
         
        return opt_vega
    
    
    @timethis
    def black_76(self, F=None, K=None, T=None, r=None, sigma=None, option=None, timing=None):
        """
        Black 76 Futures Option price 

        Parameters
        ----------
        F : Float
            Discounted Futures Price.
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

        self._refresh_params(F=F, K=K, T=T, r=r, sigma=sigma, option=option, timing=timing)
        
        carry = np.exp(-self.r * self.T)
        d1 = (np.log(self.F / self.K) + (0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.F / self.K) + (-0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
          
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        minusNd1 = si.norm.cdf(-d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        minusNd2 = si.norm.cdf(-d2, 0.0, 1.0)
               
        if self.option == "call":
            opt_price = ((self.F * carry * Nd1) - 
                              (self.K * np.exp(-self.r * self.T) * Nd2))  
        if self.option == 'put':
            opt_price = ((self.K * np.exp(-self.r * self.T) * minusNd2) - 
                              (self.F * carry * minusNd1))
               
        return opt_price
    
    
    @timethis
    def european_binomial(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                          steps=None, option=None, timing=None):
        """
        European Binomial Option price.
        Combinatorial function won't calculate with values much over 1000
    
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
        steps : Int
            Number of time steps. The default is 1000.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        Float
            European Binomial Option Price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             option=option, timing=timing)
        
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.b * dt) - d) / (u - d)
        a = int(np.log(self.K / (self.S * (d ** self.steps))) / np.log(u / d)) + 1
        
        val = 0
        
        if self.option == 'call':
            for j in range(a, self.steps + 1):
                val = val + (comb(self.steps, j) * (p ** j) * ((1 - p) ** (self.steps - j)) * 
                             ((self.S * (u ** j) * (d ** (self.steps - j))) - self.K))
        if self.option == 'put':
            for j in range(0, a):
                val = val + (comb(self.steps, j) * (p ** j) * ((1 - p) ** (self.steps - j)) * 
                             (self.K - ((self.S * (u ** j)) * (d ** (self.steps - j)))))
                               
        return np.exp(-self.r * self.T) * val                     
                
    
    @timethis
    def cox_ross_rubinstein_binomial(self, S=None, K=None, T=None, r=None, q=None, 
                                     sigma=None, steps=None, option=None, output_flag=None, 
                                     american=None, timing=None):
        """
        Cox-Ross-Rubinstein Binomial model

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
        steps : Int
            Number of time steps. The default is 1000.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        output_flag : Str
            Whether to return 'price', 'delta', 'gamma', 'theta' or 'all'. The default is 'price'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'all' : Tuple; Option Price, Option Delta, Option Gamma, Option Theta  

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             option=option, output_flag=output_flag, american=american, 
                             timing=timing)
                
        z = 1
        if self.option == 'put':
            z = -1
        
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.b * dt) - d) / (u - d)
        df = np.exp(-self.r * dt)
        optionvalue = np.zeros((self.steps + 2))
        returnvalue = np.zeros((4))
        
        for i in range(self.steps + 1):
            optionvalue[i] = max(0, z * (self.S * (u ** i) * (d ** (self.steps - i)) - self.K))
            
            
        for j in range(self.steps - 1, -1, -1):
            for i in range(j + 1):
                if self.american == True:
                    optionvalue[i] = ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df
                if self.american == False:
                    optionvalue[i] = max((z * (self.S * (u ** i) * (d ** (j - i)) - self.K)),  
                                         ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df)
            
            if j == 2:
                returnvalue[2] = ((optionvalue[2] - optionvalue[1]) / 
                                  (self.S * (u ** 2) - self.S) - (optionvalue[1] - optionvalue[0]) / 
                                   (self.S - self.S * (d ** 2))) / (0.5 * (self.S * (u ** 2) - self.S * (d ** 2)))
                returnvalue[3] = optionvalue[1]
                
            if j == 1:
                returnvalue[1] = (optionvalue[1] - optionvalue[0]) / (self.S * u - self.S * d)
            
        returnvalue[3] = (returnvalue[3] - optionvalue[0]) / (2 * dt) / 365
        returnvalue[0] = optionvalue[0]
        
        if self.output_flag == 'price':
            result = returnvalue[0]
        if self.output_flag == 'delta':
            result = returnvalue[1]
        if self.output_flag == 'gamma':
            result = returnvalue[2]
        if self.output_flag == 'theta':
            result = returnvalue[3]
        if self.output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]),
                      'Theta = '+str(returnvalue[3]))
                               
        return result
    
    
    @timethis
    def leisen_reimer_binomial(self, S=None, K=None, T=None, r=None, q=None, 
                                     sigma=None, steps=None, option=None, output_flag=None, 
                                     american=None, timing=None):
        """
        Leisen Reimer Binomial

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
        steps : Int
            Number of time steps. The default is 1000.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        output_flag : Str
            Whether to return 'price', 'delta', 'gamma' or 'all'. The default is 'price'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'all' : Tuple; Option Price, Option Delta, Option Gamma  

        """
         
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             option=option, output_flag=output_flag, american=american, 
                             timing=timing)
        
        z = 1
        if self.option == 'put':
            z = -1
        
        d1 = ((np.log(self.S / self.K) + (self.b + (0.5 * self.sigma ** 2)) * self.T) / 
              (self.sigma * np.sqrt(self.T)))
        d2 = ((np.log(self.S / self.K) + (self.b - (0.5 * self.sigma ** 2)) * self.T) / 
              (self.sigma * np.sqrt(self.T)))
        hd1 = (0.5 + np.sign(d1) * (0.25 - 0.25 * np.exp(-(d1 / (self.steps + 1 / 3 + 
               0.1 / (self.steps + 1))) ** 2 * (self.steps + 1 / 6))) ** (0.5))
        hd2 = (0.5 + np.sign(d2) * (0.25 - 0.25 * np.exp(-(d2 / (self.steps + 1 / 3 + 
               0.1 / (self.steps + 1))) ** 2 * (self.steps + 1 / 6))) ** (0.5))
        
        dt = self.T / self.steps
        p = hd2
        u = np.exp(self.b * dt) * hd1 / hd2
        d = (np.exp(self.b * dt) - p * u) / (1 - p)
        df = np.exp(-self.r * dt)
    
        optionvalue = np.zeros((self.steps + 1))
        returnvalue = np.zeros((4))
        
        for i in range(self.steps + 1):
            optionvalue[i] = max(0, z * (self.S * (u ** i) * (d ** (self.steps - i)) - self.K))
            
        for j in range(self.steps - 1, -1, -1):
            for i in range(j + 1):
                if self.american == True:
                    optionvalue[i] = ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df
                if self.american == False:
                    optionvalue[i] = max((z * (self.S * (u ** i) * (d ** (j - i)) - self.K)),  
                                         ((p * optionvalue[i + 1]) + ((1 - p) * optionvalue[i])) * df)
                    
            if j == 2:
                returnvalue[2] = ((optionvalue[2] - optionvalue[1]) / 
                                  (self.S * (u ** 2) - self.S * u * d) - (optionvalue[1] - optionvalue[0]) / 
                                   (self.S * u * d - self.S * (d ** 2))) / (0.5 * (self.S * (u ** 2) - self.S * (d ** 2)))
                returnvalue[3] = optionvalue[1]
                
            if j == 1:
                returnvalue[1] = (optionvalue[1] - optionvalue[0]) / (self.S * u - self.S * d)
            
        returnvalue[0] = optionvalue[0]
        
        if self.output_flag == 'price':
            result = returnvalue[0]
        if self.output_flag == 'delta':
            result = returnvalue[1]
        if self.output_flag == 'gamma':
            result = returnvalue[2]
        if self.output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]))
    
        return result        
    
    
    @timethis
    def trinomial_tree(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                       steps=None, option=None, output_flag=None, american=None, 
                       timing=None):
        """
        Trinomial Tree

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
        steps : Int
            Number of time steps. The default is 1000.
        option : Str
            Type of option, 'put' or 'call'. The default is 'call'.
        output_flag : Str
            Whether to return 'price', 'delta', 'gamma', 'theta' or 'all'. The default is 'price'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'all' : Tuple; Option Price, Option Delta, Option Gamma, Option Theta  

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             option=option, output_flag=output_flag, american=american, 
                             timing=timing)
                
        z = 1
        if self.option == 'put':
            z = -1
        
        dt = self.T / self.steps
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = np.exp(-self.sigma * np.sqrt(2 * dt))
        pu = ((np.exp(self.b * dt / 2) - np.exp(-self.sigma * np.sqrt(dt / 2))) / 
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pd = ((np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(self.b * dt / 2)) / 
              (np.exp(self.sigma * np.sqrt(dt / 2)) - np.exp(-self.sigma * np.sqrt(dt / 2)))) ** 2
        pm = 1 - pu - pd
        df = np.exp(-self.r * dt)
        optionvalue = np.zeros((self.steps * 2 + 2))
        returnvalue = np.zeros((4))
        
        for i in range(2 * self.steps + 1):
            optionvalue[i] = max(0, z * (self.S * (u ** max(i - self.steps, 0)) * 
                                         (d ** (max((self.steps - i), 0))) - self.K))
            
            
        for j in range(self.steps - 1, -1, -1):
            for i in range(j * 2 + 1):
                
                optionvalue[i] = (pu * optionvalue[i + 2] + pm * optionvalue[i + 1] + pd * optionvalue[i]) * df
                
                if self.american == True:
                    optionvalue[i] = max(z * (self.S * (u ** max(i - j, 0)) * 
                                              (d ** (max((j - i), 0))) - self.K), optionvalue[i])
            
            if j == 1:
                returnvalue[1] = (optionvalue[2] - optionvalue[0]) / (self.S * u - self.S * d)
                returnvalue[2] = ((optionvalue[2] - optionvalue[1]) / 
                                  (self.S * u - self.S) - (optionvalue[1] - optionvalue[0]) / 
                                   (self.S - self.S * d )) / (0.5 * ((self.S * u) - (self.S * d)))                              
                returnvalue[3] = optionvalue[0]
                
        returnvalue[3] = (returnvalue[3] - optionvalue[0]) / dt / 365
        returnvalue[0] = optionvalue[0]
        
        if self.output_flag == 'price':
            result = returnvalue[0]
        if self.output_flag == 'delta':
            result = returnvalue[1]
        if self.output_flag == 'gamma':
            result = returnvalue[2]
        if self.output_flag == 'theta':
            result = returnvalue[3]
        if self.output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]),
                      'Theta = '+str(returnvalue[3]))
                               
        return result                     
    
    
    @timethis
    def implied_trinomial_tree(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                       steps_itt=None, option=None, output_flag=None, step=None, state=None, 
                       skew=None, timing=None):
        """
        Implied Trinomial Tree

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
        steps_itt : Int
            Number of time steps. The default is 10.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        output_flag : Str
            UPM: A matrix of implied up transition probabilities
            UPni: The implied up transition probability at a single node
            DPM: A matrix of implied down transition probabilities
            DPni: The implied down transition probability at a single node
            LVM: A matrix of implied local volatilities
            LVni: The local volatility at a single node
            ADM: A matrix of Arrow-Debreu prices at a single node
            ADni: The Arrow-Debreu price at a single node (at time step - step and state - state)
            price: The value of the European option
        step : Int
            Time step used for Arrow Debreu price at single node. The default is 5.
        state : Int
            State position used for Arrow Debreu price at single node. The default is 5.
        skew : Float
            Rate at which volatility increases (decreases) for every one point decrease 
            (increase) in the strike price. The default is 0.0004.

        Returns
        -------
        result : Various
            Depending on output flag:
                UPM: A matrix of implied up transition probabilities
                UPni: The implied up transition probability at a single node
                DPM: A matrix of implied down transition probabilities
                DPni: The implied down transition probability at a single node
                LVM: A matrix of implied local volatilities
                LVni: The local volatility at a single node
                ADM: A matrix of Arrow-Debreu prices at a single node
                ADni: The Arrow-Debreu price at a single node (at time step - step and state - state)
                price: The European option price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps_itt=steps_itt, 
                             option=option, output_flag=output_flag, step=step, 
                             state=state, skew=skew, timing=timing)
        
        if output_flag is None:
            self.temp_flag = self.output_flag
        else:
            self.temp_flag = output_flag
        
        z = 1
        if self.option == 'put':
            z = -1
        
        optionvaluenode = np.zeros((self.steps_itt * 2 + 1))
        arrowdebreu = np.zeros((self.steps_itt + 1, self.steps_itt * 2 + 1), dtype='float')
        upprob = np.zeros((self.steps_itt, self.steps_itt * 2 - 1), dtype='float')
        downprob = np.zeros((self.steps_itt, self.steps_itt * 2 - 1), dtype='float')
        localvol = np.zeros((self.steps_itt, self.steps_itt * 2 - 1), dtype='float')
        
        dt = self.T / self.steps_itt
        u = np.exp(self.sigma * np.sqrt(2 * dt))
        d = 1 / u
        df = np.exp(-self.r * dt)
        arrowdebreu[0, 0] = 1 
                
        for n in range(self.steps_itt):
            for i in range(n * 2 + 1):
                val = 0
                Si1 = self.S * (u ** (max(i - n, 0))) * (d ** (max(n * 2 - n - i, 0)))
                Si = Si1 * d
                Si2 = Si1 * u
                Fi = Si1 * np.exp(self.b * dt)
                sigmai = self.sigma + (self.S - Si1) * self.skew
                if i < (n * 2) / 2 + 1:
                    for j in range(i):
                        Fj = self.S * (u ** (max(j - n, 0))) * (d ** (max(n * 2 - n - j, 0))) * np.exp(self.b * dt)
                        val = val + arrowdebreu[n, j] * (Si1 - Fj)
                        
                    optionvalue = self.trinomial_tree(S=self.S, K=Si1, T=(n + 1)*dt, r=self.r, 
                                          q=0, sigma=sigmai, steps=n + 1, option='put', 
                                          output_flag='price', american=False)
        
                    qi = (np.exp(self.r * dt) * optionvalue - val) / (arrowdebreu[n, i] * (Si1 - Si))
                    pi = (Fi + qi * (Si1 - Si) - Si1) / (Si2 - Si1)
                else:
                    optionvalue = self.trinomial_tree(S=self.S, K=Si1, T=(n + 1) * dt, r=self.r, 
                                          q=0, sigma=sigmai, steps=n + 1, option='call', 
                                          output_flag='price', american=False)
                    val = 0
                    for j in range(i + 1, n * 2 + 1):
                        Fj = self.S * (u ** (max(j - n, 0))) * (d ** (max(n * 2 - n - j, 0))) * np.exp(self.b * dt)
                        val = val + arrowdebreu[n, j] * (Fj- Si1)
    
                    pi = (np.exp(self.r * dt) * optionvalue - val) / (arrowdebreu[n, i] * (Si2 - Si1))
                    qi = (Fi - pi * (Si2 - Si1) - Si1) / (Si - Si1)
                
                # Replacing negative probabilities    
                if pi < 0 or pi > 1 or qi < 0 or qi > 1:
                    if Fi > Si1 and Fi < Si2:
                        pi = 1 / 2 * ((Fi - Si1) / (Si2 - Si1) + (Fi - Si) / (Si2 - Si))
                        qi = 1 / 2 * ((Si2 - Fi) / (Si2 - Si))
                    elif Fi > Si and Fi < Si1:
                        pi = 1 / 2 * ((Fi - Si) / (Si2 - Si))
                        qi = 1 / 2 * ((Si2 - Fi) / (Si2 - Si1) + (Si1 - Fi) / (Si1 - Si))
    
                downprob[n, i] = qi
                upprob[n, i] = pi
                
                # Calculating local volatilities
                Fo = (pi * Si2 + qi * Si + (1 - pi -qi) * Si1)
                localvol[n, i] = np.sqrt((pi * (Si2 - Fo) ** 2 + (1 - pi - qi) * (Si1 - Fo) ** 2 + qi * (Si - Fo) ** 2) / (Fo ** 2 * dt))
        
                # Calculating Arrow-Debreu prices
                if n == 0:
                    arrowdebreu[n + 1, i] = qi * arrowdebreu[n, i] * df
                    arrowdebreu[n + 1, i + 1] = (1 - pi - qi) * arrowdebreu[n, i] * df
                    arrowdebreu[n + 1, i + 2] = pi * arrowdebreu[n, i] * df
                elif n > 0 and i == 0:
                    arrowdebreu[n + 1, i] = qi * arrowdebreu[n, i] * df
                elif n > 0 and i == n * 2:
                    arrowdebreu[n + 1, i] = (upprob[n, i - 2] * arrowdebreu[n, i - 2] * df + 
                                             (1 - upprob[n, i - 1] - downprob[n, i - 1]) * 
                                             arrowdebreu[n, i - 1] * df + qi * arrowdebreu[n, i] * df)
                    arrowdebreu[n + 1, i + 1] = (upprob[n, i - 1] * arrowdebreu[n, i - 1] * 
                                                 df + (1 - pi - qi) * arrowdebreu[n, i] * df)
                    arrowdebreu[n + 1, i + 2] = pi * arrowdebreu[n, i] * df
                elif n > 0 and i == 1:
                    arrowdebreu[n + 1, i] = ((1 - upprob[n, i - 1] - downprob[n, i - 1]) * 
                                             arrowdebreu[n, i - 1] * df + qi * arrowdebreu[n, i] * df)
                else:
                    arrowdebreu[n + 1, i] = (upprob[n, i - 2] * arrowdebreu[n, i - 2] * df + 
                                             (1 - upprob[n, i - 1] - downprob[n, i - 1]) * 
                                             arrowdebreu[n, i - 1] * df + qi * arrowdebreu[n, i] * df)
    
        self.output_flag = self.temp_flag
    
        if self.output_flag == 'UPM':    
            result = upprob
        elif self.output_flag == 'UPni':    
            result = upprob[self.step, self.state]        
        elif self.output_flag == 'DPM':
            result = downprob
        elif self.output_flag == 'DPni':    
            result = downprob[self.step, self.state]
        elif self.output_flag == 'LVM': 
            result = localvol
        elif self.output_flag == 'LVni':
            result = localvol[self.step, self.state]
        elif self.output_flag == 'ADM':    
            result = arrowdebreu
        elif self.output_flag == 'ADni':    
            result = arrowdebreu[self.step, self.state]
        elif self.output_flag == 'price':
            
            # Calculation of option price using the implied trinomial tree
            for i in range(2 * self.steps_itt + 1):
                optionvaluenode[i] = max(0, z * (self.S * (u ** max(i - self.steps_itt, 0)) * 
                                                 (d ** (max((self.steps_itt - i), 0))) - self.K))    
    
            for n in range(self.steps_itt - 1, -1, -1):
                for i in range(n * 2 + 1):
                    optionvaluenode[i] = ((upprob[n, i] * optionvaluenode[i + 2] + 
                                          (1 - upprob[n, i] - downprob[n, i]) * optionvaluenode[i + 1] + 
                                          downprob[n, i] * optionvaluenode[i]) * df)
    
            result = optionvaluenode[0] * 1000000         
                               
        return result    
    
    
    @timethis
    def explicit_finite_difference(self, S=None, K=None, T=None, r=None, q=None, 
                                   sigma=None, nodes=None, option=None, american=None, 
                                   timing=None):
        """
        Explicit Finite Difference

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
        nodes : Int
            Number of price steps. The default is 100.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Float
            Option Price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, nodes=nodes, 
                             option=option, american=american, timing=timing)
        
        z = 1
        if self.option == 'put':
            z = -1
        
        dS = self.S / self.nodes
        self.nodes = int(self.K / dS) * 2
        St = np.zeros((self.nodes + 2), dtype='float')
        
        SGridtPt = int(self.S / dS)
        dt = (dS ** 2) / ((self.sigma ** 2) * 4 * (self.K ** 2))
        N = int(self.T / dt) + 1
        
        C = np.zeros((N + 1, self.nodes + 2), dtype='float')
        dt = self.T / N
        Df = 1 / (1 + self.r * dt)
          
        for i in range(self.nodes + 1):
            St[i] = i * dS # Asset price at maturity
            C[N, i] = max(0, z * (St[i] - self.K) ) # At maturity
            
        for j in range(N - 1, -1, -1):
            for i in range(1, self.nodes):
                pu = 0.5 * ((self.sigma ** 2) * (i ** 2) + self.b * i) * dt
                pm = 1 - (self.sigma ** 2) * (i ** 2) * dt
                pd = 0.5 * ((self.sigma ** 2) * (i ** 2) - self.b * i) * dt
                C[j, i] = Df * (pu * C[j + 1, i + 1] + pm * C[j + 1, i] + pd * C[j + 1, i - 1])
                if self.american == True:
                    C[j, i] = max(z * (St[i] - self.K), C[j, i])
                    
                if z == 1: # Call option
                    C[j, 0] = 0
                    C[j, self.nodes] = (St[i] - self.K)
                else:
                    C[j, 0] = self.K
                    C[j, self.nodes] = 0
        
        result = C[0, SGridtPt]
    
        return result          
    
    
    @timethis
    def implicit_finite_difference(self, S=None, K=None, T=None, r=None, q=None, 
                                   sigma=None, steps=None, nodes=None, option=None, 
                                   american=None, timing=None):
        """
        Implicit Finite Difference
        # Slow to converge - steps has small effect, need nodes 3000+
 
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
        steps : Int
            Number of time steps. The default is 1000.
        nodes : Float
            Number of price steps. The default is 100.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Float
            Option Price.
        

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             nodes=nodes, option=option, american=american, timing=timing)
        
        z = 1
        if self.option == 'put':
            z = -1
        
        # Make sure current asset price falls at grid point
        dS = 2 * self.S / self.nodes
        SGridtPt = int(self.S / dS)
        self.nodes = int(self.K / dS) * 2
        dt = self.T / self.steps
        
        CT = np.zeros(self.nodes + 1)
        p = np.zeros((self.nodes + 1, self.nodes + 1), dtype='float')
        
        for j in range(self.nodes + 1):
            CT[j] = max(0, z * (j * dS - self.K)) # At maturity
            for i in range(self.nodes + 1):
                p[j, i] = 0
                
        p[0, 0] = 1
        for i in range(1, self.nodes):
            p[i, i - 1] = 0.5 * i * (self.b - (self.sigma ** 2) * i) * dt
            p[i, i] = 1 + (self.r + (self.sigma ** 2) * (i ** 2)) * dt
            p[i, i + 1] = 0.5 * i * (-self.b - (self.sigma ** 2) * i) * dt
            
        p[self.nodes, self.nodes] = 1
        
        self.C = np.matmul(np.linalg.inv(p), CT.T)
        
        for j in range(self.steps - 1, 0, -1):
            self.C = np.matmul(np.linalg.inv(p), self.C)
            
            if self.american == True: # American option
                for i in range(1, self.nodes + 1):
                    self.C[i] = max(float(self.C[i]), z * ((i - 1) * dS - self.K))
                
        result = self.C[SGridtPt + 1]
        
        return result   
    
    
    @timethis
    def explicit_finite_difference_lns(self, S=None, K=None, T=None, r=None, q=None, 
                                       sigma=None, steps=None, nodes=None, option=None, 
                                       american=None, timing=None):
        """
        Explicit Finite Differences - rewrite BS-PDE in terms of ln(S)

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
        steps : Int
            Number of time steps. The default is 1000.
        nodes : Float
            Number of price steps. The default is 100.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Float
            Option Price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             nodes=nodes, option=option, american=american, timing=timing)
          
        z = 1
        if self.option == 'put':
            z = -1
        
        dt = self.T / self.steps
        dx = self.sigma * np.sqrt(3 * dt)
        pu = 0.5 * dt * (((self.sigma / dx) ** 2) + (self.b - (self.sigma ** 2) / 2) / dx)
        pm = 1 - dt * ((self.sigma / dx) ** 2) - self.r * dt
        pd = 0.5 * dt * (((self.sigma / dx) ** 2) - (self.b - (self.sigma ** 2) / 2) / dx)
        St = {}
        St[0] = self.S * np.exp(-self.nodes / 2 * dx)
        C = np.zeros((int(self.nodes / 2) + 1, self.nodes + 2), dtype='float')
        C[self.steps, 0] = max(0, z * (St[0] - self.K))
        
        for i in range(1, self.nodes + 1):
            St[i] = St[i - 1] * np.exp(dx) # Asset price at maturity
            C[self.steps, i] = max(0, z * (St[i] - self.K) ) # At maturity
        
        for j in range(self.steps - 1, -1, -1):
            for i in range(1, self.nodes):
                C[j, i] = pu * C[j + 1, i + 1] + pm * C[j + 1, i] + pd * C[j + 1, i - 1]
                if self.american == True:
                    C[j, i] = max(C[j, i], z * (St[i] - self.K))
                    
                C[j, self.nodes] = C[j, self.nodes - 1] + St[self.nodes] - St[self.nodes - 1] # Upper boundary
                C[j, 0] = C[j, 1] # Lower boundary
           
        result = C[0, int(self.nodes / 2)]
    
        return result   
    
    
    @timethis
    def crank_nicolson(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                       steps=None, nodes=None, option=None, american=None, timing=None):
        """
        Crank Nicolson

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
        steps : Int
            Number of time steps. The default is 1000.
        nodes : Float
            Number of price steps. The default is 100.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        american : Bool
            Whether the option is American. The default is False.

        Returns
        -------
        result : Float
            Option Price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                             nodes=nodes, option=option, american=american, timing=timing)
                
        z = 1
        if self.option == 'put':
            z = -1
        
        dt = self.T / self.steps
        dx = self.sigma * np.sqrt(3 * dt)
        pu = -0.25 * dt * (((self.sigma / dx) ** 2) + (self.b - (self.sigma ** 2) / 2) / dx)
        pm = 1 + 0.5 * dt * ((self.sigma / dx) ** 2) + 0.5 * self.r * dt
        pd = -0.25 * dt * (((self.sigma / dx) ** 2) - (self.b - (self.sigma ** 2) / 2) / dx)
        St = np.zeros(self.nodes + 2)
        pmd = np.zeros(self.nodes + 1)
        p = np.zeros(self.nodes + 1)
        St[0] = self.S * np.exp(-self.nodes / 2 * dx)
        C = np.zeros((int(self.nodes / 2) + 2, self.nodes + 2), dtype='float')
        C[0, 0] = max(0, z * (St[0] - self.K))
        
        for i in range(1, self.nodes + 1):
            St[i] = St[i - 1] * np.exp(dx) # Asset price at maturity
            C[0, i] = max(0, z * (St[i] - self.K)) # At maturity
        
        pmd[1] = pm + pd
        p[1] = (-pu * C[0, 2] - (pm - 2) * C[0, 1] - 
                pd * C[0, 0] - pd * (St[1] - St[0]))
        
        for j in range(self.steps - 1, -1, -1):
            for i in range(2, self.nodes):
                p[i] = (-pu * C[0, i + 1] - (pm - 2) * C[0, i] - 
                        pd * C[0, i - 1] - p[i - 1] * pd / pmd[i - 1])
                pmd[i] = pm - pu * pd / pmd[i - 1]
    
            for i in range(self.nodes - 2, 0, -1):
                C[1, i] = (p[i] - pu * C[1, i + 1]) / pmd[i]
                
                for i in range(self.nodes + 1):
                    C[0, i] = C[1, i]
                    if self.american == True:
                        C[0, i] = max(C[1, i], z * (St[i] - self.K))
           
        result = C[0, int(self.nodes / 2)]
    
        return result   
    
    
    @timethis
    def european_monte_carlo(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                             simulations=None, option=None, timing=None):
        """
        Standard Monte Carlo

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
        simulations : Int
            Number of Monte Carlo runs. The default is 10000.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        result : Float
            Option Price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, simulations=simulations, 
                             option=option, timing=timing)
         
        Drift = (self.b - (self.sigma ** 2) / 2) * self.T
        sigmarT = self.sigma * np.sqrt(self.T)
        val = 0
        
        z = 1
        if self.option == 'put':
            z = -1
        
        for i in range(1, self.simulations + 1):
            St = self.S * np.exp(Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
            val = val + max(z * (St - self.K), 0) 
            
        result = np.exp(-self.r * self.T) * val / self.simulations
        
        return result
    
    
    @timethis
    def european_monte_carlo_with_greeks(self, S=None, K=None, T=None, r=None, q=None, 
                                         sigma=None, simulations=None, option=None, 
                                         output_flag=None, timing=None):
        """
        Standard Monte Carlo with Greeks

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
        simulations : Int
            Number of Monte Carlo runs. The default is 10000.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.
        output_flag : Str
            Whether to return 'price', 'delta', 'gamma', 'theta', 'vega' or 'all'. 
            The default is 'price'.

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'vega' : Float; Option Vega
                'all' : Tuple; Option Price, Option Delta, Option Gamma, Option Theta, 
                               Option Vega  

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, simulations=simulations, 
                             option=option, output_flag=output_flag, timing=timing)
                
        Drift = (self.b - (self.sigma ** 2) / 2) * self.T
        sigmarT = self.sigma * np.sqrt(self.T)
        val = 0
        deltasum = 0
        gammasum = 0
        output = {}
        
        z = 1
        if self.option == 'put':
            z = -1
        
        for i in range(1, self.simulations + 1):
            St = self.S * np.exp(Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
            val = val + max(z * (St - self.K), 0) 
            if z == 1 and St > self.K:
                deltasum = deltasum + St
            if z == -1 and St < K:
                deltasum = deltasum + St
            if abs(St - self.K) < 2:
                gammasum = gammasum + 1
                
        # Option Value
        output[0] = np.exp(-self.r * self.T) * val / self.simulations       
            
        # Delta
        output[1] = np.exp(-self.r * self.T) * deltasum / (self.simulations * self.S)
        
        # Gamma
        output[2] = np.exp(-self.r * self.T) * ((self.K / self.S) ** 2) * gammasum / (4 * self.simulations)
        
        # Theta
        output[3] = (self.r * output[0] - self.b * self.S * output[1] - 0.5 * 
                     (self.sigma ** 2) * (self.S ** 2) * output[2]) / 365
        
        # Vega
        output[4] = output[2] * self.sigma * (self.S ** 2) * self.T / 100
    
        if self.output_flag == 'price':
            result = output[0]
        if self.output_flag == 'delta':
            result = output[1]
        if self.output_flag == 'gamma':
            result = output[2]
        if self.output_flag == 'theta':
            result = output[3]
        if self.output_flag == 'vega':
            result = output[4]
        if self.output_flag == 'all':
            result = ('Price = '+str(output[0]),
                      'Delta = '+str(output[1]),
                      'Gamma = '+str(output[2]),
                      'Theta = '+str(output[3]),
                      'Vega = '+str(output[4]))
                
        return result
    
    
    @timethis
    def hull_white_87(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                      vvol=None, option=None, timing=None):
        """
        Hull White 1987 - Uncorrelated Stochastic Volatility.

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
        vvol : Float
            Vol of vol. The default is 0.5.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        result : Float
            Option price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sigma=sigma, vvol=vvol, option=option, 
                             timing=timing)
        
        k = self.vvol ** 2 * self.T
        ek = np.exp(k)
        
        d1 = (np.log(self.S / self.K) + (self.b + (self.sigma ** 2) / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
           
        cgbs = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                         q=self.q, sigma=self.sigma, option='call')
        
        # Partial Derivatives
        cVV = (self.S * np.exp((self.b - self.r) * self.T) * np.sqrt(self.T) * Nd1 * 
               (d1 * d2 - 1) / (4 * (self.sigma ** 3)))
        cVVV = (self.S * np.exp((self.b - self.r) * self.T) * np.sqrt(self.T) * Nd1 * 
                ((d1 * d2 - 1) * (d1 * d2 - 3) - ((d1 ** 2) + (d2 ** 2))) / (8 * (self.sigma ** 5)))                                                             
        
        callvalue = (cgbs + 1 / 2 * cVV * (2 * self.sigma ** 4 * (ek - k - 1) / k ** 2 - self.sigma  ** 4) + 
                     1 / 6 * cVVV * self.sigma ** 6 * (ek ** 3 - (9 + 18 * k) * ek + 8 + 24 * k + 
                                                  18 * k ** 2 + 6 * k ** 3) / (3 * k ** 3))
        
        if self.option == 'call':
            result = callvalue
        if self.option == 'put': # use put-call parity to find put
            result = callvalue - self.S * np.exp((self.b - self.r) * self.T) + self.K * np.exp(-self.r * self.T)
            
        return result


    @timethis
    def hull_white_88(self, S=None, K=None, T=None, r=None, q=None, sig0=None, sigLR=None,
                      halflife=None, vvol=None, rho=None, option=None, timing=None):
        """
        Hull White 1988 - Correlated Stochastic Volatility.

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
        sig0 : Float
            Initial Volatility. The default is 0.09 (9%).
        sigLR : Float
            Long run mean reversion level of volatility. The default is 0.0625 (6.25%).
        halflife : Float
            Half-life of volatility deviation. The default is 0.1. 
        vvol : Float
            Vol of vol. The default is 0.5.
        rho : Float
            Correlation between asset price and volatility. The default is 0.
        option : Str
            Type of option. 'put' or 'call'. The default is 'call'.

        Returns
        -------
        result : Float
            Option price.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, sig0=sig0, sigLR=sigLR, halflife=halflife, 
                             vvol=vvol, rho=rho, option=option, timing=timing)

        beta = -np.log(2) / self.halflife # Find constant, beta, from Half-life
        a = -beta * (self.sigLR ** 2) # Find constant, a, from long run volatility
        delta = beta * self.T
        ed = np.exp(delta)
        v = self.sig0 ** 2

        # Average expected variance
        if abs(beta) < 0.0001:
            vbar = v + 0.5 * a * self.T 
        else:
            vbar = (v + (a / beta)) * ((ed - 1) / delta) - (a / beta)
            
        d1 = (np.log(self.S / self.K) + (self.b + (vbar / 2)) * self.T) / np.sqrt(vbar * self.T)
        d2 = d1 - np.sqrt(vbar * self.T)
        
        # standardised normal density function
        nd1 = (1 / np.sqrt(2 * np.pi)) * (np.exp(-d1 ** 2 * 0.5))
        
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        
        # Partial derivatives
        cSV = -self.S * np.exp((self.b - self.r) * self.T) * nd1 * (d2 / (2 * vbar))
        cVV = ((self.S * np.exp((self.b - self.r) * self.T) * nd1 * np.sqrt(self.T) / 
               (4 * vbar ** 1.5)) * (d1 * d2 - 1))
        cSVV = ((self.S * np.exp((self.b - self.r) * self.T) / (4 * vbar ** 2)) * 
                nd1 * ((-d1 * (d2 ** 2)) + d1 + (2 * d2)))                      
        cVVV = (((self.S * np.exp((self.b - self.r) * self.T) * nd1 * np.sqrt(self.T)) / 
                (8 * vbar ** 2.5)) * ((d1 * d2 - 1) * (d1 * d2 - 3) - ((d1 ** 2) + (d2 ** 2)))) 

        if abs(beta) < 0.0001:
            f1 = self.rho * ((a * self.T / 3) + v) * (self.T / 2) * cSV
            phi1 = (self.rho ** 2) * ((a * self.T / 4) + v) * ((self.T ** 3) / 6)
            phi2 = (2 + (1 / (self.rho ** 2))) * phi1
            phi3 = (self.rho ** 2) * (((a * self.T / 3) + v) ** 2) * ((self.T ** 4) / 8)
            phi4 = 2 * phi3
        else: # Beta different from zero
            phi1 = (((self.rho ** 2) / (beta ** 4)) * 
                    (((a + (beta * v)) * 
                    ((ed * (((delta ** 2) / 2) - delta + 1)) - 1)) + 
                    (a * ((ed * (2 - delta)) - (2 + delta)))))
            phi2 = ((2 * phi1) + 
                    ((1 / (2 * (beta ** 4))) * 
                    (((a + (beta * v)) * ((ed ** 2) - (2 * delta * ed) - 1)) - 
                    ((a / 2) * ((ed ** 2) - (4 * ed) + (2 * delta) + 3)))))
            phi3 = (((self.rho ** 2) / (2 * (beta ** 6))) * 
                    ((((a + (beta * v)) * (ed - delta * ed - 1)) - (a * (1 + delta - ed))) ** 2))
            phi4 = 2 * phi3
            f1 = ((self.rho / ((beta ** 3) * self.T)) * 
                  (((a + (beta * v)) * (1 - ed + (delta * ed))) + 
                   (a * (1 + delta - ed))) * cSV)

        f0 = self.S * np.exp((self.b - self.r) * self.T) * Nd1 - self.K * np.exp(-self.r * self.T) * Nd2
        f2 = (((phi1 / self.T) * cSV) + ((phi2 / (self.T ** 2)) * cVV) + 
              ((phi3 / (self.T ** 2)) * cSVV) + ((phi4 / (self.T ** 3)) * cVVV))
        
        callvalue = f0 + f1 * self.vvol + f2 * self.vvol ** 2
        
        if self.option == 'call':
            result = callvalue
        if self.option == 'put':
            result = callvalue - (self.S * np.exp((self.b - self.r) * self.T)) + (self.K * np.exp(-self.r * self.T))
        
        return result



class ImpliedVol(Pricer):
    
    def __init__(self):
        super().__init__(self) # Inherit methods from Pricer class

    
    @timethis
    def implied_vol_newton_raphson(self, S=None, K=None, T=None, r=None, q=None, 
                                   cm=None, epsilon=None, option=None, timing=None):
        """
        Finds implied volatility using Newton-Raphson method - needs knowledge of 
        partial derivative of option pricing formula with respect to volatility (vega)

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
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, option=option, 
                             timing=timing)
                
        # Manaster and Koehler seed value
        vi = np.sqrt(abs(np.log(self.S / self.K) + self.r * self.T) * 2 / self.T)
        ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, q=self.q, 
                                       sigma=vi, option=self.option, timing=False)    
        vegai = self.black_scholes_merton_vega(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)
        mindiff = abs(self.cm - ci)
    
        while abs(self.cm - ci) >= self.epsilon and abs(self.cm - ci) <= mindiff:
            vi = vi - (ci - self.cm) / vegai
            ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                           q=self.q, sigma=vi, option=self.option, 
                                           timing=False)
            vegai = self.black_scholes_merton_vega(S=self.S, K=self.K, T=self.T, 
                                                   r=self.r, q=self.q, sigma=vi, 
                                                   option=self.option, timing=False)
            mindiff = abs(self.cm - ci)
            
        if abs(self.cm - ci) < self.epsilon:
            result = vi
        else:
            result = 'NA'
        
        return result
    
    
    @timethis
    def implied_vol_bisection(self, S=None, K=None, T=None, r=None, q=None, cm=None, 
                              epsilon=None, option=None, timing=None):
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


        Returns
        -------
        result : Float
            Implied Volatility.

        """
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, option=option, 
                             timing=timing)
        
        vLow = 0.005
        vHigh = 4
        cLow = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                         q=self.q, sigma=vLow, option=self.option, 
                                         timing=False)
        cHigh = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                          q=self.q, sigma=vHigh, option=self.option, 
                                          timing=False)
        counter = 0
        
        vi = vLow + (self.cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
        
        while abs(self.cm - self.black_scholes_merton(S=self.S, K=self.K, T=self.T, 
                                                      r=self.r, q=self.q, sigma=vi, 
                                                      option=self.option, timing=False)) > self.epsilon:
            counter = counter + 1
            if counter == 100:
                result = 'NA'
            
            if self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                         q=self.q, sigma=vi, option=self.option, 
                                         timing=False) < self.cm:
                vLow = vi
            else:
                vHigh = vi
            
            cLow = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                             q=self.q, sigma=vLow, option=self.option, 
                                             timing=False)
            cHigh = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                              q=self.q, sigma=vHigh, option=self.option, 
                                              timing=False)
            vi = vLow + (self.cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
            
        result = vi    
            
        return result
   
    
    @timethis
    def implied_vol_naive(self, S=None, K=None, T=None, r=None, q=None, cm=None, 
                               epsilon=None, option=None, timing=None):
        """
        Finds implied volatility using simple naive iteration, increasing precision 
        each time the difference changes sign.

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
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, option=option, 
                             timing=timing)
        
        # Seed vol
        vi = 0.2
        
        # Calculate starting option price using this vol
        ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                       q=self.q, sigma=vi, option=self.option, timing=False)
        
        # Initial price difference
        price_diff = self.cm - ci
        
        if price_diff > 0:
            flag = 1
        else:
            flag = -1
        
        # Starting vol shift size
        shift = 0.01
        
        price_diff_start = price_diff
        
        while abs(price_diff) > self.epsilon:
            
            # If the price difference changes sign after the vol shift, reduce 
            # the decimal by one and reverse the sign
            if np.sign(price_diff) != np.sign(price_diff_start):
                shift = shift * -0.1                
            
            # Calculate new vol
            vi += (shift * flag)
            
            # Set initial price difference
            price_diff_start = price_diff
            
            # Calculate the option price with new vol
            ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                           q=self.q, sigma=vi, option=self.option, 
                                           timing=False)
            
            # Price difference after shifting vol
            price_diff = self.cm - ci
            
            # If values are diverging reverse the shift sign
            if abs(price_diff) > abs(price_diff_start):
                shift = -shift
       
        result = vi    
            
        return result
    
    
    @timethis
    def implied_vol_naive_verbose(self, S=None, K=None, T=None, r=None, q=None, cm=None, 
                                  epsilon=None, option=None, timing=None):
        """
        Finds implied volatility using simple naive iteration, increasing precision 
        each time the difference changes sign.

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
        
        self._refresh_params(S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, option=option, 
                             timing=timing)
        
        vi = 0.2
        ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                       q=self.q, sigma=vi, option=self.option, timing=False)
        price_diff = self.cm - ci
        if price_diff > 0:
            flag = 1
        else:
            flag = -1
        while abs(price_diff) > self.epsilon:
            while price_diff * flag > 0:
                ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)
                price_diff = self.cm - ci
                vi += (0.01 * flag)
            
            while price_diff * flag < 0:
                ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)

                price_diff = self.cm - ci
                vi -= (0.001 * flag)
            
            while price_diff * flag > 0:
                ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)

                price_diff = self.cm - ci
                vi += (0.0001 * flag)
            
            while price_diff * flag < 0:
                ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)

                price_diff = self.cm - ci
                vi -= (0.00001 * flag)
                
            while price_diff * flag > 0:
                ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)

                price_diff = self.cm - ci
                vi += (0.000001 * flag)
            
            while price_diff * flag < 0:
                ci = self.black_scholes_merton(S=self.S, K=self.K, T=self.T, r=self.r, 
                                               q=self.q, sigma=vi, option=self.option, 
                                               timing=False)

                price_diff = self.cm - ci
                vi -= (0.0000001 * flag)    
        
        result = vi    
            
        return result


class SABRVolatility(Pricer):
    """
    Stochastic, Alpha, Beta, Rho model
    
    Extension of Black 76 model to include an easily implementable stochastic volatility model
    
    Beta will typically be chosen a priori according to how traders observe market prices:
        e.g. In FX markets, standard to assume lognormal terms, Beta = 1
             In some Fixed Income markets traders prefer to assume normal terms, Beta = 0
    
    Alpha will need to be calibrated to ATM volatility         
             
    """
    
    def __init__(self, F=sabr_df_dict['df_F'], X=sabr_df_dict['df_X'], T=sabr_df_dict['df_T'], 
                 r=sabr_df_dict['df_r'], atmvol=sabr_df_dict['df_atmvol'], beta=sabr_df_dict['df_beta'], 
                 volvol=sabr_df_dict['df_volvol'], rho=sabr_df_dict['df_rho'], option=sabr_df_dict['df_option'], 
                 sabr_df_dict=sabr_df_dict):
        super().__init__(self) # Inherit methods from Pricer class
        self.F = F # Forward price
        self.X = X # Strike price
        self.T = T # Time to maturity
        self.r = r # Interest rate
        self.atmvol = atmvol # To be calibrated to Black 76 At The Money volatility
        self.beta = beta # Normal or Lognormal Stochastic Volatility
        self.volvol = volvol # Volatility of volatility
        self.rho = rho # Correlation between volatility and underlying asset
        self.option = option # Option type, call or put
        self.refresh = False # Whether to refresh parameters, set to False if called from another function
        self.sabr_df_dict = sabr_df_dict # Dictionary of default SABR parameters
    
    
    @timethis
    def price(self, option=None, timing=None):
        
        if option is None:
            option = self.option
        else:
            self.option=option
        
        if timing is None:
            timing = self.timing
        else:
            self.timing = timing    
            
        
        return self.black_76(F=self.F, K=self.X, T=self.T, r=self.r, sigma=self.black_vol, 
                             option=self.option, timing=self.timing) 
    
    
    @timethis
    def calibrate(self, timing=None):
        """
        Run the SABR calibration

        Returns
        -------
        Float
            Black-76 equivalent SABR volatility.

        """
        
        if timing is None:
            timing = self.timing
        else:
            self.timing = timing
        
        self.black_vol = self._alpha_sabr(self._find_alpha())
        
        return self.black_vol
    
    
    def _alpha_sabr(self, alpha):
        """
        The SABR skew vol function

        Parameters
        ----------
        Alpha : Float
            Alpha value.

        Returns
        -------
        result : Float
            Black-76 equivalent SABR volatility.

        """
                        
        dSABR = np.zeros(4)
        dSABR[1] = (alpha / ((self.F * self.X) ** ((1 - self.beta) / 2) * (1 + (((1 - self.beta) ** 2) / 24) * 
                    (np.log(self.F / self.X) ** 2) + ((1 - self.beta) ** 4 / 1920) * (np.log(self.F / self.X) ** 4))))
        
        if abs(self.F - self.X) > 10 ** -8:
            sabrz = (self.volvol / alpha) * (self.F * self.X) ** ((1 - self.beta) / 2) * np.log(self.F / self.X)
            y = (np.sqrt(1 - 2 * self.rho * sabrz + sabrz ** 2) + sabrz - self.rho) / (1 - self.rho)
            if abs(y - 1) < 10 ** -8:
                dSABR[2] = 1
            elif y > 0:
                dSABR[2] = sabrz / np.log(y)
            else:
                dSABR[2] = 1
        else:
            dSABR[2] = 1
            
        dSABR[3] = (1 + ((((1 - self.beta) ** 2 / 24) * alpha ** 2 / ((self.F * self.X) ** (1 - self.beta))) + 
                         0.25 * self.rho * self.beta * self.volvol * alpha / ((self.F * self.X) ** ((1 - self.beta) / 2)) + 
                         (2 - 3 * self.rho ** 2) * self.volvol ** 2 / 24) * self.T)
        
        result = dSABR[1] * dSABR[2] * dSABR[3]
        
        return result
    
    
    def _find_alpha(self):
        """
        Find alpha feeding values to _cube_root method.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        # Alpha is a function of atm vol etc
        
        self.alpha = self._cube_root(((1 - self.beta) ** 2 * self.T / (24 * self.F ** (2 - 2 * self.beta))), 
                                     (0.25 * self.rho * self.volvol * self.beta * self.T / self.F ** (1 - self.beta)), 
                                     (1 + (2 - 3 * self.rho ** 2) / 24 * self.volvol ** 2 * self.T), 
                                     (-self.atmvol * self.F ** (1 - self.beta)))
        
        return self.alpha
    
    
    def _cube_root(self, cubic, quadratic, linear, constant):
        """
        Finds the smallest positive root of the input cubic polynomial algorithm 
        from Numerical Recipes

        Parameters
        ----------
        cubic : Float
            3rd order term of input polynomial.
        quadratic : Float
            2nd order term of input polynomial.
        linear : Float
            Linear term of input polynomial.
        constant : Float
            Constant term of input polynomial.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        a = quadratic / cubic
        b = linear / cubic
        C = constant / cubic
        Q = (a ** 2 - 3 * b) / 9
        r = (2 * a ** 3 - 9 * a * b + 27 * C) / 54
        roots = np.zeros(4)
        
        if r ** 2 - Q ** 3 >= 0:
            cap_A = -np.sign(r) * (abs(r) + np.sqrt(r ** 2 - Q ** 3)) ** (1 / 3)
            if cap_A == 0:
                cap_B = 0
            else:
                cap_B = Q / cap_A
            result = cap_A + cap_B - a / 3
        else:
            theta = self._arccos(r / Q ** 1.5)
            
            # The three roots
            roots[1] = - 2 * np.sqrt(Q) * math.cos(theta / 3) - a / 3
            roots[2] = - 2 * np.sqrt(Q) * math.cos(theta / 3 + 2.0943951023932) - a / 3
            roots[3] = - 2 * np.sqrt(Q) * math.cos(theta / 3 - 2.0943951023932) - a / 3
            
            # locate that one which is the smallest positive root
            # assumes there is such a root (true for SABR model)
            # there is always a small positive root
            
            if roots[1] > 0:
                result = roots[1]
            elif roots[2] > 0:
                result = roots[2]
            elif roots[3] > 0:
                result = roots[3]
        
            if roots[2] > 0 and roots[2] < result:
                result = roots[2]
            
            if roots[3] > 0 and roots[3] < result:
                result = roots[3]
                
        return result
    
    
    def _arccos(self, y):
        """
        Inverse Cosine method

        Parameters
        ----------
        y : Float
            Input value.

        Returns
        -------
        result : Float
            Arc Cosine of input value.

        """
        result = np.arctan(-y / np.sqrt(-y * y + 1)) + 2 * np.arctan(1)
        
        return result



class Tools():
    
    def __init__(self, timing=False):
        self.timing = timing
   
   
    @timethis
    def cholesky_decomposition(self, R, timing=None):
        """
        Cholesky Decomposition.
        Return M in M * M.T = R where R is a symmetric positive definite correlation matrix

        Parameters
        ----------
        R : Array
            Correlation matrix.

        Returns
        -------
        M : Array
            Matrix decomposition.

        """
        if timing is None:
            timing = self.timing
        else:
            self.timing = timing    
                
        # Number of columns in input correlation matrix R
        n = len(R[0])
        
        a = np.zeros((n + 1, n + 1))
        M = np.zeros((n + 1, n + 1))
        
        for i in range(n + 1):
            for j in range(n + 1):
                a[i, j] = R[i, j]
                M[i, j] = 0
                
        for i in range(n + 1):
            for j in range(n + 1):
                U = a[i, j]
            for h in range(1, i):
                U = U - M[i, h] * M[j, h]
            if j == 1:
                M[i, i] = np.sqrt(U)
            else:
                M[j, i] = U / M[i, i]
        
        return M        


    def _n_choose_r(self, n, r):
        """
        Binomial Coefficients. n choose r
        Number of ways to choose an (unordered) subset of r elements from a fixed 
        set of n elements.

        Parameters
        ----------
        n : Int
            Set of elements.
        r : Int
            Subset of elements.

        Returns
        -------
        Int
            Binomial coefficient.

        """
        
        # Due to symmetry of the binomial coefficient, set r to optimise calculation
        r = min(r, n-r)
        
        # Numerator is the descending product from n to n+1-r
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        
        # Denominator is the product from 1 to r
        denom = reduce(op.mul, range(1, r+1), 1)
        
        # Binomial coefficient is calculated by dividing these two. 
        return numer // denom  # or / in Python 2


