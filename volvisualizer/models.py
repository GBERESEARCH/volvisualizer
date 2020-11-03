import math
import random
import time
import numpy as np
import operator as op
import scipy.stats as si
from functools import reduce, wraps
from operator import itemgetter
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
                print('{}.{} : {} milliseconds'.format(
                    func.__module__, func.__name__, round((
                        end - start)*1e3, 2)))
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
           'df_params_list':['S', 'F', 'K', 'T', 'r', 'q', 'sigma', 'option', 
                             'steps', 'steps_itt', 'nodes', 'vvol', 
                             'simulations', 'output_flag', 'american', 
                             'step', 'state', 'skew', 'sig0', 'sigLR', 
                             'halflife', 'rho', 'cm', 'epsilon', 'timing']}


sabr_df_dict = {'df_F':100,
                'df_K':70,
                'df_T':0.5,
                'df_r':0.05,
                'df_sigma':0.3, 
                'df_beta':0.9999, 
                'df_volvol':0.5, 
                'df_rho':-0.4,
                'df_option':'put',
                'df_timing':False, 
                'df_output_flag':'price'}


class Pricer():
    
    def __init__(
            self, 
            df_params_list=df_dict['df_params_list'], 
            df_dict=df_dict):
        
                
        # List of default parameters
        self.df_params_list = df_params_list 
        
        # Dictionary of default parameters
        self.df_dict = df_dict 
        
               
    def _refresh_params(self, **kwargs):
        """
        Set parameters for use in various pricing functions

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object
                v = df_dict['df_'+str(k)]
                self.__dict__[k] = v
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return self        
    

    def _refresh_params_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions to the
        default values.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object 
                # and to input dictionary
                v = self.df_dict['df_'+str(k)]
                self.__dict__[k] = v
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs        
    
    
    @timethis
    def black_scholes_merton(self, S=None, K=None, T=None, r=None, q=None, 
                             sigma=None, option=None, timing=None, 
                             default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.    

        Returns
        -------
        opt_price : Float
            Option Price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                    timing=timing))
        
        b = r - q
        carry = np.exp((b - r) * T)
        d1 = ((np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) 
              / (sigma * np.sqrt(T)))
        d2 = ((np.log(S / K) + (b - (0.5 * sigma ** 2)) * T) 
              / (sigma * np.sqrt(T)))
          
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        minusNd1 = si.norm.cdf(-d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        minusNd2 = si.norm.cdf(-d2, 0.0, 1.0)
               
        if option == "call":
            opt_price = ((S * carry * Nd1) - (K * np.exp(-r * T) * Nd2))  
        if option == 'put':
            opt_price = ((K * np.exp(-r * T) * minusNd2) - 
                         (S * carry * minusNd1))
               
        return opt_price
    
    
    @timethis
    def black_scholes_merton_vega(self, S=None, K=None, T=None, r=None, q=None, 
                                  sigma=None, option=None, timing=None, 
                                  default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated. 
            
        Returns
        -------
        opt_vega : Float
            Option Vega.

        """  
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, option=option, 
                    timing=timing))
        
        b = r - q
        carry = np.exp((b - r) * T)
        d1 = ((np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) 
              / (sigma * np.sqrt(T)))
        nd1 = (1 / np.sqrt(2 * np.pi)) * (np.exp(-d1 ** 2 * 0.5))
        
        opt_vega = S * carry * nd1 * np.sqrt(T)
         
        return opt_vega
    
    
    @timethis
    def black_76(self, F=None, K=None, T=None, r=None, sigma=None, option=None, 
                 timing=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated. 
            
        Returns
        -------
        opt_price : Float
            Option Price.

        """

        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            F, K, T, r, sigma, option, timing = itemgetter(
                'F', 'K', 'T', 'r', 'sigma', 'option', 
                'timing')(self._refresh_params_default(
                    F=F, K=K, T=T, r=r, sigma=sigma, option=option, 
                    timing=timing))
              
        carry = np.exp(-r * T)
        d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(F / K) + (-0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
          
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        minusNd1 = si.norm.cdf(-d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        minusNd2 = si.norm.cdf(-d2, 0.0, 1.0)
               
        if option == "call":
            opt_price = ((F * carry * Nd1) - (K * np.exp(-r * T) * Nd2))  
        if option == 'put':
            opt_price = ((K * np.exp(-r * T) * minusNd2) 
                         - (F * carry * minusNd1))
               
        return opt_price
    
    
    @timethis
    def european_binomial(self, S=None, K=None, T=None, r=None, q=None, 
                          sigma=None, steps=None, option=None, timing=None, 
                          default=None):
        """
        European Binomial Option price.
        Combinatorial function limit c1000
    
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        Float
            European Binomial Option Price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, sigma, steps, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    option=option, timing=timing))
                
        b = r - q            
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(b * dt) - d) / (u - d)
        a = int(np.log(K / (S * (d ** steps))) / np.log(u / d)) + 1
        
        val = 0
        
        if option == 'call':
            for j in range(a, steps + 1):
                val = (
                    val + (comb(steps, j) * (p ** j) 
                           * ((1 - p) ** (steps - j)) 
                           * ((S * (u ** j) * (d ** (steps - j))) - K)))
        if option == 'put':
            for j in range(0, a):
                val = (
                    val + (comb(steps, j) * (p ** j) 
                           * ((1 - p) ** (steps - j)) 
                           * (K - ((S * (u ** j)) * (d ** (steps - j))))))
                               
        return np.exp(-r * T) * val                     
                
    
    @timethis
    def cox_ross_rubinstein_binomial(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps=None, option=None, output_flag=None, american=None, 
            timing=None, default=None):
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
            Whether to return 'price', 'delta', 'gamma', 'theta' or 
            'all'. The default is 'price'.
        american : Bool
            Whether the option is American. The default is False.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'all' : Tuple; Option Price, Option Delta, Option 
                        Gamma, Option Theta  

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps, option, output_flag, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'option', 
                'output_flag', 'american', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    option=option, output_flag=output_flag, american=american, 
                    timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
        
        b = r - q
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(b * dt) - d) / (u - d)
        df = np.exp(-r * dt)
        optionvalue = np.zeros((steps + 2))
        returnvalue = np.zeros((4))
        
        for i in range(steps + 1):
            optionvalue[i] = max(
                0, z * (S * (u ** i) * (d ** (steps - i)) - K))
            
            
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                if american:
                    optionvalue[i] = (
                        (p * optionvalue[i + 1]) 
                        + ((1 - p) * optionvalue[i])) * df
                else:
                    optionvalue[i] = max(
                        (z * (S * (u ** i) * (d ** (j - i)) - K)), 
                        ((p * optionvalue[i + 1]) 
                         + ((1 - p) * optionvalue[i])) * df)
            
            if j == 2:
                returnvalue[2] = (((optionvalue[2] - optionvalue[1]) 
                                   / (S * (u ** 2) - S) 
                                   - (optionvalue[1] - optionvalue[0]) 
                                   / (S - S * (d ** 2))) 
                                  / (0.5 * (S * (u ** 2) - S * (d ** 2))))
                
                returnvalue[3] = optionvalue[1]
                
            if j == 1:
                returnvalue[1] = ((
                    optionvalue[1] - optionvalue[0]) / (S * u - S * d))
            
        returnvalue[3] = (returnvalue[3] - optionvalue[0]) / (2 * dt) / 365
        returnvalue[0] = optionvalue[0]
        
        if output_flag == 'price':
            result = returnvalue[0]
        if output_flag == 'delta':
            result = returnvalue[1]
        if output_flag == 'gamma':
            result = returnvalue[2]
        if output_flag == 'theta':
            result = returnvalue[3]
        if output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]),
                      'Theta = '+str(returnvalue[3]))
                               
        return result
    
    
    @timethis
    def leisen_reimer_binomial(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps=None, option=None, output_flag=None, american=None, 
            timing=None, default=None):
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
            Whether to return 'price', 'delta', 'gamma' or 'all'. The 
            default is 'price'.
        american : Bool
            Whether the option is American. The default is False.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'all' : Tuple; Option Price, Option Delta, Option Gamma  

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps, option, output_flag, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'option', 
                'output_flag', 'american', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    option=option, output_flag=output_flag, american=american, 
                    timing=timing))
        
                
        if option == 'call':
            z = 1
        else:
            z = -1
        
        b = r - q
        d1 = ((np.log(S / K) + (b + (0.5 * sigma ** 2)) * T) 
              / (sigma * np.sqrt(T)))
        d2 = ((np.log(S / K) + (b - (0.5 * sigma ** 2)) * T) 
              / (sigma * np.sqrt(T)))
        hd1 = (
            0.5 + np.sign(d1) * (0.25 - 0.25 * np.exp(
                -(d1 / (steps + 1 / 3 + 0.1 / (steps + 1))) ** 2 
                * (steps + 1 / 6))) ** (0.5))
        hd2 = (
            0.5 + np.sign(d2) * (0.25 - 0.25 * np.exp(
                -(d2 / (steps + 1 / 3 + 0.1 / (steps + 1))) ** 2 
                * (steps + 1 / 6))) ** (0.5))
        
        dt = T / steps
        p = hd2
        u = np.exp(b * dt) * hd1 / hd2
        d = (np.exp(b * dt) - p * u) / (1 - p)
        df = np.exp(-r * dt)
    
        optionvalue = np.zeros((steps + 1))
        returnvalue = np.zeros((4))
        
        for i in range(steps + 1):
            optionvalue[i] = max(0, z * (S * (u ** i) * (
                d ** (steps - i)) - K))
            
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                if american:
                    optionvalue[i] = (
                        (p * optionvalue[i + 1]) 
                        + ((1 - p) * optionvalue[i])) * df
                else:
                    optionvalue[i] = max(
                        (z * (S * (u ** i) * (d ** (j - i)) - K)), 
                        ((p * optionvalue[i + 1]) 
                         + ((1 - p) * optionvalue[i])) * df)
                    
            if j == 2:
                returnvalue[2] = (
                    ((optionvalue[2] - optionvalue[1]) 
                     / (S * (u ** 2) - S * u * d) 
                     - (optionvalue[1] - optionvalue[0]) 
                     / (S * u * d - S * (d ** 2))) 
                    / (0.5 * (S * (u ** 2) - self.S * (d ** 2))))
                
                returnvalue[3] = optionvalue[1]
                
            if j == 1:
                returnvalue[1] = ((optionvalue[1] - optionvalue[0]) 
                                  / (S * u - S * d))
            
        returnvalue[0] = optionvalue[0]
        
        if output_flag == 'price':
            result = returnvalue[0]
        if output_flag == 'delta':
            result = returnvalue[1]
        if output_flag == 'gamma':
            result = returnvalue[2]
        if output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]))
    
        return result        
    
    
    @timethis
    def trinomial_tree(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps=None, option=None, output_flag=None, american=None, 
            timing=None, default=None):
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
            Whether to return 'price', 'delta', 'gamma', 'theta' or 
            'all'. The default is 'price'.
        american : Bool
            Whether the option is American. The default is False.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'all' : Tuple; Option Price, Option Delta, Option Gamma, 
                        Option Theta  

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps, option, output_flag, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'option', 
                'output_flag', 'american', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    option=option, output_flag=output_flag, american=american, 
                    timing=timing))
                                
        if option == 'call':
            z = 1
        else:
            z = -1
        
        b = r - q
        dt = T / steps
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = np.exp(-sigma * np.sqrt(2 * dt))
        pu = ((np.exp(b * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2))) 
              / (np.exp(sigma * np.sqrt(dt / 2)) 
                 - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
        pd = ((np.exp(sigma * np.sqrt(dt / 2)) - np.exp(b * dt / 2)) 
              / (np.exp(sigma * np.sqrt(dt / 2)) 
                 - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
        pm = 1 - pu - pd
        df = np.exp(-r * dt)
        optionvalue = np.zeros((steps * 2 + 2))
        returnvalue = np.zeros((4))
        
        for i in range(2 * steps + 1):
            optionvalue[i] = max(
                0, z * (S * (u ** max(i - steps, 0)) 
                        * (d ** (max((steps - i), 0))) - K))
            
            
        for j in range(steps - 1, -1, -1):
            for i in range(j * 2 + 1):
                
                optionvalue[i] = (pu * optionvalue[i + 2] 
                                  + pm * optionvalue[i + 1] 
                                  + pd * optionvalue[i]) * df
                
                if self.american == True:
                    optionvalue[i] = max(
                        z * (S * (u ** max(i - j, 0)) 
                             * (d ** (max((j - i), 0))) - K), optionvalue[i])
            
            if j == 1:
                returnvalue[1] = (
                    (optionvalue[2] - optionvalue[0]) / (S * u - S * d))
                
                returnvalue[2] = (
                    ((optionvalue[2] - optionvalue[1]) / (S * u - S) 
                     - (optionvalue[1] - optionvalue[0]) / (S - S * d )) 
                    / (0.5 * ((S * u) - (S * d))))                              
                
                returnvalue[3] = optionvalue[0]
                
        returnvalue[3] = (returnvalue[3] - optionvalue[0]) / dt / 365
        
        returnvalue[0] = optionvalue[0]
        
        if output_flag == 'price':
            result = returnvalue[0]
        if output_flag == 'delta':
            result = returnvalue[1]
        if output_flag == 'gamma':
            result = returnvalue[2]
        if output_flag == 'theta':
            result = returnvalue[3]
        if output_flag == 'all':
            result = ('Price = '+str(returnvalue[0]),
                      'Delta = '+str(returnvalue[1]),
                      'Gamma = '+str(returnvalue[2]),
                      'Theta = '+str(returnvalue[3]))
                               
        return result                     
    
    
    @timethis
    def implied_trinomial_tree(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps_itt=None, option=None, output_flag=None, step=None, 
            state=None, skew=None, timing=None, default=None):
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
            UPni: The implied up transition probability at a single 
                  node
            DPM: A matrix of implied down transition probabilities
            DPni: The implied down transition probability at a single 
                  node
            LVM: A matrix of implied local volatilities
            LVni: The local volatility at a single node
            ADM: A matrix of Arrow-Debreu prices at a single node
            ADni: The Arrow-Debreu price at a single node (at 
                  time step - 'step' and state - 'state')
            price: The value of the European option
        step : Int
            Time step used for Arrow Debreu price at single node. The 
            default is 5.
        state : Int
            State position used for Arrow Debreu price at single node. 
            The default is 5.
        skew : Float
            Rate at which volatility increases (decreases) for every 
            one point decrease 
            (increase) in the strike price. The default is 0.0004.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Various
            Depending on output flag:
                UPM: A matrix of implied up transition probabilities
                UPni: The implied up transition probability at a single 
                      node
                DPM: A matrix of implied down transition probabilities
                DPni: The implied down transition probability at a 
                      single node
                LVM: A matrix of implied local volatilities
                LVni: The local volatility at a single node
                ADM: A matrix of Arrow-Debreu prices at a single node
                ADni: The Arrow-Debreu price at a single node (at 
                      time step - 'step' and state - 'state')
                price: The European option price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps_itt, option, output_flag, step, 
             state, skew, timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps_itt', 'option', 
                'output_flag', 'step', 'state', 'skew', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps_itt=steps_itt, 
                    option=option, output_flag=output_flag, step=step, 
                    state=state, skew=skew, timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
        
        optionvaluenode = np.zeros((steps_itt * 2 + 1))
        # Arrow Debreu prices
        ad = np.zeros((steps_itt + 1, steps_itt * 2 + 1), dtype='float')
        pu = np.zeros((steps_itt, steps_itt * 2 - 1), dtype='float')
        pd = np.zeros((steps_itt, steps_itt * 2 - 1), dtype='float')
        localvol = np.zeros((steps_itt, steps_itt * 2 - 1), dtype='float')
        
        dt = T / steps_itt
        u = np.exp(sigma * np.sqrt(2 * dt))
        d = 1 / u
        df = np.exp(-r * dt)
        ad[0, 0] = 1 
                
        for n in range(steps_itt):
            for i in range(n * 2 + 1):
                val = 0
                Si1 = (S * (u ** (max(i - n, 0))) 
                       * (d ** (max(n * 2 - n - i, 0))))
                Si = Si1 * d
                Si2 = Si1 * u
                b = r - q
                Fi = Si1 * np.exp(b * dt)
                sigmai = sigma + (S - Si1) * skew
                
                if i < (n * 2) / 2 + 1:
                    for j in range(i):
                        Fj = (S * (u ** (max(j - n, 0))) 
                              * (d ** (max(n * 2 - n - j, 0))) 
                              * np.exp(b * dt))
                        
                        val = val + ad[n, j] * (Si1 - Fj)
                        
                    optionvalue = self.trinomial_tree(
                        S=S, K=Si1, T=(n + 1) * dt, r=r, q=q, sigma=sigmai, 
                        steps=(n + 1), option='put', output_flag='price', 
                        american=False, timing=False, default=False)
        
                    qi = ((np.exp(r * dt) * optionvalue - val) 
                          / (ad[n, i] * (Si1 - Si)))
                    
                    pi = (Fi + qi * (Si1 - Si) - Si1) / (Si2 - Si1)
                
                else:
                    optionvalue = self.trinomial_tree(
                        S=S, K=Si1, T=(n + 1) * dt, r=r, q=q, sigma=sigmai, 
                        steps=(n + 1), option='call', output_flag='price', 
                        american=False, timing=False, default=False)
                    
                    val = 0
                    for j in range(i + 1, n * 2 + 1):
                        Fj = (S * (u ** (max(j - n, 0))) 
                              * (d ** (max(n * 2 - n - j, 0))) 
                              * np.exp(b * dt))
                        
                        val = val + ad[n, j] * (Fj- Si1)
    
                    pi = ((np.exp(r * dt) * optionvalue - val) 
                          / (ad[n, i] * (Si2 - Si1)))
                    
                    qi = (Fi - pi * (Si2 - Si1) - Si1) / (Si - Si1)
                
                # Replacing negative probabilities    
                if pi < 0 or pi > 1 or qi < 0 or qi > 1:
                    if Fi > Si1 and Fi < Si2:
                        pi = (1 / 2 * ((Fi - Si1) / (Si2 - Si1) 
                                       + (Fi - Si) / (Si2 - Si)))
                        
                        qi = 1 / 2 * ((Si2 - Fi) / (Si2 - Si))
                    
                    elif Fi > Si and Fi < Si1:
                        pi = 1 / 2 * ((Fi - Si) / (Si2 - Si))
                        
                        qi = (1 / 2 * ((Si2 - Fi) / (Si2 - Si1) 
                                       + (Si1 - Fi) / (Si1 - Si)))
    
                pd[n, i] = qi
                pu[n, i] = pi
                
                # Calculating local volatilities
                Fo = (pi * Si2 + qi * Si + (1 - pi -qi) * Si1)
                localvol[n, i] = np.sqrt(
                    (pi * (Si2 - Fo) ** 2 
                     + (1 - pi - qi) * (Si1 - Fo) ** 2 
                     + qi * (Si - Fo) ** 2) / (Fo ** 2 * dt))
        
                # Calculating Arrow-Debreu prices
                if n == 0:
                    ad[n + 1, i] = qi * ad[n, i] * df
                    ad[n + 1, i + 1] = (1 - pi - qi) * ad[n, i] * df
                    ad[n + 1, i + 2] = pi * ad[n, i] * df
                
                elif n > 0 and i == 0:
                    ad[n + 1, i] = qi * ad[n, i] * df
                
                elif n > 0 and i == n * 2:
                    ad[n + 1, i] = (
                        pu[n, i - 2] * ad[n, i - 2] * df 
                        + (1 - pu[n, i - 1] - pd[n, i - 1]) 
                        * (ad[n, i - 1]) * df + qi * (ad[n, i] * df))
                    ad[n + 1, i + 1] = (
                        pu[n, i - 1] * (ad[n, i - 1]) * df 
                        + (1 - pi - qi) * (ad[n, i] * df))
                    ad[n + 1, i + 2] = pi * ad[n, i] * df
                
                elif n > 0 and i == 1:
                    ad[n + 1, i] = (
                        (1 - pu[n, i - 1] - (pd[n, i - 1])) 
                        * ad[n, i - 1] * df + (qi * ad[n, i] * df))
                            
                else:
                    ad[n + 1, i] = (
                        pu[n, i - 2] * (ad[n, i - 2]) * df 
                        + (1 - pu[n, i - 1] - pd[n, i - 1]) 
                        * (ad[n, i - 1]) * df + qi * (ad[n, i]) * df)
    
            
        if output_flag == 'UPM':    
            result = pu
        elif output_flag == 'UPni':    
            result = pu[step, state]        
        elif output_flag == 'DPM':
            result = pd
        elif output_flag == 'DPni':    
            result = pd[step, state]
        elif output_flag == 'LVM': 
            result = localvol
        elif output_flag == 'LVni':
            result = localvol[step, state]
        elif output_flag == 'ADM':    
            result = ad
        elif output_flag == 'ADni':    
            result = ad[step, state]
        elif output_flag == 'price':
            
            # Calculation of option price using the implied trinomial tree
            for i in range(2 * steps_itt + 1):
                optionvaluenode[i] = max(
                    0, z * (S * (u ** max(i - steps_itt, 0)) 
                            * (d ** (max((steps_itt - i), 0))) - K))    
    
            for n in range(steps_itt - 1, -1, -1):
                for i in range(n * 2 + 1):
                    optionvaluenode[i] = (
                        (pu[n, i] * optionvaluenode[i + 2] 
                         + (1 - pu[n, i] - pd[n, i]) 
                         * (optionvaluenode[i + 1]) 
                         + pd[n, i] * (optionvaluenode[i])) * df)
    
            result = optionvaluenode[0] * 1000000         
                               
        return result    
    
    
    @timethis
    def explicit_finite_difference(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            nodes=None, option=None, american=None, timing=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Float
            Option Price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, nodes, option, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'nodes', 'option', 
                'american', 'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, nodes=nodes, 
                    option=option, american=american, timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
        
        b = r - q
        dS = S / nodes
        nodes = int(K / dS) * 2
        St = np.zeros((nodes + 2), dtype='float')
        
        SGridtPt = int(S / dS)
        dt = (dS ** 2) / ((sigma ** 2) * 4 * (K ** 2))
        N = int(T / dt) + 1
        
        C = np.zeros((N + 1, nodes + 2), dtype='float')
        dt = T / N
        Df = 1 / (1 + r * dt)
          
        for i in range(nodes + 1):
            St[i] = i * dS # Asset price at maturity
            C[N, i] = max(0, z * (St[i] - K) ) # At maturity
            
        for j in range(N - 1, -1, -1):
            for i in range(1, nodes):
                pu = 0.5 * ((sigma ** 2) * (i ** 2) + b * i) * dt
                pm = 1 - (sigma ** 2) * (i ** 2) * dt
                pd = 0.5 * ((sigma ** 2) * (i ** 2) - b * i) * dt
                C[j, i] = Df * (pu * C[j + 1, i + 1] + pm * C[
                    j + 1, i] + pd * C[j + 1, i - 1])
                if american:
                    C[j, i] = max(z * (St[i] - K), C[j, i])
                    
                if z == 1: # Call option
                    C[j, 0] = 0
                    C[j, nodes] = (St[i] - K)
                else:
                    C[j, 0] = K
                    C[j, nodes] = 0
        
        result = C[0, SGridtPt]
    
        return result          
    
    
    @timethis
    def implicit_finite_difference(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps=None, nodes=None, option=None, american=None, timing=None, 
            default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Float
            Option Price.
        

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps, nodes, option, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'nodes', 'option', 
                'american', 'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    nodes=nodes, option=option, american=american, 
                    timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
              
        # Make sure current asset price falls at grid point
        dS = 2 * S / nodes
        SGridtPt = int(S / dS)
        nodes = int(K / dS) * 2
        dt = T / steps
        b = r - q
        
        CT = np.zeros(nodes + 1)
        p = np.zeros((nodes + 1, nodes + 1), dtype='float')
        
        for j in range(nodes + 1):
            CT[j] = max(0, z * (j * dS - K)) # At maturity
            for i in range(nodes + 1):
                p[j, i] = 0
                
        p[0, 0] = 1
        for i in range(1, nodes):
            p[i, i - 1] = 0.5 * i * (b - (sigma ** 2) * i) * dt
            p[i, i] = 1 + (r + (sigma ** 2) * (i ** 2)) * dt
            p[i, i + 1] = 0.5 * i * (-b - (sigma ** 2) * i) * dt
            
        p[nodes, nodes] = 1
        
        C = np.matmul(np.linalg.inv(p), CT.T)
        
        for j in range(steps - 1, 0, -1):
            C = np.matmul(np.linalg.inv(p), C)
            
            if american:
                for i in range(1, nodes + 1):
                    C[i] = max(float(C[i]), z * (
                        (i - 1) * dS - K))
                
        result = C[SGridtPt + 1]
        
        return result   
    
    
    @timethis
    def explicit_finite_difference_lns(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps=None, nodes=None, option=None, american=None, timing=None, 
            default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Float
            Option Price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps, nodes, option, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'nodes', 'option', 
                'american', 'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    nodes=nodes, option=option, american=american, 
                    timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
        
        b = r - q
        dt = T / steps
        dx = sigma * np.sqrt(3 * dt)
        pu = 0.5 * dt * (((sigma / dx) ** 2) + (b - (sigma ** 2) / 2) / dx)
        pm = 1 - dt * ((sigma / dx) ** 2) - r * dt
        pd = 0.5 * dt * (((sigma / dx) ** 2) - (b - (sigma ** 2) / 2) / dx)
        St = np.zeros(nodes + 2)
        St[0] = S * np.exp(-nodes / 2 * dx)
        C = np.zeros((int(nodes / 2) + 1, nodes + 2), dtype='float')
        C[steps, 0] = max(0, z * (St[0] - K))
        
        for i in range(1, nodes + 1):
            St[i] = St[i - 1] * np.exp(dx) # Asset price at maturity
            C[steps, i] = max(0, z * (St[i] - K) ) # At maturity
        
        for j in range(steps - 1, -1, -1):
            for i in range(1, nodes):
                C[j, i] = pu * C[j + 1, i + 1] + pm * C[j + 1, i] + (
                    pd * C[j + 1, i - 1])
                if american:
                    C[j, i] = max(C[j, i], z * (St[i] - self.K))
                
                # Upper boundary    
                C[j, nodes] = C[j, nodes - 1] + (St[nodes] - St[nodes - 1]) 
                
                # Lower boundary
                C[j, 0] = C[j, 1] 
           
        result = C[0, int(nodes / 2)]
    
        return result   
    
    
    @timethis
    def crank_nicolson(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            steps=None, nodes=None, option=None, american=None, timing=None, 
            default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Float
            Option Price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, steps, nodes, option, american, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'steps', 'nodes', 'option', 
                'american', 'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, steps=steps, 
                    nodes=nodes, option=option, american=american, 
                    timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
                
        b = r - q    
        dt = T / steps
        dx = sigma * np.sqrt(3 * dt)
        pu = -0.25 * dt * (((sigma / dx) ** 2) + (b - (sigma ** 2) / 2) / dx)
        pm = 1 + 0.5 * dt * ((sigma / dx) ** 2) + 0.5 * r * dt
        pd = -0.25 * dt * (((sigma / dx) ** 2) - (b - (sigma ** 2) / 2) / dx)
        St = np.zeros(nodes + 2)
        pmd = np.zeros(nodes + 1)
        p = np.zeros(nodes + 1)
        St[0] = S * np.exp(-nodes / 2 * dx)
        C = np.zeros((int(nodes / 2) + 2, nodes + 2), dtype='float')
        C[0, 0] = max(0, z * (St[0] - K))
        
        for i in range(1, nodes + 1):
            St[i] = St[i - 1] * np.exp(dx) # Asset price at maturity
            C[0, i] = max(0, z * (St[i] - K)) # At maturity
        
        pmd[1] = pm + pd
        p[1] = (-pu * C[0, 2] 
                - (pm - 2) * C[0, 1] 
                - pd * C[0, 0] 
                - pd * (St[1] - St[0]))
        
        for j in range(steps - 1, -1, -1):
            for i in range(2, nodes):
                p[i] = (-pu * C[0, i + 1] 
                        - (pm - 2) * C[0, i] 
                        - pd * C[0, i - 1] 
                        - p[i - 1] * pd / pmd[i - 1])
                pmd[i] = pm - pu * pd / pmd[i - 1]
    
            for i in range(nodes - 2, 0, -1):
                C[1, i] = (p[i] - pu * C[1, i + 1]) / pmd[i]
                
                for i in range(nodes + 1):
                    C[0, i] = C[1, i]
                    if american:
                        C[0, i] = max(C[1, i], z * (St[i] - K))
           
        result = C[0, int(nodes / 2)]
    
        return result   
    
    
    @timethis
    def european_monte_carlo(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            simulations=None, option=None, timing=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Float
            Option Price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, simulations, option, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'simulations', 'option',
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                    simulations=simulations, option=option, timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
                    
        b = r - q
        Drift = (b - (sigma ** 2) / 2) * T
        sigmarT = sigma * np.sqrt(T)
        val = 0
        
        
        
        for i in range(1, simulations + 1):
            St = S * np.exp(
                Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
            val = val + max(z * (St - K), 0) 
            
        result = np.exp(-r * T) * val / simulations
        
        return result
    
    
    @timethis
    def european_monte_carlo_with_greeks(
            self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
            simulations=None, option=None, output_flag=None, timing=None, 
            default=None):
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
            Whether to return 'price', 'delta', 'gamma', 'theta', 
            'vega' or 'all'. The default is 'price'.
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Various
            Depending on output flag:
                'price' : Float; Option Price  
                'delta' : Float; Option Delta
                'gamma' : Float; Option Gamma
                'theta' : Float; Option Theta
                'vega' : Float; Option Vega
                'all' : Tuple; Option Price, Option Delta, Option 
                               Gamma, Option Theta, Option Vega  

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, simulations, option, output_flag,
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'simulations', 'option', 
                'output_flag', 'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, 
                    simulations=simulations, option=option, 
                    output_flag=output_flag, timing=timing))
        
        if option == 'call':
            z = 1
        else:
            z = -1
                    
        b = r - q
        Drift = (b - (sigma ** 2) / 2) * T
        sigmarT = sigma * np.sqrt(T)
        val = 0
        deltasum = 0
        gammasum = 0
        output = {}
        
        for i in range(1, simulations + 1):
            St = S * np.exp(
                Drift + sigmarT * norm.ppf(random.random(), loc=0, scale=1))
            val = val + max(z * (St - K), 0) 
            if z == 1 and St > K:
                deltasum = deltasum + St
            if z == -1 and St < K:
                deltasum = deltasum + St
            if abs(St - K) < 2:
                gammasum = gammasum + 1
                
        # Option Value
        output[0] = np.exp(-r * T) * val / simulations       
            
        # Delta
        output[1] = np.exp(-r * T) * deltasum / (simulations * S)
        
        # Gamma
        output[2] = (np.exp(-r * T) * ((K / S) ** 2) 
                     * gammasum / (4 * simulations))
        
        # Theta
        output[3] = ((r * output[0] 
                      - b * S * output[1] 
                      - (0.5 * (sigma ** 2) * (S ** 2) * output[2])) 
                     / 365)
        
        # Vega
        output[4] = output[2] * sigma * (S ** 2) * T / 100
    
        if output_flag == 'price':
            result = output[0]
        if output_flag == 'delta':
            result = output[1]
        if output_flag == 'gamma':
            result = output[2]
        if output_flag == 'theta':
            result = output[3]
        if output_flag == 'vega':
            result = output[4]
        if output_flag == 'all':
            result = ('Price = '+str(output[0]),
                      'Delta = '+str(output[1]),
                      'Gamma = '+str(output[2]),
                      'Theta = '+str(output[3]),
                      'Vega = '+str(output[4]))
                
        return result
    
    
    @timethis
    def hull_white_87(self, S=None, K=None, T=None, r=None, q=None, sigma=None, 
                      vvol=None, option=None, timing=None, default=None):
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
        default : Bool
            Whether the function is being called directly (in which 
            case values that are not supplied are set to default 
            values) or called from another function where they have 
            already been updated.     

        Returns
        -------
        result : Float
            Option price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.     
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sigma, vvol, option, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sigma', 'vvol', 'option',
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sigma=sigma, vvol=vvol, 
                    option=option, timing=timing))
        
        k = vvol ** 2 * T
        ek = np.exp(k)
        b = r - q
        d1 = ((np.log(S / K) + (b + (sigma ** 2) / 2) * T) 
              / (sigma * np.sqrt(T)))
        d2 = d1 - sigma * np.sqrt(T)
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
           
        cgbs = self.black_scholes_merton(
            S=S, K=K, T=T, r=r, q=q, sigma=sigma, option='call', timing=False, 
            default=False)
        
        # Partial Derivatives
        cVV = (S 
               * np.exp((b - r) * T) 
               * np.sqrt(T) 
               * Nd1 
               * (d1 * d2 - 1) 
               / (4 * (sigma ** 3)))
        
        cVVV = (S 
                * np.exp((b - r) * T) 
                * np.sqrt(T) 
                * Nd1 
                * ((d1 * d2 - 1) * (d1 * d2 - 3) - ((d1 ** 2) + (d2 ** 2))) 
                / (8 * (self.sigma ** 5)))                                                             
        
        callvalue = (cgbs 
                     + 1 / 2 
                     * cVV 
                     * (2 * sigma ** 4 * (ek - k - 1) / k ** 2 - sigma  ** 4) 
                     + (1 / 6 * cVVV * sigma ** 6 
                        * (ek ** 3 
                           - (9 + 18 * k) * ek 
                           + 8 
                           + 24 * k 
                           + 18 * k ** 2 
                           + (6 * k ** 3)) 
                        / (3 * k ** 3)))
        
        if option == 'call':
            result = callvalue
            
        if option == 'put': # use put-call parity
            result = callvalue - S * np.exp((b - r) * T) + K * np.exp(-r * T)
            
        return result


    @timethis
    def hull_white_88(self, S=None, K=None, T=None, r=None, q=None, sig0=None, 
                      sigLR=None, halflife=None, vvol=None, rho=None, 
                      option=None, timing=None, default=None):
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
            Long run mean reversion level of volatility. The default 
            is 0.0625 (6.25%).
        halflife : Float
            Half-life of volatility deviation. The default is 0.1. 
        vvol : Float
            Vol of vol. The default is 0.5.
        rho : Float
            Correlation between asset price and volatility. The 
            default is 0.
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
            Option price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (S, K, T, r, q, sig0, sigLR, halflife, vvol, rho, option, 
             timing) = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'sig0', 'sigLR', 'halflife', 'vvol', 
                'rho', 'option', 'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, sig0=sig0, sigLR=sigLR, 
                    halflife=halflife, vvol=vvol, rho=rho, option=option, 
                    timing=timing))
        
        b = r - q            
        # Find constant, beta, from Half-life
        beta = -np.log(2) / halflife 
        
        # Find constant, a, from long run volatility
        a = -beta * (sigLR ** 2) 
        delta = beta * T
        ed = np.exp(delta)
        v = sig0 ** 2

        # Average expected variance
        if abs(beta) < 0.0001:
            vbar = v + 0.5 * a * T 
        else:
            vbar = (v + (a / beta)) * ((ed - 1) / delta) - (a / beta)
            
        d1 = (np.log(S / K) + (b + (vbar / 2)) * T) / np.sqrt(vbar * T)
        d2 = d1 - np.sqrt(vbar * T)
        
        # standardised normal density function
        nd1 = (1 / np.sqrt(2 * np.pi)) * (np.exp(-d1 ** 2 * 0.5))
        
        # Cumulative normal distribution function
        Nd1 = si.norm.cdf(d1, 0.0, 1.0)
        Nd2 = si.norm.cdf(d2, 0.0, 1.0)
        
        # Partial derivatives
        cSV = (-S * np.exp((b - r) * T) * nd1 * (d2 / (2 * vbar)))
        cVV = (
            (S * np.exp((b - r) * T) * nd1 * np.sqrt(T) / (4 * vbar ** 1.5)) 
            * (d1 * d2 - 1))
        
        cSVV = (
            (S * np.exp((b - r) * T) / (4 * vbar ** 2)) 
            * nd1 * ((-d1 * (d2 ** 2)) + d1 + (2 * d2)))                      
        
        cVVV = (
            ((S * np.exp((b - r) * T) * nd1 * np.sqrt(T)) / (8 * vbar ** 2.5)) 
            * ((d1 * d2 - 1) * (d1 * d2 - 3) - ((d1 ** 2) + (d2 ** 2)))) 

        if abs(beta) < 0.0001:
            f1 = rho * ((a * T / 3) + v) * (T / 2) * cSV
            phi1 = (rho ** 2) * ((a * T / 4) + v) * ((T ** 3) / 6)
            phi2 = (2 + (1 / (rho ** 2))) * phi1
            phi3 = (rho ** 2) * (((a * T / 3) + v) ** 2) * ((T ** 4) / 8)
            phi4 = 2 * phi3
            
        else: # Beta different from zero
            phi1 = (
                ((rho ** 2) / (beta ** 4)) 
                * (((a + (beta * v)) 
                    * ((ed * (((delta ** 2) / 2) - delta + 1)) - 1)) 
                   + (a * ((ed * (2 - delta)) - (2 + delta)))))
            
            phi2 = (
                (2 * phi1) 
                + ((1 / (2 * (beta ** 4))) 
                   * (((a + (beta * v)) * ((ed ** 2) - (2 * delta * ed) - 1)) 
                      - ((a / 2) * ((ed ** 2) - (4 * ed) + (2 * delta) + 3)))))
            
            phi3 = (
                ((rho ** 2) / (2 * (beta ** 6))) 
                * ((((a + (beta * v)) * (ed - delta * ed - 1)) 
                    - (a * (1 + delta - ed))) ** 2))
            
            phi4 = 2 * phi3
            
            f1 = (
                (rho / ((beta ** 3) * T)) 
                * (((a + (beta * v)) * (1 - ed + (delta * ed))) 
                   + (a * (1 + delta - ed))) * cSV)

        f0 = S * np.exp((b - r) * T) * Nd1 - (K * np.exp(-r * T) * Nd2)
        
        f2 = (
            ((phi1 / T) * cSV) 
            + ((phi2 / (T ** 2)) * cVV) 
            + ((phi3 / (T ** 2)) * cSVV) 
            + ((phi4 / (T ** 3)) * cVVV))
        
        callvalue = f0 + f1 * vvol + f2 * vvol ** 2
        
        if option == 'call':
            result = callvalue
        else:
            result = (
                callvalue - (S * np.exp((b - r) * T)) + (K * np.exp(-r * T)))
        
        return result



class ImpliedVol(Pricer):
    
    def __init__(self):
        super().__init__(self) # Inherit methods from Pricer class

    
    @timethis
    def implied_vol_newton_raphson(self, S=None, K=None, T=None, r=None, 
                                   q=None, cm=None, epsilon=None, option=None, 
                                   timing=None, default=None):
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
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, cm, epsilon, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'cm', 'epsilon', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, 
                    option=option, timing=timing))
               
        # Manaster and Koehler seed value
        vi = np.sqrt(abs(np.log(S / K) + r * T) * (2 / T))
        
        ci = self.black_scholes_merton(
            S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
            default=False)    
        
        vegai = self.black_scholes_merton_vega(
            S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
            default=False)
        
        mindiff = abs(cm - ci)
    
        while abs(cm - ci) >= epsilon and abs(cm - ci) <= mindiff:
            vi = vi - (ci - cm) / vegai
            
            ci = self.black_scholes_merton(
                S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
                default=False)
            
            vegai = self.black_scholes_merton_vega(
                S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
                default=False)
            
            mindiff = abs(cm - ci)
            
        if abs(cm - ci) < epsilon:
            result = vi
        else:
            result = 'NA'
        
        return result
    
    
    @timethis
    def implied_vol_bisection(self, S=None, K=None, T=None, r=None, q=None, 
                              cm=None, epsilon=None, option=None, timing=None, 
                              default=None):
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
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, cm, epsilon, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'cm', 'epsilon', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, 
                    option=option, timing=timing))
                    
        vLow = 0.005
        vHigh = 4
        cLow = self.black_scholes_merton(
            S=S, K=K, T=T, r=r, q=q, sigma=vLow, option=option, timing=False, 
            default=False)
        
        cHigh = self.black_scholes_merton(
            S=S, K=K, T=T, r=r, q=q, sigma=vHigh, option=option, timing=False, 
            default=False)
        
        counter = 0
        
        vi = vLow + (cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
        
        while abs(cm - self.black_scholes_merton(
                S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
                default=False)) > epsilon:
            
            counter = counter + 1
            if counter == 100:
                result = 'NA'
            
            if self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False) < cm:
                vLow = vi
            
            else:
                vHigh = vi
            
            cLow = self.black_scholes_merton(
                S=S, K=K, T=T, r=r, q=q, sigma=vLow, 
                option=self.option, timing=False)
            
            cHigh = self.black_scholes_merton(
                S=S, K=K, T=T, r=r, q=q, sigma=vHigh, option=option, 
                timing=False)
            
            vi = vLow + (cm - cLow) * (vHigh - vLow) / (cHigh - cLow)
            
        result = vi    
            
        return result
   
    
    @timethis
    def implied_vol_naive(self, S=None, K=None, T=None, r=None, q=None, 
                          cm=None, epsilon=None, option=None, timing=None, 
                          default=None):
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
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, cm, epsilon, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'cm', 'epsilon', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, 
                    option=option, timing=timing))
        
        # Seed vol
        vi = 0.2
        
        # Calculate starting option price using this vol
        ci = self.black_scholes_merton(
            S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
            default=False)
        
        # Initial price difference
        price_diff = cm - ci
        
        if price_diff > 0:
            flag = 1
        
        else:
            flag = -1
        
        # Starting vol shift size
        shift = 0.01
        
        price_diff_start = price_diff
        
        while abs(price_diff) > epsilon:
            
            # If the price difference changes sign after the vol shift, 
            # reduce the decimal by one and reverse the sign
            if np.sign(price_diff) != np.sign(price_diff_start):
                shift = shift * -0.1                
            
            # Calculate new vol
            vi += (shift * flag)
            
            # Set initial price difference
            price_diff_start = price_diff
            
            # Calculate the option price with new vol
            ci = self.black_scholes_merton(
                S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
                default=False)
            
            # Price difference after shifting vol
            price_diff = cm - ci
            
            # If values are diverging reverse the shift sign
            if abs(price_diff) > abs(price_diff_start):
                shift = -shift
       
        result = vi    
            
        return result
    
    
    @timethis
    def implied_vol_naive_verbose(
            self, S=None, K=None, T=None, r=None, q=None, cm=None, 
            epsilon=None, option=None, timing=None, default=None):
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
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            S, K, T, r, q, cm, epsilon, option, timing = itemgetter(
                'S', 'K', 'T', 'r', 'q', 'cm', 'epsilon', 'option', 
                'timing')(self._refresh_params_default(
                    S=S, K=K, T=T, r=r, q=q, cm=cm, epsilon=epsilon, 
                    option=option, timing=timing))
        
        vi = 0.2
        ci = self.black_scholes_merton(
            S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, timing=False, 
            default=False)
        
        price_diff = cm - ci
        if price_diff > 0:
            flag = 1
        else:
            flag = -1
        while abs(price_diff) > epsilon:
            while price_diff * flag > 0:
                ci = self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False)
                
                price_diff = self.cm - ci
                vi += (0.01 * flag)
            
            while price_diff * flag < 0:
                ci = self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False)

                price_diff = self.cm - ci
                vi -= (0.001 * flag)
            
            while price_diff * flag > 0:
                ci = self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False)

                price_diff = self.cm - ci
                vi += (0.0001 * flag)
            
            while price_diff * flag < 0:
                ci = self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False)

                price_diff = self.cm - ci
                vi -= (0.00001 * flag)
                
            while price_diff * flag > 0:
                ci = self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False)

                price_diff = self.cm - ci
                vi += (0.000001 * flag)
            
            while price_diff * flag < 0:
                ci = self.black_scholes_merton(
                    S=S, K=K, T=T, r=r, q=q, sigma=vi, option=option, 
                    timing=False, default=False)

                price_diff = self.cm - ci
                vi -= (0.0000001 * flag)    
        
        result = vi    
            
        return result


class SABRVolatility(Pricer):
    """
    Stochastic, Alpha, Beta, Rho model
    
    Extension of Black 76 model to include an easily implementable 
    stochastic volatility model
    
    Beta will typically be chosen a priori according to how traders 
    observe market prices:
        e.g. In FX markets, standard to assume lognormal terms, Beta = 1
             In some Fixed Income markets traders prefer to assume 
             normal terms, Beta = 0
    
    Alpha will need to be calibrated to ATM volatility         
             
    """
    
    def __init__(
            self, 
            sabr_df_dict=sabr_df_dict):
        
        # Inherit methods from Pricer class
        super().__init__(self) 
        
        # Dictionary of default SABR parameters
        self.sabr_df_dict = sabr_df_dict 
    
    
    def _refresh_sabr_params_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions to the
        default values.

        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.

        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided

        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object 
                # and to input dictionary
                v = self.sabr_df_dict['df_'+str(k)]
                self.__dict__[k] = v
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs        
    
    
    @timethis
    def calibrate(self, F=None, K=None, T=None, r=None, atmvol=None, beta=None, 
                  volvol=None, rho=None, option=None, timing=None, 
                  default=None, output_flag=None):
        """
        Run the SABR calibration

        Returns
        -------
        Float
            Black-76 equivalent SABR volatility and price.

        """
        
        if default is None:
            default = True

        # If default is set to False the price is to be used in another 
        # function so the parameters will already be provided.    
        if default:
            # Update pricing input parameters to default if not supplied
            (F, K, T, r, atmvol, beta, volvol, rho, option, timing, 
             output_flag) = itemgetter(
                'F', 'K', 'T', 'r', 'atmvol', 'beta', 'volvol', 'rho', 
                'option', 'timing', 
                'output_flag')(self._refresh_sabr_params_default(
                    F=F, K=K, T=T, r=r, atmvol=atmvol, beta=beta, 
                    volvol=volvol, rho=rho, option=option, timing=timing, 
                    output_flag=output_flag))
                
        black_vol = self._alpha_sabr(
            F, K, T, beta, volvol, rho, self._find_alpha(
                F=F, T=T, atmvol=atmvol, beta=beta, volvol=volvol, rho=rho,))
        
        black_price = self.black_76(
            F=F, K=K, T=T, r=r, sigma=black_vol, option=option, timing=timing, 
            default=False)
        
        if output_flag == 'vol':
            return black_vol
        
        elif output_flag == 'price':
            return black_price
    
        elif output_flag == 'both':
            return ('Price = '+str(black_price),
                    'Vol = '+str(black_vol))
    
    
    def _alpha_sabr(self, F, K, T, beta, volvol, rho, alpha):
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
        dSABR[1] = (
            alpha 
            / ((F * K) ** ((1 - beta) / 2) 
               * (1 
                  + (((1 - beta) ** 2) / 24) 
                  * (np.log(F / K) ** 2) 
                  + ((1 - beta) ** 4 / 1920) 
                  * (np.log(F / K) ** 4))))
        
        if abs(F - K) > 10 ** -8:
            sabrz = (volvol / alpha) * (F * K) ** (
                (1 - beta) / 2) * np.log(F / K)
            y = (np.sqrt(1 - 2 * rho * sabrz + (
                sabrz ** 2)) + sabrz - rho) / (1 - rho)
            if abs(y - 1) < 10 ** -8:
                dSABR[2] = 1
            elif y > 0:
                dSABR[2] = sabrz / np.log(y)
            else:
                dSABR[2] = 1
        else:
            dSABR[2] = 1
            
        dSABR[3] = (1 + ((((1 - beta) ** 2 / 24) * alpha ** 2 / (
            (F * K) ** (1 - beta))) + (
                0.25 * rho * beta * volvol * alpha) / (
                    (F * K) ** ((1 - beta) / 2)) + (
                        2 - 3 * rho ** 2) * volvol ** 2 / 24) * T)
        
        result = dSABR[1] * dSABR[2] * dSABR[3]
        
        return result
    
    
    def _find_alpha(self, F, T, atmvol, beta, volvol, rho):
        """
        Find alpha feeding values to _cube_root method.

        Returns
        -------
        result : Float
            Smallest positive root.

        """
        # Alpha is a function of atm vol etc
        
        alpha = self._cube_root(
            ((1 - beta) ** 2 * T / (24 * F ** (2 - 2 * beta))), 
            (0.25 * rho * volvol * beta * T / (F ** (1 - beta))), 
            (1 + (2 - 3 * rho ** 2) / 24 * volvol ** 2 * T), 
            (-atmvol * F ** (1 - beta)))
        
        return alpha
    
    
    def _cube_root(self, cubic, quadratic, linear, constant):
        """
        Finds the smallest positive root of the input cubic polynomial 
        algorithm from Numerical Recipes

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
            cap_A = -np.sign(r) * (abs(r) + np.sqrt(
                r ** 2 - Q ** 3)) ** (1 / 3)
            if cap_A == 0:
                cap_B = 0
            else:
                cap_B = Q / cap_A
            result = cap_A + cap_B - a / 3
        else:
            theta = self._arccos(r / Q ** 1.5)
            
            # The three roots
            roots[1] = - 2 * np.sqrt(Q) * math.cos(
                theta / 3) - a / 3
            roots[2] = - 2 * np.sqrt(Q) * math.cos(
                theta / 3 + 2.0943951023932) - a / 3
            roots[3] = - 2 * np.sqrt(Q) * math.cos(
                theta / 3 - 2.0943951023932) - a / 3
            
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
        Return M in M * M.T = R where R is a symmetric positive definite 
        correlation matrix

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
        Number of ways to choose an (unordered) subset of r elements 
        from a fixed 
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
        
        # Due to symmetry of the binomial coefficient, set r to 
        # optimise calculation
        r = min(r, n-r)
        
        # Numerator is the descending product from n to n+1-r
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        
        # Denominator is the product from 1 to r
        denom = reduce(op.mul, range(1, r+1), 1)
        
        # Binomial coefficient is calculated by dividing these two. 
        return numer // denom  # or / in Python 2


