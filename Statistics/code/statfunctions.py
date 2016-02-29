# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:47:08 2016

@author: Brendon

includes definitions for statistical functions that I frequently use and trust
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import seaborn as sns
sns.set_style('white')

from scipy.optimize import curve_fit
from scipy.special import erfinv, erf
from scipy.misc import factorial

from sympy import latex, symbols, exp
from sympy.utilities.lambdify import lambdify

import inspect
import re


def damage_control(x, dx):
    """
    given input numpy arrays x and dx (corresponding)
    to means and uncertainties, set x->0 and dx->np.inf
    in the case that x is np.inf or np.nan
    """
    if x.__class__.__name__ != 'ndarray':
        x = np.array(x)
    if dx.__class__.__name__ != 'ndarray':
        dx = np.array(dx)
        
    condition = np.where(
                    np.logical_or(
                    np.logical_or(
                    np.logical_or(
                    np.logical_or(np.isinf(x),
                                  np.isnan(x)),
                                  np.isinf(dx)),
                                  np.less_equal(dx, 0)),
                                  np.isnan(dx)))

    x[condition] = 0
    dx[condition]= np.inf
    
    return
    
def weighted_mean(x,dx,axis=None, keepdims=False):
    """
    compute the weighted mean of the input data x with
    repect to the weights dx along the specified axis.
    (if no axis is specified it averages over all data)
    """
    damage_control(x,dx)
    variance = 1/np.sum(1/(dx**2),axis=axis, keepdims=keepdims)
    mean     = np.sum(x/(dx**2),axis=axis, keepdims=keepdims)*variance
    return mean, np.sqrt(variance)

def unweighted_mean(x,dx=None,axis=None, keepdims=False):
    """
    compute the unweighted mean of the input data. If no errors are
    input, then the standard error in the mean is computed. If errors
    are input, the output errors are determined by the input errors.
    """
    if dx is not None:
        damage_control(x,dx)
    
    y  = np.mean(x, axis=axis, keepdims=keepdims)
    
    if axis is None:
        N = np.prod(x.shape)   
    else:
        N = x.shape[axis]    
    
    if dx is None:        
        dy = np.std(x,                              # calculate standard error in the mean
                    axis=axis, 
                    keepdims=keepdims, 
                    ddof=1)/(np.sqrt(N))  # ddof=1 provides an unbiased estimate of the std
    else:
        dy = np.sqrt(np.sum(dx**2, axis=axis, keepdims=keepdims))/N           # calculate the error
        
    return y, dy

def t_statistic(x, dx, axis=None, flatten=True):
    y, dy = weighted_mean(x, dx, axis=axis, keepdims=True)
    t = (x - y) / dx
    if flatten:
        t = np.flatten(t)
    return t

def chi_squared(x, dx, mean_axis=None, chi2_axis=None, keepdims=False, ddof=1, return_t=False, subtract_mean=True):
    """
    calculates chi^2 and dchi^2. If ddof=1, then this function evaluates the
    waited mean and then subtracts that mean from x, accross mean_axis. If ddof != 1
    then it is assumed that the model has already been subtracted from x and t=x/dx.
    Then it computes chi^2 across the chi2_axis.
    """
    
    damage_control(x, dx)
    
    if ddof == 1 and subtract_mean:
        t = t_statistic(x, dx, axis=mean_axis, flatten=False)
    else:
        t = x/dx
    
    if chi2_axis is None:
        N = np.prod(x.shape)
    else:
        N = t.shape[chi2_axis]
        
    chi2 = np.sum(t**2, axis=chi2_axis, keepdims=keepdims)/float(N - ddof)
    try:
        dchi2 = np.sqrt(2/float(N - ddof))
    except:
        dchi2 = np.nan
    
    if return_t:
        return t, chi2, dchi2
    else:
        return chi2, dchi2
    
def correlation_coefficient(x, y, dx=None, dy=None, 
                            axis=None, keepdims=False, 
                            alpha=.317310, error_type='symmetric'):
    
    if dx is None:
        dx = np.ones(x.shape)
    if dy is None:
        dy = np.ones(y.shape)
    
    x_mean, dx_mean = weighted_mean(x, dx, axis=axis, keepdims=keepdims)
    y_mean, dy_mean = weighted_mean(y, dy, axis=axis, keepdims=keepdims)
    
    chi2x, dchi2x = chi_squared(x, dx)
    chi2y, dchi2y = chi_squared(y, dy)
    
    dx = dx * np.sqrt(chi2x)
    dy = dy * np.sqrt(chi2y)
    
    r = np.sum((x - x_mean) * (y - y_mean) / (dx * dy), 
               axis=axis, keepdims=keepdims)/(len(x) - 1)
    
    zr = (1/2.) * np.log((1 + r) / (1 - r))
    za = (-erfinv(alpha - 1)) * np.sqrt(2) / np.sqrt(len(x)-3)
    
    ru = np.tanh(zr + za)
    rl = np.tanh(zr - za)
    
    dr = (ru - rl) / 2
    
    if error_type == 'symmetric':
        return r, dr
    elif error_type is None:
        return r
    elif error_type == 'asymmetric':
        return r, rl, ru
    elif error_type == 'all':
        return r, dr, rl, ru
    
def autocorrelation(x, dx=None, offset=1, offsets=None, **kwargs):
    
    if offsets is None:
        offsets = [offset]
        
        
    a  = []
    da = []
    
    if dx is None:
        dx = np.ones(x.shape)
    
    for offset in offsets:
    
        if offset == 0:
            r, dr = 1, 0
        else:
            r, dr = correlation_coefficient(x[offset:], x[:-offset], 
                                            dx=dx[offset:], dy=dx[:-offset],
                                            error_type='symmetric', 
                                            **kwargs)
        
        a.append(r)
        da.append(dr)
        
    a = np.array(a)
    da = np.array(da)
    
    if len(a) == 1:
        a  = a[0]
        da = da[0]
        
    return a, da

    
def scientific_string(x, dx, mode='scientific', pm='\pm', style='latex'):
    
    if style == 'latex':
        pl = '{'
        pr = '}'
        pm = '\pm'
        times = '\\times'
    elif style == 'plain':
        pl = '('
        pr = ')'
        pm = '+/-'
        times = 'x'
    else:
        raise IOError('invalid style input')
        
    if np.isnan(x):
        return 'NaN'
    
    if np.isnan(dx):
        return ('%.3f') % (x,)
    
    if x == 0:
        exp = 0
    else:
        exp  = np.floor(np.log10(abs(x)))
    dexp = np.floor(np.log10(abs(dx)))
    
    if mode == 'scientific':
        if int(exp) in [-1,0,1]:
            sigs = np.abs(exp-dexp) + 1
            X  = ('%.' + str(int(exp-dexp+2)) + 'g') % (x,)
            dX = ('%.' + str(2) + 'g') % (dx,)
            string = '(' + X + pm + dX +')'
            
        else:
            sigs = int(np.abs(exp - dexp) + 1)
            x  = x/10**exp;
            dx = dx/10**exp;
            X  = ('%.' + str(sigs) + 'f') % (x,)
            dX = ('%.' + str(sigs) + 'f') % (dx,)
            string= ('(' + X + pm + dX + ')' + times + 
                            '10^' + pl + str(int(exp)) + pr)
            
    elif mode == 'normal':
        if dexp < 0:
            sigs = int(np.abs(dexp))
        else:
            sigs = 0
            
        X  = ('%.' + str(sigs) + 'f') % (x,)
        dX = ('%.' + str(sigs) + 'f') % (dx,)
        
        string= X + pm + dX
    else:
        raise IOError('youre input mode is not valid.')
        
    return string
        
    
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2/(2*sigma**2))

def sin_cos_to_amp_phi(s, c):
    """
    calculates amplitude, phase, and jacobian matrix given the
    sine and cosine amplitudes.
    """
    amp = np.sqrt(s**2 + c**2)
    phi = np.arctan2(s, c)
    
    J = np.array([[ s / np.sqrt(s**2 + c**2), c / np.sqrt(s**2 + c**2)],
                  [ c / (s**2 + c**2),       -s / (s**2 + c**2)       ]])
    
    return amp, phi, J
    

def t_hist(x, dx, axis=None, option='normal', 
           ddof=1, bins=None , yscale='linear', subtract_mean=True):
    
    colors = sns.color_palette()
    #obtain the t statistic and chi^2
    t, chi21, dchi21 = chi_squared(x, dx, mean_axis=axis, 
                                 ddof=ddof, return_t=True, subtract_mean=True)
    #flatten the t-statistic array:
    t = t.flatten()
    
    #compute the range of the data:
    sigma = float(1)
    limit = np.sqrt(2)*erfinv(1-sigma/len(t))
    
    if bins is None:
        num = int(np.round(np.log(len(x) + 100)**3/15))
        bins = np.linspace(-limit,limit,num)
    elif not(isinstance(bins,(tuple,list))):
        bins = np.linspace(-limit,limit,bins)
        
    hist = np.histogram(t,bins=bins)
    
    bins = (hist[1][0:-1] + hist[1][1:]) / 2
    hist = hist[0]

    bins=np.array(bins)
    hist=np.array(hist)
    
    step = bins[1] - bins[0]
        
    coeffs, cov = curve_fit(gauss, bins, hist, p0=[max(hist),0,1])
    xplot = np.linspace(-limit, limit, 200)
    fit = gauss(xplot, coeffs[0], coeffs[1], coeffs[2])
    norm = gauss(bins, 1, 0, 1)
    expect = sum(hist) * gauss(xplot, 1, 0, 1) / sum(norm)
    norm   = sum(hist) * norm / sum(norm)
    
    chi22, dchi22 = chi_squared(hist - norm, np.sqrt(hist), ddof=0, subtract_mean=True)
                                 
    chi2string1 = scientific_string(chi21, dchi21)
    chi2string2 = scientific_string(chi22, dchi22)
    label='$\\chi^2_\\mathrm{fit} = ' + chi2string1 + ', \\chi^2_\\mathrm{dist} = ' + chi2string2 + '$'
        
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    ax.fill_between(xplot, expect + np.sqrt(expect), expect - np.sqrt(expect), color=colors[0], linewidth=3, alpha=.05)
    ax.plot(xplot, expect, '-', color=colors[0], linewidth=3, alpha=.05)
    # ax.plot(xplot, fit, '-r', linewidth=3, alpha=.3, label='gaussian fit')
    ax.errorbar(bins, hist, yerr=np.sqrt(hist), fmt='.', alpha=.75, label=label, color=colors[0])
    ax.set_xlabel('$(x-\\left< x \\right>)/\\delta x$')
    ax.set_ylabel('counts')
    ax.legend(loc='best')
        
    ax.set_xlim([-limit, limit])
    ax.set_ylim([.1, max(hist)*(5/4.)])
                
    if yscale == 'log':
        ax.set_yscale('log')
         
    return

def split_dataset(data, split_fraction=.75):
    """
    takes a dataset, and splits it into several test, train sets.
    """
    #select a random permutation
    perm = list(np.random.permutation(len(data)))
    
    selections = []
    
    #split up the permutation into groups
    small_size = int(np.floor(len(y)*min([split_fraction, 1-split_fraction])))
    groups     = len(y)//small_size
    
    perms = [(perm[i * small_size: (i + 1) * small_size], 
              perm[0: i * small_size] + perm[(i + 1) * small_size:]) for i in range(groups)]
    
    #train_datasets      = [Data(x_functions=None)]
    #validation_datasets = 
    
    X = [([a[perm[0]] for a in x],[a[perm[1]] for a in x]) for perm in perms]
    Y = [(y[perm[0]], y[perm[1]]) for perm in perms]
    dY= [(dy[perm[0]], dy[perm[1]]) for perm in perms]
    
    return X, Y, dY

def linear_regression(x_functions, y, dy=None, weights=None, return_c=False):
    """
    this is the core linear regression function
    """
    #initiallize the c vector and M matrix:
    if x_functions.__class__.__name__ != 'ndarray':    
        X = np.array(x_functions)
    else:
        X = x_functions
    
    if dy is None and weights is None:
        raise IOError('you must specify dy or weights')
    if dy is not None:
        weights = 1 / dy**2
        
    c = np.dot(X, y * weights)
    M = np.dot(np.dot(X, np.diag(weights)), X.T)
        
    #take the inverse of the M matrix 
    covariance = np.linalg.inv(M)

    #evaluate the best fit coefficients:
    coefficients  = np.dot(covariance, c)
    dcoefficients = np.sqrt(np.diagonal(covariance))
    
    if return_c:
        return covariance, c
    else:
        return coefficients, dcoefficients, covariance

def linear_regression_with_linear_constraints(x_functions, y, dy, constraints=None):
    """
    this is the same as the previous core linear regression function
    but with added complexity to include linear constraints. The problem
    is still solved by matrix inversion, but is a bit more complicated.
    """
    
    if constraints is None:
        return linear_regression(x_functions, y, dy)
    else:
        if constraints.__class__.__name__ != 'Constraints':
            constraints = Constraints(constraints)
        
        Minv, c = linear_regression(x_functions, y, dy,
                                          return_c=True)
        
        dim = Minv.shape[0]
        
        S = constraints.S.T # this is the rectangular matrix of constraint coefficients
        d = constraints.d # this is the vector of constraint constants
        
        SMS    = np.dot(np.dot(S.T, Minv), S)
        SMSinv = np.linalg.inv(SMS)
        Minv_S_SMSinv = np.dot(np.dot(Minv, S), SMSinv)
        
        P = np.eye(dim) - np.dot(Minv_S_SMSinv, S.T)
        
        coefficients = np.dot(np.dot(P, Minv), c) + np.dot(Minv_S_SMSinv, d)
        covariance   = np.dot(np.dot(P, Minv), P.T)
        dcoefficients= np.sqrt(np.diagonal(covariance))

        return coefficients, dcoefficients, covariance

def evaluate_linear_regression_model(x_functions, coefficients, dcoefficients, covariance):
    if x_functions.__class__.__name__ != 'ndarray':    
        X = np.array(x_functions)
    else:
        X = x_functions
        
    y_est  = np.dot(X.T, coefficients)
    dy_est = np.sqrt(np.diagonal(np.dot(np.dot(X.T, covariance), X)))
    
    return y_est, dy_est

def normal_p_to_n_sigma(p, N=1):
    """
    what is the boundary n_sigma that I can place such that there is a
    probability p that 1 or more of the N datapoints lie outside of the +/-n_sigma range?
    """
    return np.sqrt(2) * erfinv((1 - p)**(1 / float(N)))

def normal_n_sigma_to_p(n, N=1):
    return (1 - erf(n/np.sqrt(2))**N)

class Constraints(object):
    
    def __init__(self, constraints=None, coefficient_names=None):
        """
        input constraints in the following way. For a set of constraints:
        
        a_0 + 2 * a_2 - 3 * a_4 == 1
        a_0 + a_2 == 0
        
        for linear regression with 6 dimensions, input:
        
        constraint_1 = ([1, 0, 2, 0, 3, 0], 1)
        constraint_2 = ([1, 0, 1, 0, 0, 0], 0)
        
        constraints = Constraints([constraint_1, constraint_2])
        """
        self.constraints = constraints
        self.S = np.array([constraint[0] for constraint in constraints])
        self.d = np.array([constraint[1] for constraint in constraints])
        self.coefficient_names = coefficient_names
        
    def __str__(self):
        return None
    
    def __repr__(self):
        return str(self)


class Data(object):
    
    def __init__(self, x=None, y=None, dy=None):
        
        if x is None:
            self.x = None
            self.y = None
            self.dy= None
        else:
            self.x = np.array(x)
            self.y = np.array(y)
            self.dy= np.array(dy)
            
            if len(x.shape) > 2:
                raise IOError('x can be 1 or 2 dimensional')
            elif len(x.shape) == 1:
                self.x = np.expand_dims(self.x, 0)
                
        
    def __len__(self):
        if self.y is None:
            return 0
        else:
            return len(self.y)
        
    def append(self,data, inds=None):
        
        if inds is None:
            inds = list(range(len(data)))
            
        if len(data.x.shape) > 1:
            x = data.x[:, inds]
        else:
            x = data.x[inds]
        
        if self.x is None:
            self.x = x
            self.y = data.y[inds]
            self.dy= data.dy[inds]
        else:
            self.x = np.append(self.x, x)
            self.y = np.append(self.y, data.y[inds])
            self.dy = np.append(self.dy, data.dy[inds])
                
        return
                
    def delete(self, inds):
        
        self.x = np.delete(self.x, inds, axis=-1)
        self.y = np.delete(self.y, inds)
        self.dy= np.delete(self.dy, inds)
        
        return
    
    def split(self, split_fraction=.75):
        """
        takes a dataset, and splits it into several test, train sets.
        """
        #select a random permutation
        perm = list(np.random.permutation(len(self)))

        selections = []

        #split up the permutation into groups
        small_size = int(np.floor(len(self)*min([split_fraction, 1-split_fraction])))
        groups     = len(self)//small_size

        perms = [(perm[i * small_size: (i + 1) * small_size], 
                  perm[0: i * small_size] + perm[(i + 1) * small_size:]) for i in range(groups)]
        
        create_data_sets = lambda i: [Data(x=np.array([a[perm[i]] for a in self.x]),
                                           y=self.y[perm[i]],
                                           dy=self.dy[perm[i]]) 
                                                                  for perm in perms]
        
        test  = create_data_sets(0)
        train = create_data_sets(1)

        return test, train


class RegressionModel(object):
    """
    this is intended to be a meta-class for various types of regression
    that I will commonly use
    """
    
    model_type = 'meta-class for generic linear regression'
    
    def __init__(self, x, y, dy=None, data=None,
                 label=None, renormalize_error=False, plot_errorbars=None, 
                 auto_fit=True, constraints=None, **kwargs):
        
        self._state = 'initialized'
        
        # extract data from dataframe if input.
        self.dataframe_input = (data is not None)
        if self.dataframe_input:
            x, y, dy = self.from_dataframe(x, y, dy, data)
            
        
        # modify parameters contingent on inputting errorbars
        if dy is None:
            renormalize_error = True
            dy = np.ones(y.shape)
            
            if plot_errorbars is None:
                plot_errorbars = False
        else:
            if plot_errorbars is None:
                plot_errorbars = True
                
        # manage possible problems with forms of the input data
        damage_control(y, dy)
        x = np.array(x)
        
        # save regression options to fields
        self._renormalize_error= renormalize_error
        self.plot_errorbars    = plot_errorbars
        self.label             = label
        self.auto_fit          = auto_fit

        # save the data to an internal field
        self.data = Data(x, y, dy)

        # initialize outlier fields
        self.outliers = Data()
        
        try:
            self.ddof
        except:
            self.ddof = self.evaluate_x_functions(self.data.x).shape[0]
        
        self.set_model_names(**kwargs)
        
            
    def from_dataframe(self, x, y, dy, data):
        """
        grab data out of the dataframe is a dataframe is input.
        """
        if not(isinstance(x, (str, list, tuple))) or not(isinstance(y, str)):
            raise IOError("""x, y must be strings corresponding to dataframe
                             column names when a dataframe is input""")
        data = data.copy()
        if isinstance(x, (list, tuple)):
            for i in range(len(x)):
                data = data[data[x[i]].notnull()]
        else:
            data = data[data[x].notnull()]
        data = data[data[y].notnull()]
        if dy is not None:
            data = data[data[dy].notnull()]
        if isinstance(x, (list, tuple)):
            new_x = []
            for i in range(len(x)):
                new_x.append(data[x[i]])
            x = new_x
        else:
            x = data[x]
        y = data[y]
        if dy is not None:
            dy = data[dy]
        
        return x, y, dy
    
                        
    def set_model_names(self, basis_names=None, 
                              coefficient_names=None, 
                              coefficient_units=None,
                              y_name='y',
                              y_unit=None, **kwargs):
        
        # for the case where I have sympy model definitions:
        try:
            coefficient_names = [self.variables[i].name 
                                 for i in range(1,len(self.variables))]
        except:
            pass
        
        if coefficient_names is None:
            if basis_names is not None:
                coefficient_names = [('a_{' + basis_names[i] + '}') 
                                         for i in range(self.ddof)]
            else:
                coefficient_names = [('a_{' + str(i) + '}') 
                                             for i in range(self.ddof)]
        if basis_names is None:
            basis_names =       [('x_{' + str(i) + '}') 
                                         for i in range(self.ddof)]
        if coefficient_units is None:
            coefficient_units = ['' for i in range(self.ddof)]

        self.coefficient_names = coefficient_names
        self.basis_names       = basis_names
        self.coefficient_units = coefficient_units
        self.y_name            = str(y_name)
        self.y_unit            = str(y_unit)
        self.model             = self.model_string()
        return
    
    def model_string(self, style='plain'):
        try:
            if style == 'plain':
                return str(self.expression)
            elif style == 'latex':
                return str(latex(self.expression))
        except:
            model = [(a + b) for a, b in 
                     zip(self.coefficient_names, self.basis_names)]
            return self.y_name + ' = ' + ' + '.join(model)
    
    def model_string_with_coefficients(self, style='plain'):
        coefficient_strings = [scientific_string(x, dx, style=style) 
                               for x, dx in 
                               zip(self.coefficients, self.dcoefficients)]
        if style == 'plain':
            p = lambda b: (('(' + b + ')') if b != '' else '')
            model = [(a + p(b) + c ) for a, b, c in zip(coefficient_strings, 
                                                    self.coefficient_units, 
                                                    self.basis_names)]
        elif style == 'latex':
            p = lambda b: (('(\\mathrm{' + b + '})') if b != '' else '')
            model = [(a + p(b) + c ) for a, b, c in zip(coefficient_strings, 
                                                    self.coefficient_units, 
                                                    self.basis_names)]
        else:
            raise IOError('')
                        
        output = self.y_name + ' = ' + ' + '.join(model)
        if style == 'latex':
            output = '$' + output + '$'
        
        return output
        
        
    def evaluate_x_functions(self, x):
        try:
            if len(x.shape) == 2 and x.shape[0] == 1:
                x = x[0,:]
            return np.array(self.model_function(x))
        except:
            return x
    
    
    def __len__(self):
        return len(self.data)
    
    def get_coefficients(self):
        # create a dataframe with the coefficients
        t = np.abs(self.coefficients/self.dcoefficients)
        df = pd.DataFrame({'names': self.coefficient_names,
                           'units': self.coefficient_units,
                           'value': self.coefficients,
                           'dvalue': self.dcoefficients,
                           'significance': t,
                           'p(zero)': normal_n_sigma_to_p(t, N=len(t))})
        # reorder the columns
        df = df[['names','units','value','dvalue', 
                 'significance', 'p(zero)']]
        
        return df
        
    def __str__(self):
        header = ''.join([':']*25 + [' REGRESSION MODEL '] + [':']*25)
        if self._state == 'initialized':
            return header
        elif self._state == 'fit':

            df = self.get_coefficients()
            
            output = [header,
                      'label:\t\t\t' + str(self.label),
                      'type:\t\t\t' + str(self.model_type) + ' with ' + str(self.ddof) + ' parameters',
                      'model:\t\t\t' + str(self.model),
                      'chi^2:\t\t\t' + scientific_string(self._chi2, 
                                                       self._dchi2, 
                                                       style='plain'),
                      'error renormalized?:\t' + str(self._renormalize_error),
                      '# datapoints:\t\t' + str(len(self.data)),
                      '# outliers:\t\t' + str(len(self.outliers)),
                      'coefficients:',
                      str(df)]
            return '\n\n'.join(output)
        else:
            return None
    
    def __repr__(self):
        return str(self)
    
    def renormalize_error(self, coefficients=True):
        """
        take all of the fit parameters that are denote variance, and rescale
        by chi^2
        """
        self._renormalize_error = True
        
        if self._state == 'fit':
            if coefficients:
                self.dcoefficients *= np.sqrt(self.chi2)
                self.covariance    *= self.chi2

            self.dy_est        *= np.sqrt(self.chi2)
            self.data.dy       *= np.sqrt(self.chi2)
            self.dchi2         /= self.chi2
            self.chi2           = 1
            
        return
    
    def unnormalize_error(self, coefficients=True):
        """
        if the error was previously normalized, it can be renormalized here.
        (the original chi2 is saved in self._chi2)
        """
        self._renormalize_data = False
        
        if self._state == 'fit':
            if coefficients:
                self.dcoefficients /= np.sqrt(self._chi2)
                self.covariance    /= self._chi2

            self.dy_est        /= np.sqrt(self._chi2)
            self.data.dy       /= np.sqrt(self._chi2)
            self.dchi2         *= self._chi2
            self.chi2           = self._chi2
            
        return
    
    def histogram(self, **kwargs):
        return t_hist(self.data.y - self.y_est, self.data.dy, 
                      ddof=self.ddof, subtract_mean=False, **kwargs)
    
    def residuals(self):
        return self.data.y - self.y_est
    
    def data_limits(self, residuals=False):
        
        if residuals:
            y = self.residuals()
        else:
            y = self.data.y
            
        if self.plot_errorbars:
            ylim = [min(y - self.data.dy), max(y + self.data.dy)]
        else:
            ylim = [min(y), max(y)]
        if len(self.data.x.shape) > 1:
            xlim = []
            for i in range(self.data.x.shape[0]):
                xlim.append([min(self.data.x[i]), max(self.data.x[i])])
            if len(xlim) == 1:
                xlim = xlim[0]
        else:
            xlim = [min(self.data.x), max(self.data.x)]
            
        return xlim, ylim
        
    def remove_outliers(self, p=.25):
        """
        remove outliers from the dataset with a probability of p
        that one or more data points will be falsely accused of being outliers
        """
        while True:
            N = len(self.data.y)
            n_sigma = normal_p_to_n_sigma(p, N)
            inds = np.where(np.abs((self.data.y - self.y_est) / self.data.dy) > n_sigma)
            if len(inds[0]) == 0:
                break
            self.outliers.append(self.data, inds)
            self.data.delete(inds)
            self.fit()
        return
    
class LinearRegressionModel(RegressionModel):
    
    model_type = 'meta class for linear regression'
    
    def __init__(self, x, y, dy=None, constraints=None, model_function=None, **kwargs):
        
        #x = np.array(x)
        #self.model_function = model_function
        #x_functions = self.evaluate_x_functions(x)
        #print x_functions.shape
        #self.ddof   = x_functions.shape[1]
        
        super(LinearRegressionModel, self).__init__(x, y, dy, model_function=model_function,**kwargs)
        
        self.set_constraints(constraints, self.coefficient_names)
        
        if self.auto_fit:
            self.fit()
        
        
    def set_constraints(self, constraints=None, coefficient_names=None):
        if constraints is None:
            self.constraints = None
        elif constraints.__class__.__name__ == 'Constraints':
            if constraints.coefficient_names is None:
                constraints.coefficient_names = coefficient_names
            
            self.constraints = constraints
        else:
            self.constraints = Constraints(constraints, coefficient_names)
            
        if self.auto_fit and self._state == 'fit':
            self.fit()
        return
    
    def evaluate_model(self, x=None, x_functions=None, transformation=None):
        if x_functions is None:
            if x is None:
                x_functions = self.evaluate_x_functions(self.data.x)
            else:
                x_functions = self.evaluate_x_functions(x)
        if transformation is not None:
            x_functions = transformation(x)
                
        return evaluate_linear_regression_model(x_functions, 
                                                self.coefficients, 
                                                self.dcoefficients, 
                                                self.covariance)
    
    def fit(self):
        
        x_functions = self.evaluate_x_functions(self.data.x)
        
        self.coefficients, self.dcoefficients, self.covariance = \
                            linear_regression_with_linear_constraints(x_functions, 
                                                                      self.data.y, 
                                                                      self.data.dy,
                                                                      constraints=self.constraints)
            
        self.y_est, self.dy_est = self.evaluate_model()
        
        self.chi2, self.dchi2 = chi_squared(self.data.y - self.y_est, self.data.dy,
                                            ddof=self.ddof, subtract_mean=False)
        
        self._chi2 = self.chi2 # this is stored so that i can undo a normalization of chi_2 to 1
        self._dchi2 = self.dchi2
        
        self._state = 'fit'
        
        if self._renormalize_error:
            self.renormalize_error()
            
        return
        
        
class NonLinearFit1D(RegressionModel):
    
    model_type = '1D non-linear regression model'
    
    def __init__(self, x, y, dy=None, start_point=None, 
                 variables=None, expression=None, 
                 model_function=None, model_function_derivative=None, **kwargs):
        
        # we can directly input the model function and the model function derivative,
        # or we can input sympy variables and a sympy expression
        
        self.generate_model_function(variables, expression, 
                                     model_function, model_function_derivative)
        
        self.ddof = len(inspect.getargspec(self.model_function).args) - 1
        
        if start_point is None:
            start_point = [1 for i in range(self.ddof)]
        self.start_point = start_point
        
        super(NonLinearFit1D, self).__init__(x, y, dy, **kwargs)
        
        if self.auto_fit:
            self.fit()
            
    def generate_model_function(self, variables, expression, 
                                model_function, model_function_derivative):
        """
        generate a model function and generate the model function derivative
        from input sympy expressions.
        """
        
        if (variables is not None) and (expression is not None):
            model_function = lambdify(variables, expression, modules='numpy')

            derivatives = [lambdify(variables, 
                                    expression.diff(variables[i]), 
                                    modules='numpy')
                                    for i in range(1, len(variables))]

            def model_function_derivative(x, *args):
                output = []
                for d in derivatives:
                    term = np.array(d(x, *args))
                    if term.shape == x.shape:
                        output.append(term)
                    else:
                        output.append(term * np.ones(x.shape))
                return np.array(output)
        
        self.expression = expression
        self.variables  = variables
        self.model_function = model_function
        self.model_function_derivative = model_function_derivative
        return
            
    def evaluate_model(self, x=None):
        if x is None:
            x = self.data.x
        x = np.squeeze(x)
        y_est = self.model_function(x, *self.coefficients)
        deriv = self.model_function_derivative(x, *self.coefficients)
        dy_est = np.sqrt(np.array([np.dot(np.dot(deriv[:,i].T, self.covariance), deriv[:,i]) 
                                   for i in range(deriv.shape[1])]))
        
        return y_est, dy_est
    
    def fit(self):
        
        x = np.squeeze(self.data.x)

        self.coefficients, self.covariance = curve_fit(self.model_function, 
                                                       x, self.data.y, sigma=self.data.dy, 
                                                       p0=self.start_point,
                                                       absolute_sigma=True)
        
        self.dcoefficients = np.sqrt(np.diag(self.covariance))
            
        self.y_est, self.dy_est = self.evaluate_model()
        
        self.chi2, self.dchi2 = chi_squared(self.data.y - self.y_est, self.data.dy,
                                            ddof=self.ddof, subtract_mean=False)
        
        self._chi2 = self.chi2 # this is stored so that i can undo a normalization of chi_2 to 1
        self._dchi2 = self.dchi2
        
        self._state = 'fit'
        
        if self._renormalize_error:
            self.renormalize_error()
            
        return
    
    def plot_labels(self, which, style='latex'):
        if which == 'x':
            return 'x'
        elif which == 'y':
            return 'y'
    
    def plot(self, **kwargs):
        return regression_plot(self, **kwargs)
    
    
class GaussianFit1D(NonLinearFit1D):
    
    model_type = '1d non-linear gaussian fit'
    
    def __init__(self, x, y, dy=None, start_point=None,
                 offset=False, **kwargs):
        
        x1, a, x0, s, c = symbols('x a x0 s c')
        
        if offset:
            variables  = (x1, a, x0, s, c)
            expression = a * exp(-(x1 - x0)**2 / (2 * s**2)) + c
        else:
            variables = (x1, a, x0, s)
            expression = a * exp(-(x1 - x0)**2 / (2 * s**2))
      
        if start_point is not None:
            start_point = start_point[:len(variables) - 1]
        else:
            start_point = [max(y) - min(y), np.mean(x), (max(x) - min(x))/4]
            if offset:
                start_point.append(min(y))
        
        super(GaussianFit1D, self).__init__(x, y, dy=dy, start_point=start_point,
                                            variables=variables, 
                                            expression=expression,
                                            **kwargs)
        
    
class LinearRegression(LinearRegressionModel):
    
    model_type = 'generic linear regression'
    
    def __init__(self, x, y, dy=None, **kwargs):
        
        super(LinearRegression, self).__init__(x, y, dy=dy, **kwargs)


class LinearRegressionND(LinearRegressionModel):
    
    model_type = 'generic Nd linear regression'
    
    def __init__(self, x, y, dy=None, model_function=None,
                 x_names=None, x_units=None, **kwargs):
        
        if self.model_function is None:
            raise IOError('you must input a model function')
        
        self.model_function = model_function
        self.dims = len(x)
        
        if x_units is None:
            x_units = [None] * self.dims
        if x_names is None:
            x_names = [None] * self.dims
        
        self.x_units = x_units
        self.x_names = x_names
                                            
        super(LinearRegressionND, self).__init__(x, y, dy=dy, **kwargs)
    
    
class LinearRegression2D(LinearRegressionModel):
    
    model_type = 'generic 2d linear regression'
    
    def __init__(self, x, y, dy=None, model_function=None,
                 x_names=['X', 'Y'], x_units=None, y_name='Z', **kwargs):
        
        if x_units is None:
            x_units = [None] * 2
        
        self.x_units = x_units
        self.x_names = x_names
        
        if model_function is None:
            raise IOError('you must insert a model function')
        
        self.model_function = model_function
                                            
        super(LinearRegression2D, self).__init__(x, y, dy=dy, 
                                                 y_name=y_name, **kwargs)
                                            
    
    def plot_labels(self, which, style='latex'): 
        
        if style == 'latex':
            s = lambda i: '\, (\\mathrm{' + str(i) + '})'
        else:
            s = lambda i: ' ($' + str(i) + '$)'
            
        labels = {'x':[str(self.x_names[0]), str(self.x_units[0])],
                  'y':[str(self.x_names[1]), str(self.x_units[1])],
                  'z':[str(self.y_name), str(self.y_unit)]}
        
        string = labels[which][0]
        if labels[which][1] != 'None':
            string = string + s(labels[which][1])
            
        return string
        
    
    def plot(self, **kwargs):
        return plot3D(self, **kwargs)
    

class PolynomialFit2D(LinearRegression2D):
    
    model_type = '2d polynomial linear regression'
    
    def __init__(self, x, y, dy=None, order=1, orders=None, 
                 max_sum=True, **kwargs):
        # max_sum=True means that the maximum sum of the product
        # of all exponents in a sum is equal to order, so for order=1
        # we would have a model 1 + x + y. Otherwise,
        # order corresponds to the highest exponent for each variable
        # so that order = 1 corresponds to 1 + x + y + x * y
        
        if orders is None:
            if max_sum:
                orders = [(i, j) for i in range(order + 1) 
                                 for j in range(order + 1) if i + j <= order]
            else:
                orders = [(i, j) for i in range(order + 1) 
                                 for j in range(order + 1)]
    
        def model_function(x):
            x_functions = []
            for exponents in orders:
                term = np.ones(x[0].shape)
                for i in range(len(exponents)):
                    term *= x[i]**exponents[i]
                x_functions.append(term)
            return x_functions
            
        self.orders = orders
        
        basis_names, coefficient_units = self.generate_names(**kwargs)

        super(PolynomialFit2D, self).__init__(x, y, dy=dy,
                                              model_function=model_function, basis_names=basis_names,
                                              coefficient_units=coefficient_units, **kwargs)
                                              
    def generate_names(self, x_names=['$x$', '$y$'], y_name='$z$', 
                       x_units=None, y_unit=None, **kwargs):
        
        # here are two functions that I can use to create units
        num_dem = lambda a, p: ''.join([str(k) + '^{' + str(abs(a[k])) + '}' 
                                       if abs(a[k]) > 1 else str(k)
                                       for k in a.keys() if a[k] * p > 0])
        
        fract  = lambda a: '$\\frac{' + num_dem(a, 1) + '}{' + num_dem(a, -1) + '}$'
        
        basis_names = []
        coefficient_units = []
        
        for order in self.orders:
            if order[0] == 0 and order[1] == 0:
                basis_names.append('')
                coefficient_units.append(str('$' + y_unit + '$'))
            else:
                
                term = ''
                units = {}
                units[x_units[0]] = 0
                units[x_units[1]] = 0
                units[y_unit] = 1

                for i in range(len(order)):
                    if order[i] == 1:
                        term = term + x_names[i]
                        units[x_units[i]] += -1
                    elif order[i] > 1:
                        term = term + x_names[i] + '$^{' + str(order[i]) + '}$'
                        units[x_units[i]] += -order[i]
                        
                basis_names.append(term)
                coefficient_units.append(fract(units))
                
        if x_units is None or y_unit is None:
            coefficient_units = [''] * len(self.orders)
        
        return basis_names, coefficient_units
    
    
    def model_string_with_coefficients(self, style='latex', p=.05, 
                                       only_significant_terms=True, max_line_length = 100):

        num = ['$' + scientific_string(coeff, dcoeff) + '$' 
               for coeff, dcoeff in zip(self.coefficients, self.dcoefficients)]
        
        terms = [n + '(' + unit + ')' + basis_name if unit != '' else n + basis_name
                 for n, unit, basis_name in zip(num, self.coefficient_units, self.basis_names)]
        
        n_sigma = np.abs(self.coefficients/self.dcoefficients)
        inds = np.flipud(np.argsort(n_sigma))
        if not(only_significant_terms):
            p = 1.0
        significant = (normal_n_sigma_to_p(n_sigma, N=len(self.coefficients)) <= p)
        all_terms = np.all(significant)
        significant_terms = [terms[i] for i in inds[significant]]

        title = '$+$'.join(significant_terms)

        lines = []
        total = 0
        line = []
        for term in significant_terms:
            total += len(term)
            if total > max_line_length:
                lines.append(line)
                line = [term]
                total = len(term)
            else:
                line.append(term)

        title = self.y_name + '$=$' + '$+$\n'.join(['$+$'.join(line) for line in lines])
        if not(all_terms):
            title = title + '$+$ ...'

        return title

    
class LinearRegression1D(LinearRegressionModel):
    
    model_type = 'generic 1d linear regression'
    
    def __init__(self, x, y, dy=None, model_function=None, 
                 x_name='x', x_unit=None, **kwargs):
        
        self.x_unit = str(x_unit)
        self.x_name = str(x_name)
            
        if model_function is None:
            raise IOError('you must input a model function')
            
        self.model_function = model_function
        
        super(LinearRegression1D, self).__init__(x, y, dy=dy, **kwargs)
        
    
    def plot_labels(self, which, style='latex'):
        
        if style == 'latex':
            s = lambda i: '\, (\\mathrm{' + str(i) + '})'
        else:
            s = lambda i: ' (' + str(i) + ')'
        
        if which == 'x':
            string = self.x_name
            if self.x_unit != 'None':
                string = string + s(self.x_unit)
        elif which == 'y':
            string = self.y_name
            if self.y_unit != 'None':
                string = string + s(self.y_unit)
        elif which == 'residuals':
            string = '\\mathrm{residuals}'
            if self.y_unit != 'None':
                string = string + s(self.y_unit)
        else:
            raise IOError('')
            
        if style == 'latex':
            return '$' + string + '$'
        else:
            return string
            
        
    def plot(self, legend=True, fit_expression=True, **kwargs):
        return regression_plot(self, legend=legend, fit_expression=fit_expression, **kwargs)
    
    
class PolynomialFit1D(LinearRegression1D):
    
    model_type = '1d polynomial linear regression'
    
    def __init__(self, x, y, dy=None, order=1, orders=None, 
                 x_name='x', x_unit=None, y_name='y', y_unit=None, **kwargs):
        
        #set the polynomial orders that are being used
        if orders is None:
            orders = list(range(order + 1))
            
        self.orders = orders
        
        basis_names = []
        coefficient_units = []
        coefficient_names = []
        
        x_unit = str(x_unit)
        y_unit = str(x_unit)
        
        for order in self.orders:
            if order == 0:
                term = ''
                unit = y_unit
            elif order == 1:
                term = x_name
                if y_unit != x_unit:
                    unit = y_unit + '/' + x_unit
                else:
                    unit = ''
            elif order > 1:
                term = x_name + '^' + str(order)
                if y_unit != x_unit:
                    unit = y_unit + '/' + x_unit + '^' + str(order)
                else:
                    unit = '1/' + x_unit + '^' + str(order - 1)
                
            coefficient_names.append('a(' + str(order) + ')')
            basis_names.append(term)
            coefficient_units.append(unit)
            
        if x_unit == 'None' or y_unit == 'None' is None:
            coefficient_units = ['' for i in range(len(orders))]
            
        kwargs['coefficient_names'] = coefficient_names
        kwargs['basis_names']       = basis_names
        kwargs['coefficient_units'] = coefficient_units
        
        model_function = lambda x: np.array([x**order for order in self.orders])
        
        super(PolynomialFit1D, self).__init__(x, y, dy=dy, 
                                              model_function=model_function, 
                                              x_name=x_name, x_unit=x_unit, 
                                              y_name=y_name, y_unit=y_unit, 
                                              **kwargs)
        
    def get_derivative(self, x, n=1):
        derivative = lambda x: np.array([(factorial(self.orders[i]) / 
                                          factorial(self.orders[i] - n)) * 
                                          x[i]**(self.orders[i] - n) 
                                          if self.orders[i] >= n else np.zeros(x.shape)
                                          for i in range(len(self.orders))])
        return derivative
    
    def get_integral(self, x, n=1):
        integral = lambda x: np.array([(factorial(self.orders[i] - n) / factorial(self.orders[i])) *
                                       x[i]**(self.orders[i] + n)
                                       for i in range(len(self.orders))])
        return integral

class SineFit1D(LinearRegression1D):
    
    model_type = '1d sinusoidal linear regression'
    
    def __init__(self, x, y, dy=None, frequencies=None, phases=None, offset=False, 
                 x_name='x', x_unit=None, y_name='y', y_unit=None, **kwargs):
        
        # do some conditioning of the input frequencies and phases:
        if frequencies is None:
            raise IOError('you must input at least one frequency')
            
        try:
            frequencies[0]
        except:
            frequencies = [frequencies]
            
        if phases is None:
            phases = [None] * len(frequencies)
            
        try:
            phases[0]
        except:
            phases = [phases]
            
        if len(phases) != len(frequencies):
            raise IOError('length of frequencies must match length of phases')
        
        frequencies = np.array(frequencies)
        
        # store the model parameters
        self.frequencies = frequencies
        self.phases      = phases
        self.offset      = offset
        
        self._frequencies = []
        self._phases      = []
        if offset:
            self._frequencies.append(0)
            self._phases.append(np.pi/2)
        for i in range(len(frequencies)):
            if phases[i] is None:
                self._frequencies.append(frequencies[i])
                self._frequencies.append(frequencies[i])
                self._phases.append(0)
                self._phases.append(np.pi/2)
            else:
                self._frequencies.append(frequencies[i])
                self._phases.append(phases[i])
                
        self._frequencies = np.array(self._frequencies)
        self._phases = np.array(self._phases)
        
        # define the model function
        def model_function(x):
            x_functions = []
            
            if offset:
                x_functions.append(np.ones(x.shape))
                
            for i in range(len(frequencies)):
                w = 2 * np.pi * frequencies[i]
                if phases[i] == None:
                    x_functions.append(np.sin(w * x))
                    x_functions.append(np.cos(w * x))
                else:
                    x_functions.append(np.sin(w * x + phases[i]))
            
            return np.array(x_functions)
        
        self.ddof = len(model_function(np.array([0])))
        
        basis_names = []
        coefficient_units = []
        coefficient_names = []
        
        if offset:
            coefficient_names.append('c(0)')
            basis_names.append('')
            coefficient_units.append(y_unit)
                
        for i in range(len(frequencies)):
            w = 2 * np.pi * frequencies[i]
            if phases[i] == None:
                coefficient_names.append('s(' + str(i + 1) + ')')
                basis_names.append('\\sin{(\\omega_{' + str(i + 1) + '} ' + x_name + ')}')
                coefficient_units.append(y_unit)
                
                coefficient_names.append('c(' + str(i + 1) + ')')
                basis_names.append('\\cos{(\\omega_{' + str(i + 1) + '} ' + x_name + ')}')
                coefficient_units.append(y_unit)
            else:
                coefficient_names.append('s(' + str(i + 1) + ')')
                basis_names.append('\\sin{(\\omega_{' + str(i + 1) + '} ' + x_name + ' + \\phi_{' + str(i + 1) + '})}')
                coefficient_units.append(y_unit)
            
        if y_unit is None:
            coefficient_units = ['' for i in range(self.ddof)]
            
        kwargs['coefficient_names'] = coefficient_names
        kwargs['basis_names']       = basis_names
        kwargs['coefficient_units'] = coefficient_units
        
        super(SineFit1D, self).__init__(x, y, dy=dy, 
                                              model_function=model_function, 
                                              x_name=x_name, x_unit=x_unit, 
                                              y_name=y_name, y_unit=y_unit, 
                                              **kwargs)
        
    def to_sin_cos(self, return_separate=False):
        """
        convert from the linear regression coefficient representation
        to the sin/cos representation
        """

        frequencies = self.frequencies
        phases      = self.phases
        covariance  = self.covariance

        if self.offset:
            frequencies = np.array([0] + list(frequencies))
            phases      = np.array([np.pi/2] + list(phases))

        # initialize an array of sine and cosine amplitudes
        sin_cos = np.zeros(2 * len(frequencies))
        # initialize an array for the jacobian transformation
        J  = np.zeros((2 * len(frequencies), len(self.coefficients)))    

        counter = 0
        for i in range(len(frequencies)):
            if phases[i] is not None:
                coeff = self.coefficients[counter]
                
                sin_cos[2 * i]     = coeff * np.cos(phases[i])
                sin_cos[2 * i + 1] = coeff * np.sin(phases[i])
                J[2*i: 2*i+2, counter: counter+1] = np.array([[np.cos(phases[i])],
                                                              [np.sin(phases[i])]])
                counter += 1
            else:
                sin_cos[2 * i]     = self.coefficients[counter]
                sin_cos[2 * i + 1] = self.coefficients[counter + 1]

                J[2*i: 2*i+2, counter: counter+2] = np.eye(2)
                counter += 2

        sin_cos_covariance = np.dot(np.dot(J, covariance), J.T)

        if return_separate:

            dsin_cos = np.sqrt(np.diag(sin_cos_covariance))
            extract = lambda array, offset: np.array([array[2 * i + offset] for i in range(len(array)//2)])

            sin  = extract(sin_cos, 0)
            dsin = extract(dsin_cos, 0)
            cos  = extract(sin_cos, 1)
            dcos = extract(dsin_cos, 1)

            return frequencies, sin, dsin, cos, dcos
        else:
            return frequencies, sin_cos, sin_cos_covariance

    def to_amp_phi(self, return_separate=False):
        """
        convert from the linear regression coefficient representation to 
        the amplitude and phase representation
        """
        # first convert to the sin_cos representation
        frequencies, sin_cos, covariance = self.to_sin_cos()

        # initialize an array of amplitudes and phases
        amp_phi = np.zeros(len(sin_cos))
        J = np.zeros((len(sin_cos), len(sin_cos)))

        for i in range(len(sin_cos)//2):
            sin = sin_cos[2 * i]
            cos = sin_cos[2 * i + 1]
            amp_phi[2*i : 2*i+2] = np.array([np.sqrt(sin**2 + cos**2), 
                                             (np.arctan2(sin,cos) % 2 * np.pi) - np.pi])
            J[2*i: 2*i+2, 2*i: 2*i+2] = np.array([[sin/np.sqrt(sin**2 + cos**2), cos/np.sqrt(sin**2 + cos**2)],
                                                  [cos/(sin**2 + cos**2),       -sin/(sin**2 + cos**2)       ]])

        amp_phi_covariance = np.dot(np.dot(J, covariance), J.T)

        if return_separate:

            damp_phi = np.sqrt(np.diag(amp_phi_covariance))
            extract = lambda array, offset: np.array([array[2 * i + offset] for i in range(len(array)//2)])

            amp  = extract(amp_phi, 0)
            damp = extract(damp_phi, 0)
            phi  = extract(amp_phi, 1)
            dphi = extract(damp_phi, 1)

            return frequencies, amp, damp, phi, dphi
        else:
            return frequencies, amp_phi, amp_phi_covariance
        
class FourierTransform(SineFit1D):
    
    model_type = 'fourier transform'
    
    def __init__(self, x, y, dy=None, n_frequencies=None, f_name='f', f_unit=None, **kwargs):
        
        self.f_name = str(f_name)
        self.f_unit = str(f_unit)
        
        damage_control(y, dy)
        x = np.array(x)
        
        sample_time = (max(x) - min(x)) / float(len(x) - 1)
        frequencies = np.fft.rfftfreq(len(x), sample_time)
        
        self.even_parity = (len(x) % 2 == 0)
        self._phi =  2 * np.pi * frequencies[-1] * min(x)
        
        if self.even_parity:
            phases = ([np.pi/2] + [None] * (len(frequencies) - 2) 
                      + [np.pi/2 + self._phi])
        else:
            phases = [np.pi/2] + [None] * (len(frequencies) - 1)
        
        super(FourierTransform, self).__init__(x, y, dy=dy, frequencies=frequencies, 
                                               phases=phases, offset=False, **kwargs)

        
    def get_fourier_transform(self, abs_phi=False):
        
        if abs_phi:
            frequencies, abs, dabs, phi, dphi = self.to_amp_phi(return_separate=True)
            
            abs[0]  = 2 * abs[0]
            dabs[0] = 2 * dabs[0]
         
            phi = np.pi - phi
            
            if self.even_parity:
                abs[-1]  = 2 * abs[-1]
                dabs[-1] = 2 * dabs[-1]
                
            return frequencies, abs, dabs, phi, dphi
            
        else:
            frequencies, sin, dsin, cos, dcos = self.to_sin_cos(return_separate=True)
        
            real  = cos
            dreal = dcos
            imag  = -sin
            dimag = dsin

            real[0] = 2 * real[0]
            dreal[0]= 2 * dreal[0]
            imag[0] = 2 * imag[0]
            dimag[0]= 2 * dimag[0]

            if self.even_parity:
                real[-1]  = 2 * real[-1]
                dreal[-1] = 2 * dreal[-1]
                imag[-1]  = 2 * imag[-1]
                dimag[-1] = 2 * dimag[-1]

            return frequencies, real, dreal, imag, dimag
            
    
    def calculate_rfft(self):
        """
        this performs np.rfft, and then applies a transformation 
        from my fourier expansion definition
        to the rfft fourier expansion definition
        """
        fft = np.fft.rfft(self.data.y)/(len(self.frequencies) - .5 - .5 * self.even_parity)
        fft = fft * np.exp(-2 * np.pi * 1j * self.frequencies * np.min(self.data.x))
        real = np.real(fft)
        imag = np.imag(fft)
        return real, imag
        
    def plot_fourier_transform(self, enlargement_factor = 25., show_fft=False, 
                               colors=None, separation=1/10.0, show_negative=False,
                               amplitude=False, phase=False):
        
        if colors is None:
            colors = sns.color_palette()
        
        if amplitude:
            frequencies, x, dx, y, dy = self.get_fourier_transform(abs_phi=True)
            label_x = 'Abs[LRFT]'
            label_y = 'Arg[LRFT]'
        else:
            frequencies, x, dx, y, dy = self.get_fourier_transform()
            label_x = 'Re[LRFT]'
            label_y = 'Im[LRFT]'
        
        df = frequencies[1] - frequencies[0]
        
        if show_negative:
            symmetrize = lambda a: np.array(list(np.flipud(a[1:])) + list(a)) 
            frequencies = np.array(list(-np.flipud(frequencies[1:])) 
                                   + list(frequencies))
            x = symmetrize(x)
            dx= symmetrize(dx)
            y = symmetrize(y)
            dy= symmetrize(dy)
        
        x_range = max(frequencies) - min(frequencies)
        xlim = [min(frequencies) - x_range/enlargement_factor, 
                max(frequencies) + x_range/enlargement_factor]
        
        plt.plot(xlim,[0, 0], '--k', alpha=.4)
       
        ax1 = plt.gca()
        if amplitude and phase:
            for tl in ax1.get_yticklabels():
                tl.set_color(colors[0])
        
        plt.errorbar(frequencies - df * separation, x, dx,
                     fmt='.', label=label_x, color = colors[0])
        
        if amplitude and phase:
            ax2 = ax1.twinx()
            ax2.set_ylabel('phase (rad)', color=colors[1])
            for tl in ax2.get_yticklabels():
                tl.set_color(colors[1])
        else:
            ax2 = plt.gca()
        
        if phase or not(amplitude):
            ax2.errorbar(frequencies + df * separation, y, dy,
                         fmt='.', label=label_y, color = colors[1])
        
        label = self.f_name
        if self.f_unit == 'None':
            if self.x_unit != 'None':
                label = label + '\, (\\frac{1}{\\mathrm{' + self.x_unit + '}})'
        else:
            label = label + '\, (\\mathrm{' + self.f_unit + '})'
        
        label = '$' + label + '$'
        plt.xlabel(label)
        
        plt.ylabel(self.plot_labels('y',style='latex'))
        
        if show_fft:
            real, imag = self.calculate_rfft()
            if show_negative:
                real = symmetrize(real)
                imag = symmetrize(imag)
            plt.plot(frequencies - df * separation, real, '.', label='Re[RFFT]', 
                     markeredgecolor=colors[0], markerfacecolor='w', markeredgewidth=1)
            plt.plot(frequencies + df * separation, imag, '.', label='Im[RFFT]',
                     markeredgecolor=colors[1], markerfacecolor='w', markeredgewidth=1)
        
        plt.xlim(xlim)
        
        
        return
    
    def get_reduced_dimension_model(self, p=.25, **kwargs):
        n = normal_p_to_n_sigma(p=p,N=len(self.dcoefficients))
        condition = np.where(np.abs(self.coefficients/self.dcoefficients) > n)
        if len(condition[0]) == 0:
            condition = np.where(np.abs(self.coefficients/self.dcoefficients) ==
                                  max(np.abs(self.coefficients/self.dcoefficients)))
        frequencies = list(self._frequencies[condition])
        phases      = list(self._phases[condition])
        regr = SineFit1D(self.data.x, self.data.y, self.data.dy,
                         frequencies=frequencies, phases=phases, offset=False,
                         x_name=self.x_name, y_name=self.y_name,
                         x_unit=self.x_unit, y_unit=self.y_unit, **kwargs)
        return regr
    

class KernelEstimator1D(RegressionModel):
    
    model_type = 'kernel estimator'
    
    def __init__(self, x, y, dy=None, 
                 width=None, order=None, 
                 method='quadratic', kernel='normal',
                 x_name='x', x_unit=None, **kwargs):
        
        self.x_unit = str(x_unit)
        self.x_name = str(x_name)
        
        if width is None:
            base = 3
            factor = {'smooth': 0, 'linear': 1, 'quadratic':2, 
                      'polynomial': (order if order is not None else 0),
                      'cubic': 3, 'constant': 0} 
            width = (base + factor[method]) * (max(x) - min(x)) / len(x)
            
        # this is the polynomial order that is used for polynomial local regression
        self.order = order
            
        #save the model method
        self.width = width
        self.kernel = self.get_kernel_function(kernel)
        self.method = self.get_method_function(method)
        
        super(KernelEstimator1D, self).__init__(x, y, dy, **kwargs)
        
        if self.auto_fit:
            self.fit()
            
    def evaluate_model(self, x=None):
        if x is None:
            x = self.data.x.T
             
        y_est  = []
        dy_est = []
         
        i = 0
        for x_i in x:
            y_est_i, dy_est_i = self.method(self.kernel(x_i), 
                                            self.data.x, 
                                            self.data.y, 
                                            self.data.dy, 
                                            i)
            i += 1
            
            y_est.append(y_est_i)
            dy_est.append(dy_est_i)
            
        return np.array(y_est), np.array(dy_est)
    
    def fit(self):
            
        self.y_est, self.dy_est = self.evaluate_model()
        
        self.chi2, self.dchi2 = chi_squared(self.data.y - self.y_est, self.data.dy,
                                            ddof=self.ddof, subtract_mean=False)
        
        self._chi2 = self.chi2 # this is stored so that i can undo a normalization of chi_2 to 1
        self._dchi2 = self.dchi2
        
        self._state = 'fit'
        
        if self._renormalize_error:
            self.renormalize_error()
            
        return
            
    def get_kernel_function(self, kernel):
        """
        returns a kernel function vs x.
        """
        
        u = lambda x: (self.data.x - x) / self.width
        square = lambda x: (u(x) >= -1) * (u(x) <= 1)
        
        if kernel == 'uniform':
            return lambda x: (1 / 2.) * square(x)
        elif kernel == 'triangular':
            return lambda x: (1 - np.abs(u(x))) * square(x)
        elif kernel in ['epanechnikov', 'quadratic']:
            return lambda x: (3 / 4.) * (1 - (u(x))**2) * square(x)
        elif kernel in ['biweight', 'quartic']:
            return lambda x: (15 / 16.) * (1 - (u(x))**2)**2 * square(x)
        elif kernel == 'triweight':
            return lambda x: (35 / 32.) * (1 - (u(x))**2)**3 * square(x)
        elif kernel == 'tricube':
            return lambda x: (70 / 81.) * (1 - (np.abs(u(x)))**3)**3 * square(x)
        elif kernel in ['gaussian', 'normal']:
            return lambda x: (1 / np.sqrt(2 * np.pi)) * np.exp(-(u(x))**2/2.)
        elif kernel == 'cosine':
            return lambda x: (np.pi / 4.) * np.cos((np.pi / 2.) * u(x)) * square(x)
        elif kernel == 'logistic':
            return lambda x: 1 / (np.exp(u(x)) + 2 + np.exp(-u(x)))
        elif kernel == 'silverman':
            return lambda x: ((1 / 2.) * np.exp(-np.abs(u(x))/np.sqrt(2)) *
                              np.sin(np.abs(u) / np.sqrt(2) + np.pi / 4.))
        else:
            IOError('the kernel type that you requested is invalid.')
            
    def get_method_function(self, method):
        
        if method in ['smooth', 'constant']:
            def smooth(k, x, y, dy, i):
                norm   = np.sum(k / dy**2)
                y_est  = np.sum(k * y / dy**2) / norm
                dy_est = np.sqrt(np.sum(k**2 / dy**2)) / norm
                return y_est, dy_est
            
            self.ddof = 1
            return smooth
        
        elif method in ['polynomial', 'linear', 'quadratic', 'cubic']:
            if method == 'linear':
                order = 1
            elif method == 'quadratic':
                order = 2
            elif method == 'cubic':
                order = 3
            elif method == 'polynomial':
                if self.order is None:
                    raise IOError('for polynomial regression, you must enter an order')
                else:
                    order = self.order
            
            def local_linear_regression(k, x, y, dy, i):
                
                x_function  = np.array([np.squeeze(x**j) for j in range(order + 1)])
                weights     = np.squeeze(dy/np.sqrt(k))
                
                coefficients, dcoefficients, covariance = linear_regression(x_function, y, weights)
                
                y_est, dy_est = evaluate_linear_regression_model(x_function,
                                                                 coefficients, 
                                                                 dcoefficients, 
                                                                 covariance)
                
                return y_est[i], dy_est[i]
            
            self.ddof = 1 + order
            return local_linear_regression
        
    def plot_labels(self, which, style='latex'):
        
        if style == 'latex':
            s = lambda i: '\, (\\mathrm{' + str(i) + '})'
        else:
            s = lambda i: ' (' + str(i) + ')'
        
        if which == 'x':
            string = self.x_name
            if self.x_unit != 'None':
                string = string + s(self.x_unit)
        elif which == 'y':
            string = self.y_name
            if self.y_unit != 'None':
                string = string + s(self.y_unit)
        elif which == 'residuals':
            string = '\\mathrm{residuals}'
            if self.y_unit != 'None':
                string = string + s(self.y_unit)
        else:
            raise IOError('')
            
        if style == 'latex':
            return '$' + string + '$'
        else:
            return string
        
    def plot(self, legend=True, chi2=False, **kwargs):
        return regression_plot(self, legend=legend, 
                               fit_expression=False, chi2=chi2, **kwargs)
        
        

                   
def regression_plot(regrs, colors=None, y0_line=True, x0_line=False, 
                    legend=False, labels=None, fit_expression=False, chi2=None,
                    max_label_length=150, enlargement_factor=25., alpha_origin=.4,
                    alpha_data=.75, alpha_fit=.5, alpha_dfit=.25,
                    residuals=False, derivative=None, integral=None, 
                    linear_transformation=None, show_data=True, errorbars=None, 
                    fit_errors=True, outliers=False, axis_style='plain', **kwargs):
    """
    given an input of an array of regression objects, make a plot
    """
    
    
    #we cant display data if we are plotting integrals or derivatives
    if ((derivative is not None) or (integral is not None)
        or (linear_transformation is not None)):
        show_data = False
        
        # get the correct linear transformation if we want to plot
        # derivatives or integrals
        if (derivative is not None):
            if isinstance(derivative, bool):
                derivative = 1
            linear_transformation = self.get_derivative(n=derivative)
        
        if (integral is not None):
            if isinstance(integral, bool):
                integral = 1
            linear_transformation = self.get_integral(n=integral)
    
    # enable a list of regression objects, or a single regression
    # object as input
    if not(isinstance(regrs,(list, tuple))):
        regrs = [regrs]
    
    #get default color palette
    if colors is None:
        colors = sns.color_palette()
        
    # organize the plot labels
    if labels is None:
        labels = []
        count = 1
        for regr in regrs:
            if regr.label is None:
                labels.append(['dataset ' + str(count)])
            else:
                labels.append([regr.label])
            count += 1
    
    # insert chi2 into the plot labels
    
    for i in range(len(regrs)):
        if chi2 is None:
            add_chi2 = not(regrs[i]._renormalize_error)
        else:
            add_chi2 = chi2
        if add_chi2:
            labels[i].append(', $\chi^2 = ' + scientific_string(regrs[i].chi2, 
                                                                regrs[i].dchi2, 
                                                                style='latex') + '$')
            
    # insert the fit expression into the plot labels
    if fit_expression:
        for i in range(len(regrs)):
            latex = regrs[i].model_string_with_coefficients(style=axis_style)
            latex = ', ' + latex
            labels[i] = labels[i] + re.split(r'(\+)',latex)
    
    
    labels = [truncate(label, max_label_length) for label in labels]
        
    #figure out the x and y extents of the plot:
    xlim = []
    ylim = []
    for regr in regrs:
        regr_xlim, regr_ylim = regr.data_limits(residuals=residuals)

        xlim = [min(xlim + regr_xlim), max(xlim + regr_xlim)]
        ylim = [min(ylim + regr_ylim), max(ylim + regr_ylim)]
        
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    xlim = [xlim[0] - x_range/enlargement_factor, xlim[1] + x_range/enlargement_factor]
    ylim = [ylim[0] - y_range/enlargement_factor, ylim[1] + y_range/enlargement_factor]
        

    #plot the origin axes, if requested
    if y0_line:
        plt.plot(xlim,[0, 0],'--k', alpha=alpha_origin)
    if x0_line:
        plt.plot([0, 0], ylim, '--k', alpha=alpha_origin)
    
    # plot the model:
    x_eval = np.linspace(xlim[0], xlim[1], 200)
    
    count = 0
    for regr in regrs:
        
        if regr._state == 'fit':
            y_eval, dy_eval = regr.evaluate_model(x_eval)

            if residuals:
                y = np.zeros(y_eval.shape)
            else:
                y = y_eval
            if fit_errors:
                plt.fill_between(x_eval, y + dy_eval, y - dy_eval, 
                                 alpha=alpha_dfit, color=colors[count%len(colors)])
            plt.plot(x_eval, y, '--', alpha=alpha_fit, color=colors[count%len(colors)])
            count += 1
        
        
    # plot the data
    count = 0
    for regr in regrs:
        if residuals:
            y = regr.residuals()
        else:
            y = regr.data.y
            
        x = np.squeeze(regr.data.x)
        if errorbars is None:
            plot_errorbars = regr.plot_errorbars
        else:
            plot_errorbars = errorbars
            
        if plot_errorbars:
            plt.errorbar(x, y, regr.data.dy, 
                         fmt='.', color=colors[count%len(colors)], label=labels[count], alpha=alpha_data)
        else:
            plt.plot(x, y,
                     '.', color=colors[count%len(colors)], label=labels[count], alpha=alpha_data)
            
        if outliers:
            x = np.squeeze(regr.outliers.x)
            if residuals:
                y_est, dy_est = regr.evaluate_model(x)
                y = regr.outliers.y - y_est
            else:
                y = regr.outliers.y
                
            if regr.plot_errorbars:
                plt.errorbar(x, y, regr.outliers.dy, 
                         fmt='.', label=labels[count], alpha=alpha_data,
                         markeredgecolor=colors[count%len(colors)], markerfacecolor='w', markeredgewidth=1)
        count += 1
        
    # rescale the axes
    plt.axis(xlim + ylim)
    plt.xlabel(regrs[0].plot_labels('x', style=axis_style))
    if residuals:
        plt.ylabel(regrs[0].plot_labels('residuals', style=axis_style))
    else:
        plt.ylabel(regrs[0].plot_labels('y', style=axis_style))
    
    if legend and not fit_expression:
        plt.legend(loc='best')
    elif fit_expression:
        plt.legend(bbox_to_anchor=(0., 1.00, 1., .100), loc=3)
            
    return
        
def truncate(list_of_strings, max_length):
    """
    take a list of strings, combine them so that the max length is less than max_length
    and concatenate them while ensuring that there is an even number of $$ signs.
    """
    count = 0
    index = 0
    for string in list_of_strings:
        count += len(string)
        if count >= max_length:
            list_of_strings = list_of_strings[0:index - 1]
            list_of_strings.append('+ ...')
            break
        index += 1
    
    output = ''.join(list_of_strings)
    if output.count('$') % 2 == 1:
        output = output + '$'
    return output


def plot3D(regrs, colors=None, legend=False, labels=None, 
           fit_expression=False, chi2=True, max_label_length=100, 
           enlargement_factor=25., pane_color=(1.0, 1.0, 1.0, 1.0),
           alpha_data=.75, alpha_fit=.3,  grid_linewidth=.5, 
           v_angle=30, h_angle=45, contour=None, fit_errors=True,
           error_linewidth=.5, show=False, viewing_distance=12., 
           colormap=None, axis_style='plain', p_coefficient=.05, 
           max_line_length=140, **kwargs):
    
    if not(isinstance(regrs,(list, tuple))):
        regrs = [regrs]
        
    if len(regrs) == 1 and colormap is None:
        colormap = cm.coolwarm
        
    if contour is None:
        contour = (len(regrs) == 1)
    
    #get default color palette
    if colors is None:
        colors = sns.color_palette()
        
    # organize the plot labels
    if labels is None:
        labels = []
        count = 1
        for regr in regrs:
            if regr.label is None:
                labels.append(['dataset ' + str(count)])
            else:
                labels.append([regr.label])
            count += 1
            
    if chi2:
        for i in range(len(regrs)):
            labels[i].append(', $\chi^2 = ' + scientific_string(regrs[i].chi2, 
                                                                regrs[i].dchi2, 
                                                                style='latex') + '$')
        
    if fit_expression:
        for i in range(len(regrs)):
            latex = regrs[i].model_string_with_coefficients(style='latex', p=p_coefficient, 
                                                            max_line_length=max_line_length)
            #latex = ', ' + latex
            #labels[i] = labels[i] + re.split(r'(\+)',latex)
            labels[i].append(latex)
    
    labels = [''.join(label) for label in labels]
    #labels = [truncate(label, max_label_length) for label in labels]
        
    #figure out the x, y, and z extents of the plot:
    xlim = []
    ylim = []
    zlim = []
                             
    for regr in regrs:
        regr_xylim, regr_zlim = regr.data_limits()
        regr_xlim = regr_xylim[0]
        regr_ylim = regr_xylim[1]
            
        xlim = [min(xlim + regr_xlim), max(xlim + regr_xlim)]
        ylim = [min(ylim + regr_ylim), max(ylim + regr_ylim)]
        zlim = [min(zlim + regr_zlim), max(zlim + regr_zlim)]
        
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    
    xlim = [xlim[0] - x_range/enlargement_factor, xlim[1] + x_range/enlargement_factor]
    ylim = [ylim[0] - y_range/enlargement_factor, ylim[1] + y_range/enlargement_factor]
    zlim = [zlim[0] - z_range/enlargement_factor, zlim[1] + z_range/enlargement_factor]

    # plot the model:
    N = 75
    x_reduced = np.linspace(xlim[0], xlim[1], N)
    y_reduced = np.linspace(ylim[0], ylim[1], N)

    x_2d = np.repeat(np.expand_dims(x_reduced, 1), N, axis=1).T
    y_2d = np.repeat(np.expand_dims(y_reduced, 1), N, axis=1)

    x_eval = x_2d.flatten()
    y_eval = y_2d.flatten()
    
    fig = plt.gcf()    
    ax = fig.gca(projection='3d')
    
    ax.w_xaxis.set_pane_color(pane_color)
    ax.w_yaxis.set_pane_color(pane_color)
    ax.w_zaxis.set_pane_color(pane_color)

    ax.w_xaxis.gridlines.set_lw(grid_linewidth)
    ax.w_yaxis.gridlines.set_lw(grid_linewidth)
    ax.w_zaxis.gridlines.set_lw(grid_linewidth)
    
    count = 0
    for regr in regrs:
        z_eval, dz_eval = regr.evaluate_model(np.array([x_eval, y_eval]))

        z_2d = np.reshape(z_eval,(N, N))
        dz_2d = np.reshape(dz_eval,(N, N))
                             
        ax.plot_surface(x_2d, y_2d, z_2d,
                        rstride=8, 
                        cstride=8, 
                        alpha=alpha_fit, 
                        cmap=colormap,
                        color=colors[count % len(colors)])
        if contour:      
            con = ax.contour(x_2d, y_2d, z_2d, zdir='z', 
                             offset=zlim[0], cmap=colormap, alpha=alpha_fit, 
                             color=colors[count % len(colors)])
                             
        if len(regrs) == 1:
            data_color = 'k'
        else:
            data_color = colors[count % len(colors)]
                
        ax.scatter(regr.data.x[0], regr.data.x[1], regr.data.y, alpha=alpha_data, 
                   label=labels[count], color=data_color)

        if fit_errors:
            for i in range(len(regr.data)):
                ax.plot([regr.data.x[0][i], regr.data.x[0][i]], 
                         [regr.data.x[1][i], regr.data.x[1][i]],
                         [regr.data.y[i], regr.y_est[i]], '-', 
                         linewidth=error_linewidth, alpha=alpha_data,
                         color=data_color)
                             
        count += 1


    ax.view_init(v_angle, h_angle)
    ax.dist = viewing_distance

    ax.set_xlabel(regrs[0].plot_labels('x', style=axis_style))
    ax.set_ylabel(regrs[0].plot_labels('y', style=axis_style))
    ax.set_zlabel(regrs[0].plot_labels('z', style=axis_style))
    
                             
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
        
    if legend and not fit_expression:
        plt.legend(loc='best')
    elif fit_expression:
        plt.legend(bbox_to_anchor=(.1, .9, .75, .1), loc=3)
    
    if show:
        plt.show()
    
    return fig