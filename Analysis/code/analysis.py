import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')

import os
from datetime import datetime
import sympy

import sys
sys.path.insert(0, '../Statistics/code')

from statfunctions import NonLinearFit1D, regression_plot
from analysis_settings import settings


def find_data_files(directory, run=None, block=None, 
                    recursive=True, return_dataframe=True):
    """
    crawl through the given directory and find all data files
    consisting of binary files and associated header text files

    inputs:
    directory (path) - path to search
    run (int) - if specified, only returns files of that run.
    block (int) - if specified, only returns files of that block.
    """
    if not(os.path.isdir(directory)):
        raise IOError('the input *directory* must be an accessible folder')
        
    # convert block and run inputs into lists
    if block is not None:
        try:
            block[0]
        except:
            block = [block]
    
    if run is not None:
        try:
            run[0]
        except:
            run = [run]
    
    files = os.listdir(directory)
    
    found_data    = []
    
    while len(files) > 0:
        f = files.pop()
        sub_path = os.path.join(directory, f)
        if recursive:
            if os.path.isdir(sub_path):
                new_data = find_data_files(sub_path, run=run, block=block,
                                           return_dataframe=False)
                found_data += new_data
        split = f.split('.')
        file_extension = split[-1]
        if (len(split) > 1 and 
           (split[-2] == 'header') or split[-1] == 'bin'):
            Date, Run, Block, Trace = split_filename(f)
            condition = True
            if run is not None:
                condition &= (Run in run)
            if block is not None:
                condition &= (Block in block)
                
            if condition:
                
                if file_extension == 'bin':
                    possible_filename = '.'.join(split[:-1]) + '.header.txt'
                    sub_path_header = os.path.join(directory, possible_filename)
                    try:
                        ind = files.index(possible_filename)
                        header_file = files.pop(ind)
                        found_data.append((Date, Run, Block, Trace, sub_path, sub_path_header))
                    except:
                        pass
                elif file_extension == 'txt':
                    possible_filename = '.'.join(split[:-2]) + '.bin'
                    sub_path_binary = os.path.join(directory, possible_filename)
                    try:
                        ind = files.index(possible_filename)
                        binary_file = files.pop(ind)
                        found_data.append((Date, Run, Block, Trace, sub_path_binary, sub_path))
                    except:
                        pass
    if return_dataframe:
        return files_to_dataframe(found_data)
    else:
        return found_data

def files_to_dataframe(found_data):
    #date, run, block, trace, binary_path, header_path = zip(*found_data)
    columns = ['Date', 'Run', 'Block', 'Trace', 'BinaryPath', 'HeaderPath']
    return pd.DataFrame(found_data,
                         columns=columns)

def split_filename(filename):
    """
    given binary and header filenames provide
    the date, run, and block
    """
    f = filename.split('.')
    date  = datetime.strptime(f[0], '%Y%m%d')
    run   = int(f[1])
    block = int(f[2])
    trace = int(f[3])
    return date, run, block, trace

def format_from_matlab(expression):
    expression = expression.replace('{','[').replace('}',']')
    expression = eval(expression)
    if isinstance(expression, list):
        if len(expression) == 0:
            expression = None
        elif isinstance(expression[1], (int, float)):
            expression = np.array(expression)
    return expression

def plot_trace(trace, t_ablation, t_subbin, title, subbin_mult=1.0):
    """
    create a plot to show the trace as a function of:
        
    time after ablation
    time within subbin
    PMT index 1/2
    polarization X/Y
    """
    def mult_factor(mult):
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()
        r = lambda lim : (lim[1] - lim[0])
        plt.text(xlim[0] + r(xlim)/20.0, ylim[1] - r(ylim)/20.0,
                 'x' + str(mult),va='top',ha='left')
        return
    
    ax1 = plt.subplot(2,2,1)
    plt.plot(t_ablation, np.mean(trace, axis=(1,2)))
    plt.legend(['PMT 1', 'PMT 2'])
    plt.ylabel('PMT voltage')
    plt.xlim([min(t_ablation), max(t_ablation)])
        
    ax2 = plt.subplot(2,2,2, sharey=ax1)
    plt.plot(t_subbin, subbin_mult * np.mean(trace, axis=(0,1)))
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.legend(['PMT 1', 'PMT 2'])
    if subbin_mult != 1.0:
        mult_factor(subbin_mult)
        
    ax3 = plt.subplot(2,2,3, sharex=ax1)
    plt.plot(t_ablation, np.mean(trace, axis=(2,3)))
    plt.xlabel('time after ablation (ms)')
    plt.ylabel('PMT voltage')
    plt.xlim([min(t_ablation), max(t_ablation)])
    plt.legend(['X polarization', 'Y polarization'])
    plt.setp(ax1.get_xticklabels(), visible=False)
        
    ax4 = plt.subplot(2,2,4, sharex=ax2, sharey=ax3)
    plt.plot(t_subbin, subbin_mult * np.mean(trace, axis=(0,3)).T)
    plt.xlim([min(t_subbin), max(t_subbin)])
    plt.legend(['X polarization', 'Y polarization'])
    plt.xlabel('time within subbin ($\mu$s)')
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    if subbin_mult != 1.0:
        mult_factor(subbin_mult)
        
    plt.suptitle(str(title))
    plt.subplots_adjust(wspace=0, hspace=0)
    return


class Trace(object):
    """
    this is an object used to import a raw experiment trace.
    """
    def __init__(self, path):
        """
        inputs:
        path - dataframe with "Date", Run", "Block", "Trace", "BinaryPath", "HeaderPath" fields
        """
        self.path       = path
        
        # these are defined for easy access to these parameters:
        self.date       = path['Date']
        self.run        = path['Run']
        self.block      = path['Block']
        self.trace_num  = path['Trace']
        self.binarypath = path['BinaryPath']
        self.headerpath = path['HeaderPath']
        
        # perform the data import:
        trace  = Trace.import_trace(path['BinaryPath'])
        header = Trace.import_header(path['HeaderPath'])
        
        # bin the data:
        t_ablation, t_subbin, trace = Trace.binning(trace, header, 
                                                    return_times=True)
        
        # primary data:
        self.trace  = trace  # binned trace
        self.header = header # imported header in dictionary format
        
        # time arrays for plotting
        self.t_ablation = t_ablation # time after ablation in ms
        self.t_subbin   = t_subbin   # time within subbin in us
        
    def __str__(self):
        output = [('date', str(self.date.date())),
                  ('run', self.run),
                  ('block', self.block),
                  ('trace', self.trace_num)]
        output = [str(i[0]) + ': ' + str(i[1]) for i in output]
        output = ', '.join(output)
        output = 'Trace: {' + output + '}'
        return output
    
    def __repr__(self):
        return str(self)

    @staticmethod
    def import_trace(binarypath, dtype=np.float64, n_pmts=2, scaling=None):
        """
        import the 64 bit binary files that encode the PMT
        voltage data for each of the two PMTs
        """
        if scaling is None:
            scaling = settings.voltage_to_count_rate
        trace = np.fromfile(binarypath, dtype)
        trace = - scaling * np.reshape(trace, (len(trace)//n_pmts, n_pmts))
        return trace
    
    @staticmethod
    def import_header(headerpath):
        """
        import the header file in a dictionary format
        """
        with open(headerpath, 'rb') as f:
            lines = f.readlines()

        header = {}
        for line in lines:
            line = line.split(';')[0].split('%')[0]
            line = line.split('=')
            if len(line) == 2:
                key = line[0].split(' ')[0]
                value = format_from_matlab(line[1])
                header[key] = value
        return header
    
    @staticmethod
    def binning(trace, header, return_times=False, time_index_offset=None):
        """
        this method, given an imported trace and header file
        bin the input trace.
        
        this results in a 4D array:
        dim 0: time after ablation
        dim 1: X/Y polarization
        dim 2: time with a subbin
        dim 3: pmt index
        """
        dt = header['dt']
        f_pol_chop = header['f_pol_chop']
        
        if time_index_offset is None:
            time_index_offset = settings.polarization_switching_offset

        trace = trace[int(time_index_offset):,:]
        trace = trace[:int(f_pol_chop * int(trace.shape[0]//f_pol_chop)),:]

        n_per_cycle = int(1/(dt * f_pol_chop * 10**3))
        new_shape = (trace.shape[0]//n_per_cycle,
                     2,
                     n_per_cycle//2,
                     trace.shape[1])
        
        binned = np.reshape(trace,new_shape)
        if return_times:
            t_ablation = 10**3 * (n_per_cycle * dt * np.arange(binned.shape[0]) + header['t0_delay'])
            t_subbin   = 10**6 * dt * np.arange(n_per_cycle//2)
            return t_ablation, t_subbin, binned
        else:
            return binned
        
    def plot(self):
        plot_trace(self.trace, self.t_ablation, self.t_subbin, str(self))
        return
    
    
class State(object):
    """
    Combines experiment traces with the same experiment state with respect
    to certain experiment parameters. 
    
    -Performs averaging amongst these traces
    -Performs state-wise background subtraction
    -Computes asymmetry and signal
    -Performs binning of the asymmetry and signal to estimate errorbars
    """
    def __init__(self, traces, parameters):
        trace  = State.merge_traces(traces)
        self.raw_trace = trace
        self.header = State.merge_headers(traces)
        self.t_ablation = traces[0].t_ablation
        self.t_subbin   = traces[0].t_subbin
        self.parameters = parameters
        self.background, self.trace = State.background_subtraction(trace, self.t_ablation)
        self.asymmetry, self.signal, self.binned_asymmetry, self.binned_signal,\
                                    self.dbinned_asymmetry,self.dbinned_signal,\
                                                                 self.t_binned\
                             = State.bin_asymmetry_and_signal(self.trace, self.t_ablation)
    
    @staticmethod
    def merge_traces(traces):
        return np.mean([trace.trace for trace in traces], axis=0)
    
    @staticmethod
    def merge_headers(traces):
        keys = set(traces[0].header.keys())
        for trace in traces:
            new_keys = set(trace.header.keys())
            keys &= new_keys
        
        combined_header = {}
        for key in keys:
            items = []
            for trace in traces:
                items.append(trace.header[key])
            if isinstance(items[0],(int, float, np.ndarray)):
                mean = np.mean(np.array(items), axis=0)
                std  = np.std( np.array(items),  axis=0)/np.sqrt(len(traces))
                items = (mean, std)
            if isinstance(items[0], str):
                items = items[0]
            combined_header[key] = items
        return combined_header
            
    
    @staticmethod
    def background_subtraction(trace, t_ablation, duration=None,
                               method=None):
        if duration is None:
            duration = settings.background_subtract_time
        if method is None:
            method = settings.background_subtraction_method
            
        dt = (t_ablation[2] - t_ablation[1]) * 10**(-3)
        n_subtract = int(duration/dt)

        background = np.mean(trace[:n_subtract], axis=0, keepdims=True)
        full_background = background
        if not('xy' in method):
            background = np.mean(background, axis=1, keepdims=True)
        if not('time' in method):
            background = np.mean(background, axis=2, keepdims=True)
        if not('pmt' in method):
            background = np.mean(background, axis=3, keepdims=True)
            
        return full_background, (trace - background)
    
    @staticmethod
    def bin_asymmetry_and_signal(avgd_trace, t_ablation, n_bin=None, subbins=None, avg_pmts=None):
        if subbins is None:
            subbins = settings.subbins
        if avg_pmts is None:
            avg_pmts = settings.avg_pmts
        if n_bin is None:
            n_bin = settings.asymm_grouping
        trace = avgd_trace
        if avg_pmts:
            trace = np.mean(trace, axis=3, keepdims=True)
        new_shape = list(trace.shape)
        new_shape[2] = len(subbins) 
        new_trace = np.zeros(new_shape)
        for index, subbin in enumerate(subbins):
            start, end = subbin
            a = np.mean(trace[:,:,start:(end + 1), :], 
                        axis=2)
            new_trace[:,:,index,:] = a
            
        asymmetry = ((new_trace[:,0,:,:] - new_trace[:,1,:,:])/
                     (new_trace[:,0,:,:] + new_trace[:,1,:,:]))
        signal    = (new_trace[:,0,:,:] + new_trace[:,1,:,:])/2.0
        
        n_points = (asymmetry.shape[0]//n_bin)
        
        new_shape = list(asymmetry.shape)
        new_shape[0] = n_bin
        new_shape.insert(0, n_points)
        
        binning  = lambda x: np.reshape(x[:n_points*n_bin,:,:], new_shape)
        mean_bin = lambda x: np.mean(binning(x), axis=1)
        dmean_bin= lambda x: np.std(binning(x), axis=1)/np.sqrt(n_bin - 1)
        
        binned_asymmetry = mean_bin(asymmetry)
        dbinned_asymmetry= dmean_bin(asymmetry)
        binned_signal    = mean_bin(signal)
        dbinned_signal   = dmean_bin(signal)
        t_binned         = np.mean(np.reshape(t_ablation[:n_points*n_bin],(n_points, n_bin)),axis=1)
        
        return asymmetry, signal, binned_asymmetry, binned_signal, dbinned_asymmetry, dbinned_signal, t_binned
    
    def signal_cut(self, threshold=10):
        pass
    
    def asymmetry_cut(self, threshold=1):
        pass
    
    def chi2_cut(self, threshold=.05):
        pass
    
    @staticmethod
    def get_parameters(traces, state_parameters):
        """
        traces is a list of Trace objects to be combined into a list of State objects
        
        state_parameters is a list of parameters that must be consistent accross
        all traces to be combined. items in state_parameters can be either
        strings, in which case they are interpreted as keys to header dictionaries,
        or functions that take header dictionaries as input.
        """
        trace_parameters = []
        for trace in traces:
            trace_params = []
            for param in state_parameters:
                if hasattr(param, '__call__'):
                    trace_params.append(param(trace.header))
                elif isinstance(param, str):
                    trace_params.append(trace.header[param])
                else:
                    raise IOError('parameters must be a function or a string')
            trace_parameters.append(tuple(trace_params))
        unique_parameters = set(trace_parameters)
        
        param_dict = {k:[] for k in unique_parameters}
        for i in param_dict.keys():
            inds = [j for j in range(len(trace_parameters)) 
                    if trace_parameters[j] == i]
            param_dict[i] = inds
        return param_dict
    
    @staticmethod
    def create_states_by_parameters(traces, state_parameters):
        """
        traces is a list of Trace objects to be combined into a list of State objects
        
        state_parameters is a list of parameters that must be consistent accross
        all traces to be combined. items in state_parameters can be either
        strings, in which case they are interpreted as keys to header dictionaries,
        or functions that take header dictionaries as input.
        """
        param_names = []
        for p in state_parameters:
            if hasattr(p, '__call__'):
                param_names.append(p.func_name)
            elif isinstance(p, str):
                param_names.append(p)
            else:
                raise IOError('state_parameters must be functions or strings')
                
        # create a set of tuples corresponding to the state parameters for each
        parameters = State.get_parameters(traces, state_parameters)
        
        states = []
        for param, inds in parameters.items():
            trace_subset = [traces[i] for i in inds]
            param_dict   = dict(zip(param_names, param))
            states.append(State(trace_subset, param_dict))
            
        return states
    
    def plot_trace(self, subbin_mult=4.0):
        plot_trace(self.trace, self.t_ablation, self.t_subbin, 
                   str(self), subbin_mult=subbin_mult)
        return
    
    def plot_background(self):
        background = self.background
        t_subbin = self.t_subbin
        
        b0 = np.mean(background, axis=(0,3)).T
        b1 = np.mean(background, axis=(0,1))
        plt.subplot(1,2,1)
        plt.plot(t_subbin, b0)
        plt.xlabel('t within subbin ($\mu$s)')
        plt.ylabel('PMT voltage')
        plt.legend(['X polarization', 'Y polarization'], loc='best')
        plt.subplot(1,2,2)
        plt.plot(t_subbin, b1)
        plt.xlabel('t within subbin ($\mu$s)')
        plt.ylabel('PMT voltage')
        plt.legend(['PMT 1', 'PMT 2'], loc='best')
        plt.tight_layout()
        return
    
    
class StroboscopicFit1D(NonLinearFit1D):
    
    model_type = '1d stroboscopic fit \n A(erf((x-B)/D) - erf((x-C)/D))'
    
    def __init__(self, x, y, dy=None, start_point=None,
                 **kwargs):
        
        x1, A, B, C, D = sympy.symbols('x A B C D')
        
        variables = (x1, A, B, C, D)
        expression = A * (sympy.erf((x1 - (B - C))/D) - sympy.erf((x1 - (B + C))/D))
      
        if start_point is not None:
            start_point = start_point[:len(variables) - 1]
        else:
            r = np.max(x) - np.min(x)
            start_point = [np.max(y), 
                           np.min(x) + r/4.0,
                           np.min(x) + 3 * r/4.0,
                           r/4.0]
        
        super(StroboscopicFit1D, self).__init__(x, y, dy=dy, start_point=start_point,
                                                variables=variables, 
                                                expression=expression,
                                                **kwargs)

class StroboscopicAnalysis(object):
    """
    an object for containing data and performing
    analysis on stroboscopic data with microwave pulses
    """
    
    def __init__(self, traces, 
                 dt_pump=150E-6, 
                 Dt_pump=400E-6,
                 dt_MW=5E-6,
                 t_start=2E-3,
                 tau_guess=1.1E-3,
                 signal_threshold=.003,
                 length=.23,
                 subbin_ind=0,
                 n_min=5,
                 fit=StroboscopicFit1D):
        """
        inputs: 
        traces: list of Trace objects to be analyzed
        dt_pump: duration of the pump ON-time
        Dt_pump: time between stobe pulses
        dt_MW: duration of the microwave pulse
        t_start: start of the first strobe pulse
        tau_guess: initial guess for the precession time
        """

        e_field_state = lambda header :int(np.sign(header['e_field_east_set']) * header['e_field_lead_set'])
        e_field_state.func_name = 'e_field_state'

        parameters = ['Microwave_freq',
                      e_field_state,
                      'h_state_omega_doublet']

        self.states    = State.create_states_by_parameters(traces, parameters)
        self.avg_state = State(traces, [])

        self.dt_pump = dt_pump
        self.Dt_pump = Dt_pump
        self.dt_MW   = dt_MW
        self.t_start = t_start
        self.tau_guess = tau_guess
        self.length  = length

        t = self.avg_state.t_ablation * 1E-3 - t_start + tau_guess
        tscale = ((t/Dt_pump) - np.floor(t/Dt_pump))
        mask = (t > 0) * (tscale < dt_pump/Dt_pump)
        self.t_centers_start = t_start + dt_pump/2.0 + Dt_pump * np.arange(100)
        self.t_centers_guess = self.t_centers_start + tau_guess
        
        self.t = self.avg_state.t_ablation
        
        # find the locations of the peaks above threshold
        s = np.squeeze(np.mean(self.avg_state.signal[:,subbin_ind,:], axis=1))
        self.ending_indices, self.peak_centers = StroboscopicAnalysis.find_peaks(s, signal_threshold, n_min)
        self.t_centers = np.array([t[int(i)] for i in self.peak_centers])
        
        # fit the data assuming normal velocity distribution
        self.fits, self.fit_parameters = StroboscopicAnalysis.perform_fits(self.t, self.peak_centers, s)
        
        # match the found centers of the pulses to the start of the pulses:
        self.match_pulses()
        self.evaluate_tau()
        
        
    @staticmethod
    def find_peaks(signal, signal_threshold, n_min):
        """
        given a signal trace, find the starting and ending points
        for peaks that have at least n_min number of points consecutively
        above the threshold signal_threshold.
        
        returns a list of tuples corresponding the start and end
        """
        above_threshold = (signal > signal_threshold)
        inds_above_threshold = np.array([i for i in range(len(signal))
                                           if above_threshold[i]])
        meet_n_threshold = ((inds_above_threshold[1:] - inds_above_threshold[:-1]) > n_min)
        inds_meet_n_threshold = np.array([i for i in range(len(meet_n_threshold) - 1)
                                         if meet_n_threshold[i]])
        ending_indices =  np.array(
                           [[inds_above_threshold[0]] 
                           + [inds_above_threshold[i + 1] for i in inds_meet_n_threshold],
                             [inds_above_threshold[i] for i in inds_meet_n_threshold] +
                           [inds_above_threshold[-1]]])
                           
        peak_centers = np.floor(np.mean(ending_indices, axis=0))
        return ending_indices, peak_centers
    
    @staticmethod
    def perform_fits(t, centers, signal, fit=StroboscopicFit1D):
        width = np.min((centers[1:] - centers[:-1])//2)
        fits = []
        fit_parameters = {'center':[], 'dcenter':[], 
                          'd':[], 'dd':[], 'amplitude':[]}
        for center in centers:
            t_i = t[center - width: center + width + 1]
            s_i = signal[center - width: center + width + 1]
            A0 = max(s_i)/4.0
            D0 = (max(t_i) - min(t_i))/8.0
            B0 = t[center] # min(t_i) + (3/2.0) * D0
            C0 = (max(t_i) - min(t_i))/2.0 # max(t_i) - (3/2.0) * D0
            
            start_point = (A0, B0, C0, D0)
            
            regr = fit(t_i, s_i, start_point=start_point, max_iter=1000000,
                       x_name='t after ablation', x_unit='ms',
                       y_name='PMT voltage', y_unit='V')

            fits.append(regr)
            fit_parameters['center'].append(regr.coefficients[1])
            fit_parameters['dcenter'].append(regr.dcoefficients[1])
            fit_parameters['d'].append(abs(regr.coefficients[2]))
            fit_parameters['dd'].append(regr.dcoefficients[2])
            fit_parameters['amplitude'].append(np.sum(s_i))
        
        fit_parameters = {k:np.array(v) for k, v in fit_parameters.items()}
        return fits, fit_parameters
    
    def match_pulses(self):
        difference = [0] + list(np.cumsum(
                  np.round((
                  self.fit_parameters['center'][1:] -
                  self.fit_parameters['center'][:-1])/(self.Dt_pump * 1E3)),
                  dtype=int)
            )
        difference = np.array(difference)
        return difference
        
    
    def plot_fit(self, fit_errors=False, **kwargs):
        regression_plot(self.fits, colors=sns.color_palette()[0:2], 
                        fit_errors=fit_errors, **kwargs)
        #plt.xlabel('t after ablation (ms)')
        #plt.ylabel('pmt voltage')
        _=plt.title('fits to stroboscopic data assuming normal velocity distributions')
        return
    
    def evaluate_tau(self):
        inds = self.match_pulses() + 1
        centers = self.fit_parameters['center'] * 1E-3
        deviation = [np.sum((self.t_centers_guess[inds + i] - centers)**2)
                     for i in range(len(self.t_centers_guess) - len(inds))]
        optimum_ind = np.argmin(deviation)
        self.fit_parameters['tau'] = (centers - 
                                      self.t_centers_start[optimum_ind + inds]) * 1E3
        self.fit_parameters['dtau'] = self.fit_parameters['dcenter']
        self.fit_parameters['vbar'] = 1E3 * self.length /(self.fit_parameters['tau'])
        self.fit_parameters['dvbar'] = (self.fit_parameters['vbar'] * 
                                        self.fit_parameters['dtau']/self.fit_parameters['tau'])
        self.fit_parameters['sigmav'] = (1E-3 * self.fit_parameters['d'] * 
                                        self.fit_parameters['vbar']**2/(np.sqrt(2) * self.length))
        return self.fit_parameters['tau']
    
    def plot_velocity_distributions(self, extent=4.0, colors=None, p=.05):
        if colors is None:
            colors = sns.color_palette()[0:2]
        v  = self.fit_parameters['vbar']
        dv = self.fit_parameters['sigmav']
        amplitude = self.fit_parameters['amplitude']
        vs = np.linspace(min(v - extent * dv),
                          max(v + extent * dv), 300)
        distributions = np.zeros(len(vs))
        index = 0
        for v_i, dv_i, a_i in zip(v, dv, amplitude):
            if (self.fits[index].get_coefficients()['p(zero)'][0] < p):
                dist = a_i * np.exp(-(vs - v_i)**2/(2 * dv_i**2))
                plt.plot(vs, dist, color=colors[index % len(colors)])
                distributions += dist
            index += 1

        plt.plot(vs, distributions, '-k')
        return
    
    def get_velocities(self, p=.05):
        v  = self.fit_parameters['vbar']
        dv = self.fit_parameters['sigmav']
        a  = self.fit_parameters['amplitude']
        t  = self.fit_parameters['center']
        
        use = np.array(
              [i for i in range(len(v))
              if (self.fits[i].get_coefficients()['p(zero)'][0] < p)]
              )
        return t[use], a[use], v[use], dv[use]