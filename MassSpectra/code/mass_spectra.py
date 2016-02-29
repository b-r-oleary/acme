import sys
sys.path.append('../Statistics/code')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')
import xmltodict
import os
from os import listdir
from os.path import isfile, join
import re
import pickle
import numpy as np
import statfunctions
import pandas as pd


class MassSpectrumLibrary(object):
    """
    this is a class which imports a directory of .jdx mass spectrum files
    and inserts the data within those files into a dictionary.
    
    these .jdx files are a custom physics data format, and I have added
    and additional field to mine database which includes a ##GROUP field
    which groups a certain compound into certain categories (AIR, HYDROCARBONS,
    NOBLE GASES, ...).
    
    methods:
    X: provides the mass data for the peaks in a given spectrum
    Y: provides the intensity data for the peaks in a given spectrum
    normalized_models: generates normalized mass spectra across a designated mass
                       range for all of the compounds in the library.
    plot: plots the mass spectrum for a certain compound or set of compounds.
    """
    
    def __init__(self, path=None, library_path=None, create_library=False):
        """
        inputs:
        path: this is a path to the directory of .jdx files
        library_path: this the path in which the library has been saved, or if it doesn't
                      exist, where it will be saved.
        create_library: by default, load an existing library if library_path is input and the
                      file exists. Otherwise, or if this is True, reload all of the .jdx files.
        """
        
        if path is None and library_path is None:
            raise IOError('must insert a path or a library path')
        
        if library_path is not None and not(create_library):
            self.library_path = library_path
            if isfile(library_path):
                self.load_library(library_path)
            else:
                create_library = True
        else:
            create_library =True
            
        if path is not None and create_library:
            if path[-1] != '/':
                path = path + '/'
            self.path = path
            if library_path is None:
                library_path = path + 'library.pkl'
            self.library_path = library_path
            
            self.create_library()
            
        self.create_groups()
        
    def __str__(self):
        return 'mass spec peak library: ' + str({k: v.keys() for k, v in self.groups.items()})
        
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.library)
            
    def create_groups(self):
        self.groups = {}
        for k, v in self.library.items():
            if 'GROUP' in v.keys():
                group = v['GROUP']
                if group not in self.groups.keys():
                    self.groups[group] = {}
                self.groups[group][k] = v
        return
            
    def create_library(self):
        file_extensions = ['jdx']
        
        files = [f for f in listdir(self.path) 
                 if (f.split('.')[-1] in file_extensions)]
        
        self.library = {}
        
        for f in files:
            output = self.parse_jdx(self.path + f)
            output['PATH'] = self.path + f
            self.library[output['TITLE']] = output
            
        self.save_library()
        return
    
    def search_for_peaks(self, masses):
        try:
            masses[0]
        except:
            masses = [masses]
            
        matches = []
        values  = []
        for k in self.library.keys():
            total = 0
            passing = True
            for m in masses:
                if m not in self.X(k):
                    passing = False
                else:
                    total += self.Y(k)[self.X(k).index(m)]
            if passing:
                matches.append(k)
                values.append(total/float(sum(self.Y(k))))
            
        inds = np.flipud(np.argsort(values))
        matches = [matches[ind] for ind in inds]
        values  = [values[ind] for ind in inds]
        
        return pd.DataFrame({'compound':matches,'match metric':values})
    
    def normalized_models(self, low_mass=None, high_mass=None, masses=None):
        if masses is None:
            if low_mass is None or high_mass is None:
                raise IOError('you must specify a mass range for the model')
            else:
                masses = np.array(list(range(int(low_mass), int(high_mass) + 1)))
                
        models = {}
        for k in self.library.keys():

            X = self.X(k)
            Y = self.Y(k)
            
            model = np.zeros(len(masses))
            for x, y in zip(X, Y):
                model[np.where(masses == x)] = y
            s = np.sum(model)
            if s != 0:
                model = model / s
                models[k] = model
        return models
            
    def save_library(self):
        pickle.dump(self.library, open(self.library_path, 'wb'))
        return
    
    def load_library(self):
        self.library = pickle.load(open(self.library_path, 'rb'))
        return
          
    def parse_jdx(self, path):
        
        methods = [self._extract_list_of_floats_jdx, 
                   self._extract_floats_jdx,
                   self._extract_string_jdx]
        
        with open(path) as jdx:
            lines = jdx.read().split('##') 
            output = {}
            for line in lines:
                split = line.split('=')
                if len(split) == 2:
                    for method in methods:
                        try:
                            output[split[0]] = method(split[1])
                            break
                        except:
                            pass
        return output
                
    def _extract_list_of_floats_jdx(self, string):
        if len(re.findall('^\(XY..XY\)',string)) != 1:
            raise IOError('this is not appropriately formatted.')
        XY_pairs = re.findall(r'([\d\.]*),([\d\.]*)',string)
        X = [float(xy[0]) for xy in XY_pairs]
        Y = [float(xy[1]) for xy in XY_pairs]
        return X, Y
    
    def _extract_floats_jdx(self, string):
        if len(re.findall(r'([a-z])',string)) > 0:
            raise RuntimeError('this is probably a string not a float')
        return float(re.findall(r'[\d\.]*',string)[0])
    
    def _extract_string_jdx(self, string):
        return [string for string in re.findall(r'(.*)',string) if string != ''][0]
    
    def X(self, key):
        return self.library[key]['PEAK TABLE'][0]
    
    def Y(self, key):
        return self.library[key]['PEAK TABLE'][1]
    
    def plot(self, values=None, yscale='log', colors=None):
        if colors is None:
            colors = sns.color_palette()
        
        if values is None:
            values = self.library.keys()
        if isinstance(values, str):
            values = [values]

        counter = 0
        for value in values:
            X = self.X(value)
            Y = self.Y(value)

            plt.bar(X, Y, color=colors[counter%len(colors)], 
                    label=value, alpha=.3, align='center')
            counter += 1

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('signal (arb)')
        plt.xlabel('mass to charge ratio')
        plt.yscale(yscale)
        return
    
    
class MassSpectrum(object):
    """
    This object houses a mass spectrum and enables analysis of the mass spectrum.
    
    The ExTorr RGA that we have produces a certain .xml file format, and this class,
    dependent on the xmltodict module, imports from that format. This class could be
    extended to various source formats if needed.
    
    after importing, this class does the following analysis:
        - subtract a background from the high mass end of the spectrum
        - extract the RGA lineshape averaged over the whole spectrum
        - fit each mass unit with respect to the RGA lineshape to obtain the
          amplitude at each mass unit.
        - detects statistically significant peaks above background
        - if a MassSpectrumLibrary is input, then it will fit these amplitudes
          using linear least squares to that library. This fitting routine
          renormalizes the error in the amplitude data by default and then trims
          the model in iterations to only keep statistically significant contributions to
          the fit above some statistical significance threshold.
          
    methods:
    plot: plots the mass spectrum with the lineshape fit, and with the detected peaks, 
          and fit amplitudes
    plot_lineshape: plots the empirically extracted mass spectrum lineshape
    plot_fit: plots the fit of the mass spectrum to the library with the detected peak amplitudes
    print_report: returns the plots obtainef from `plot` and `plot_fit` in two subplots in the same
                  figure, and then saves the result to a pdf.
    """
    
    def __init__(self, path, background_region=4, 
                 p_detection=.01, title=None, discard_beginning=2,
                 auto_fit=True, library=None, rga=None, p_fit_cutoff=.1,
                 renormalize_error=True):
        """
        perform background subtraction over the last *background_region* amus
        """
        # create an automated title is one is not provided.
        if title is None:
            title = os.path.split(path)[-1]
            title = 'RGA Mass Spectrum - ' + title.split('.')[0]
            
        self.title = title
        self.path = path
        self.background_region = background_region
        self.discard_beginning = discard_beginning
        
        # determine rga from file extension if not provided
        if rga is None:
            mapping = {'xml':'extorr', 'txt':'srs'}
            file_extension = path.split('.')[-1]
            if file_extension in mapping.keys():
                rga = mapping[file_extension]
            else:
                raise IOError('unrecognized file format - please specify the RGA type')
           
        #import the data
        if rga == 'extorr':
            self._import_from_extorr_format()
        elif rga == 'srs':
            self._import_from_srs_format()
        else:
            raise IOError('RGA format is not recognized')
        
        self.rga = rga
        
        M = np.reshape(self.mass,(self.high_mass - self.low_mass + 1, self.samples_per_amu))
        P = np.reshape(self.pressure,(self.high_mass - self.low_mass + 1, self.samples_per_amu))
        
        self.lineshape = np.mean(P, axis=0)
        self.lineshape = self.lineshape - min(self.lineshape)

        self.lineshape = self.lineshape / max(self.lineshape)
        
        self.lineshape_mass = np.array(list(range(self.samples_per_amu))) / float(self.samples_per_amu)
        self.lineshape_mass = self.lineshape_mass - np.mean(self.lineshape_mass)
        
        self.discrete_mass = np.mean(M, axis=1)
        self.discrete_pressure = np.mean(M, axis=1)
        
        self.threshold = statfunctions.normal_p_to_n_sigma(p_detection,N=len(self.discrete_mass))
        
        self.fit_mass = self.discrete_mass
        self.fit_pressure = np.array([sum(self.lineshape * p)/sum(self.lineshape**2) for p in P])
        
        self.fit_to_lineshape = np.reshape(np.array([self.fit_pressure[i] * self.lineshape 
                                              for i in range(len(self.fit_pressure))]),
                                                          (len(self.pressure),1))
        
        residuals = np.array([np.mean((y - f * self.lineshape)**2) for y, f in zip(P, self.fit_pressure)])
        self.dfit_pressure = np.sqrt(residuals / np.sum(self.lineshape**2))
        
        peak_locations = np.where(self.fit_pressure/self.dfit_pressure > self.threshold)
        
        self.peaks_pressure = self.fit_pressure[peak_locations]
        self.peaks_dpressure = self.dfit_pressure[peak_locations]
        self.peaks_mass = self.fit_mass[peak_locations]
        
        self.models = None
        self.fit = None
        if auto_fit and (library is not None):
            self.fit_to_library(library, p_cutoff=p_fit_cutoff, 
                                renormalize_error=renormalize_error)
        
    def _import_from_extorr_format(self):
        with open(self.path) as xml:
            doc = xmltodict.parse(xml.read())
        
        doc = doc['Data']
        
        #self.pirani_pressure = float(doc['@PiraniPressure'])
        #self.total_pressure  = float(doc['@TotalPressure'])
        self.units           = doc['@Units']
        
        self.samples_per_amu = int(doc['@SamplesPerAMU'])
        self.high_mass       = float(doc['@HighMass'])
        self.low_mass        = float(doc['@LowMass'])
        
        self.pressure        = np.array([float(v.values()[0]) for v in doc['Sample']])
        
        self.background      = np.mean(self.pressure[-5 * self.samples_per_amu: -1])
        self.dbackground     = np.std(self.pressure[-5 * self.samples_per_amu: -1])
        
        self.pressure        = self.pressure - self.background
        
        self.mass            = np.array(range(len(self.pressure))) / float(self.samples_per_amu)
        self.mass            = self.mass - np.mean(self.mass[0:self.samples_per_amu]) +self.low_mass
        
        self.pressure = self.pressure[self.discard_beginning * self.samples_per_amu:]
        self.mass     = self.mass[self.discard_beginning * self.samples_per_amu:]
        self.low_mass = self.low_mass + self.discard_beginning
        return
    
    def _import_from_srs_format(self):

        with open(self.path, 'rb') as f:
            data = f.readlines()

        def find_and_extract(data, prefix):
            for line in data:
                result = re.findall(r'^' + prefix + ',\s(.+?)[\s\r\n|,]', 
                                   line)
                if len(result) > 0:
                    return result[0]
            return None

        self.samples_per_amu = int(find_and_extract(data, 'Points Per AMU'))
        self.units           = find_and_extract(data, 'Units')

        def get_pressure_data(data):
            counter = 0
            for line in data:
                counter += 1
                if line == '\r\n':
                    if data[counter] == '\r\n':
                        break
            counter += 1

            m = []
            p = []
            for i in range(counter, len(data)):
                line = data[i].split(',')
                m.append(float(line[0]))
                p.append(float(line[1]))

            return np.array(m), np.array(p)

        self.mass, self.pressure = get_pressure_data(data)

        inds = np.where(np.logical_and(
                        self.mass > int(self.discard_beginning) + .5,
                        self.mass <= np.floor(max(self.mass) - .5) + .5))

        self.pressure = self.pressure[inds]
        self.mass     = self.mass[inds]

        self.background      = np.mean(self.pressure[-self.background_region * self.samples_per_amu:])
        self.dbackground     = (np.std(self.pressure[-self.background_region * self.samples_per_amu:])
                                                        /np.sqrt(self.samples_per_amu * self.background_region))

        self.pressure        = self.pressure - self.background

        self.high_mass  = np.floor(max(self.mass))
        self.low_mass = np.ceil(min(self.mass))
        return
        
    def plot(self, scan=True, fit=True, bar=True, peaks=True, 
             yscale='log', color=None, label=True, full_xlim=False, legend=True):
        if label is None:
            label = [None] * 4
            legend =False
        elif label == True:
            label = ['amplitude', 'lineshape fit', 'scan', 'detected_peaks']
        elif isinstance(label, str):
            label = [label] * 4
        if bar:
            plt.bar(self.fit_mass, self.fit_pressure,alpha=.2,align='center', label=label[0])
        if fit:
            plt.plot(self.mass, self.fit_to_lineshape, label=label[1],alpha=.75)
        if scan:
            plt.plot(self.mass, self.pressure, label=label[2], alpha=.75)
        if peaks:
            plt.errorbar(self.peaks_mass, self.peaks_pressure, self.peaks_dpressure, label=label[3],fmt='.', alpha=1)
            
        plt.yscale(yscale)
        plt.xlabel('mass/charge (amu)')
        plt.ylabel('pressure (' + self.units + ')')
        
        ax = plt.gca()
        ylim = list(ax.get_ylim())
        ylim[0] = self.dbackground/4.0
        plt.ylim(ylim)
        
        if not(full_xlim):
            xlim = [min(self.peaks_mass), max(self.peaks_mass)]
            x_range = xlim[1] - xlim[0]
            xlim = [max([.5, xlim[0] - x_range/10.0]), 
                    min([xlim[1] + x_range/10.0, max(self.mass)])]
            plt.xlim(xlim)
        
        plt.title(self.title)
        if legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return
            
    
    def plot_lineshape(self):
        
        plt.plot(self.lineshape_mass, self.lineshape)
        plt.xlabel('mass/charge (amu)')
        plt.ylabel('signal (arb)')
        plt.xlim([min(self.lineshape_mass), max(self.lineshape_mass)])
        
    def print_report(self, full_xlim=False, legend=True, **kwargs):
        
        if self.fit is not None:
            plt.subplot(2,1,1)
            self.plot(full_xlim=full_xlim, legend=legend, **kwargs)
            plt.subplot(2,1,2)
            self.plot_fit(full_xlim=full_xlim, legend=legend)
            plt.tight_layout()
        else:
            self.plot(**kwargs)
        
        path = os.path.split(self.path)[0]
        if path[-1] not in ['\\','/']:
            path = path + '/'
        path = path + self.title + '.pdf'
        plt.savefig(path)
        return
    
    def fit_to_library(self, library, plot=False, p_cutoff=.1, renormalize_error=True):
        models = library.normalized_models(self.low_mass, self.high_mass)

        y = self.fit_pressure
        dy= self.dfit_pressure

        while True:
            regr = statfunctions.LinearRegression(models.values(), y, dy, 
                                                  renormalize_error=renormalize_error, 
                                                  coefficient_names=models.keys(),
                                                  coefficient_units=[self.units]*len(models))
            coefficients = regr.get_coefficients()
            statistically_significant = list((coefficients['p(zero)'] <= p_cutoff)
                                                     & (coefficients['value'] > 0))

            if sum(statistically_significant) == len(statistically_significant):
                break
            elif sum(statistically_significant) == 0:
                print models
                raise RuntimeError('none of the models fit sufficiently well to merit a fit.')
            else:
                models = {models.keys()[i]: models[models.keys()[i]] 
                          for i in range(len(models.keys())) if 
                                          statistically_significant[i]}
        
        self.fit = regr
        self.models = models
        if plot:
            self.plot_fit()
        
        return
        
    def plot_fit(self, full_xlim=False, legend=True):
        if self.fit is None:
            raise RuntimeError('you must first fit to a model before plotting')
            
        coefficients = self.fit.get_coefficients()
        coefficients = coefficients.sort('value')

        data  = []
        names = []
        for i in range(len(coefficients)):
            index = coefficients.index[i]
            a = coefficients['value'][index]
            data.append(self.models.values()[index] * a)
            names.append(self.models.keys()[index])

        colors = sns.color_palette("coolwarm", len(data))

        baseline = self.dbackground / 2.0

        y_offset = np.zeros(len(data[0]))
        counter = 0
        for row, label in zip(data, names):
            plt.bar(self.fit_mass, row, bottom=y_offset, 
                    color=colors[counter%len(colors)], 
                    label=label, alpha=.3, align='center')

            y_offset += row
            counter += 1
        plt.errorbar(self.peaks_mass, self.peaks_pressure, self.peaks_dpressure,
                     fmt='.k', label='data')
        
        if legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.yscale('log')

        plt.xlabel('mass/charge (amu)')
        plt.ylabel('pressure (torr)')

        ax = plt.gca()
        ylim = list(ax.get_ylim())
        ylim[0] = baseline
        _ = plt.ylim(ylim)
        
        if not(full_xlim):
            xlim = [min(self.peaks_mass), max(self.peaks_mass)]
            x_range = xlim[1] - xlim[0]
            xlim = [max([.5, xlim[0] - x_range/10.0]), min([xlim[1] + x_range/10.0, max(self.mass)])]

            plt.xlim(xlim)
        
        return
    
    def peak_assignments(self):
        error  = []
        first_contributor  = []
        second_contributor = []
        for m in self.peaks_mass:
            index = list(self.fit_mass).index(m)
            error.append(100 * abs(
                    (self.fit.y_est[index] - self.fit.data.y[index]) 
                                        / self.fit.data.y[index]))

            values = self.fit.data.x[:, index] * self.fit.coefficients
            inds   = np.flipud(np.argsort(values))
            inds   = [ind for ind in inds if values[ind] > 0]
            if len(inds) == 0:
                first_contributor.append(None)
            else:
                first_contributor.append(self.fit.coefficient_names[inds[0]])
            if len(inds) <= 1:
                second_contributor.append(None)
            else:
                second_contributor.append(self.fit.coefficient_names[inds[1]])

        table = pd.DataFrame({'mass (amu)': self.peaks_mass,
                              'pressure (torr)': self.peaks_pressure,
                              'model error (%)': error,
                              'primary component': first_contributor,
                              'secondary component': second_contributor})
        return table