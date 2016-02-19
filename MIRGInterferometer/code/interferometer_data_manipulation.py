import sys
sys.path.append('C:/Users/Brendon/Documents/PythonScripts/Jupyter/Research/Statistics/code')
from statfunctions import *

import re
from datetime import datetime
import numpy as np
import time

class GaussianFit(object):
    """
    this is a simple class that is just used as a 
    convenient holder for gaussian fit parameters
    """
    
    def __init__(self, amplitude=None, width=None, center=None, offset=None,
                    damplitude=None, dwidth=None, dcenter=None, doffset=None):
    
        self.amplitude  = amplitude
        self.damplitude = damplitude
        self.width      = width
        self.dwidth     = dwidth
        self.center     = center
        self.dcenter    = dcenter
        self.offset     = offset
        self.doffset    = doffset
        
    def __str__(self):
        return  str({'amplitude':  self.amplitude,
                    'damplitude': self.damplitude,
                    'width':      self.width,
                    'dwidth':     self.dwidth,
                    'center':     self.center,
                    'dcenter':    self.dcenter})
    
    def __repr__(self):
        return str(self)

class InterferometerData(object):
    """
    this is a minimal class used to import, fit, and
    hold the MIRG interferometer scan data
    """
    def __init__(self, date=None, 
                 x_position=None, y_position=None,
                 arm_positions=None, fringe_contrasts=None,
                 labview_gaussian_fit=None):
        """
        inputs:
        date: (datetime) time when scan was taken
        x_position: (float in cm) x position of velmex translation stage
        y_position: (float in cm) y position of velmex translation stage
        arm_positions: (numpy array of floats in um) position of zaber interferometer translation stage
        fringe_contrasts: (numpy array of floats in arb units) measured fringe contrast from interferometer
        labview_gaussian_fit: (GaussianFit object) contains fit parameters from labview gaussian fit to the data
        """
        
        self.date = date
        self.x_position = x_position
        self.y_position = y_position
        self.arm_positions = arm_positions
        self.fringe_contrasts = fringe_contrasts
        self.labview_gaussian_fit = labview_gaussian_fit
        self.regr = GaussianFit1D(self.arm_positions, self.fringe_contrasts, 
                                      offset=True, x_name='arm displacement', y_name='fringe contrast',
                                      x_unit='\mu m', y_unit='arb',auto_fit=False,
                                      start_point=[self.labview_gaussian_fit.amplitude,
                                                   self.labview_gaussian_fit.center,
                                                   self.labview_gaussian_fit.width,
                                                   min(self.fringe_contrasts * (self.fringe_contrasts > 0))])
        try:
            self.regr.fit()
            self.gaussian_fit = GaussianFit(amplitude=self.regr.coefficients[0],
                                            damplitude=self.regr.dcoefficients[0],
                                            center=self.regr.coefficients[1],
                                            dcenter=self.regr.dcoefficients[1],
                                            width=self.regr.coefficients[2],
                                            dwidth=self.regr.dcoefficients[2],
                                            offset=self.regr.coefficients[3],
                                            doffset=self.regr.dcoefficients[3])
        except:
            print('fit to scan failed')
            self.gaussian_fit = None
        
    def __str__(self):
        return str({'date': self.date,
                    'x_position (cm)': x_position,
                    'y_position (cm)': y_position,
                    'labview_gaussian_fit': str(self.labview_gaussian_fit)})
    
    def __repr__(self):
        return str(self)
        
    @staticmethod
    def read_from_txt(filenames=None, path=None):
        """
        this is a static method for reading in data from a list of
        input filenames and returning a list of InterferometerScan objects
        with the corresponding data
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        
        # for reading tab delimited float data:
        string_to_array = lambda string: np.array([float(s) for s in string.split('\t')])
        strip_end_tabs  = lambda string: re.findall(r'^(.*?)[\t]*$',string)[0]
        
        scans = []
        for filename in filenames:
            print('importing ' + filename) 
            with open(filename, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if len(re.findall(r'Interferometer Scan', lines[i])) > 0:
                        
                        # import and format the date
                        date       = datetime.strptime(
                                        strip_end_tabs(lines[i + 1]),
                                        '%A, %B %d, %Y\t%I:%M:%S %p'
                                     )
                        
                        x_position = float(lines[i + 3])
                        y_position = float(lines[i + 5])
                        
                        arm_positions    = string_to_array(lines[i + 7])
                        fringe_contrasts = string_to_array(lines[i + 9])
                        
                        gaussian = string_to_array(strip_end_tabs(lines[i + 14]))
                        labview_gaussian_fit = GaussianFit(amplitude = gaussian[0],
                                                          damplitude = gaussian[1],
                                                           center    = gaussian[2],
                                                          dcenter    = gaussian[3],
                                                           width     = gaussian[4],
                                                          dwidth     = gaussian[5])
                        
                        scans.append(InterferometerData(date, x_position, y_position,
                                                        arm_positions, fringe_contrasts,
                                                        labview_gaussian_fit))
        return scans
    
    def plot(self, **kwargs):
        self.regr.plot(**kwargs)
        
        
class InterferometerMap(object):
    def __init__(self, scans, center_x=None, center_y=None, 
                 reverse_x=True, reverse_y=False,
                 null_peak_position=0, setpoint=45000,
                 order=4, max_sum=True, 
                 apply_filters=True, renormalize_error=True,
                 remove_outliers=False, **kwargs):
        """
        create a map of the field plate separation from MIRG
        interferometer scans.
        
        inputs:
        scans: (InterferometerScan or a list thereof)
        center_x: (float cm) position of the origin of the x axis relative to that in the data
        center_y: (float cm) position of the origin of the y axis relative to that in the data
        reverse_x: (Boolean) after shifting x by center_x, reverse the axis
        reverse_y: (Boolean) after shifting y by center_y, reverse the axis,
        null_peak_position: (float um) position of the null interferometer peak
        setpoint: (float um) position
        order: (Integer) order of the 2D polynomial fit to the field plate separation
        max_sum: (Boolean) indicate whether the order should be the maximum sum of
                 exponents, or the maximum exponent
        apply_filters: (Boolean) indicate whether or not to filter non-viable scans
        renormalize_error: (Boolean) renormalize the error in the map fit such that chi^2=1
        remove_outliers: (Boolean) iterate and remove datapoints that would be unlikely to occur
                 given accurate errorbars and gaussian statistics.
        **kwargs: passed to the PolynomialFit2D() regression object.
        """
        
        initial_len_scans = len(scans)
        
        if apply_filters:
            scans = [scan for scan in scans 
                     if (scan.gaussian_fit is not None)]
            scans = [scan for scan in scans
                     if not(np.any(np.isinf(scan.regr.dcoefficients)))]
            scans = [scan for scan in scans
                     if scan.gaussian_fit.amplitude >= 2 * scan.gaussian_fit.damplitude]
            scans = [scan for scan in scans
                     if (np.abs(scan.gaussian_fit.width) >= 2 * scan.gaussian_fit.dwidth)]
            
        self.number_filtered = initial_len_scans - len(scans)
        
        print(str(self.number_filtered) + 
              """ interferometer scans were found to have had problems with fitting.\n""" 
              + str(len(scans)) + """ scans will be included in the map.\n""")
                
        
        self.scans = scans
        
        self.center_x = center_x
        self.center_y = center_y
        self.reverse_x = reverse_x
        self.reverse_y = reverse_y
        
        self.null_peak_position = null_peak_position
        self.setpoint = setpoint
        
        self.aggregate_lineshape()
        self.create_map(order, max_sum, 
                        renormalize_error=renormalize_error,
                        remove_outliers=remove_outliers, **kwargs)
        
        print('the interferometer scan map has been created.')
        
    def __len__(self):
        return len(self.scans)
        
    def aggregate_lineshape(self):
        displacements = []
        contrasts = []
        for scan in self.scans:
            displacements = displacements + list(scan.arm_positions - 
                                                 scan.gaussian_fit.center)
            contrasts = contrasts + list((scan.fringe_contrasts - 
                                          scan.gaussian_fit.offset)
                                         /scan.gaussian_fit.amplitude)
        displacements = np.array(displacements)
        contrasts = np.array(contrasts)
        self.lineshape = GaussianFit1D(displacements, contrasts, offset=False)
        return 
    
    def create_map(self, order, max_sum, remove_outliers=False, **kwargs):
        x = []
        y = []
        center = []
        dcenter= []
        for scan in self.scans:
            x.append(scan.x_position)
            y.append(scan.y_position)
            center.append(scan.gaussian_fit.center)
            dcenter.append(scan.gaussian_fit.dcenter)
            
        x = np.array(x)
        y = np.array(y)
        if self.center_x is None:
            self.center_x = np.mean(x)
        if self.center_y is None:
            self.center_y = np.mean(y)
        x = x - self.center_x
        y = y - self.center_y
        if self.reverse_x:
            x = -x
        if self.reverse_y:
            y = -y
        
        center = np.array(center)
        dcenter= np.array(dcenter)
        
        center = np.abs(center - self.null_peak_position) - self.setpoint
        
        y_name = 's'
        if self.setpoint != 0:
            y_name = y_name + ' - ' + str(self.setpoint)
        y_name = '$' + y_name + '$'
        
        self.map = PolynomialFit2D([x, y], center, dcenter,
                                          order=order, max_sum=max_sum,
                                          x_names=['$x$', '$y$'], x_units=['\\mathrm{cm}', '\\mathrm{cm}'],
                                          y_name=y_name, y_unit='\\mu \\mathrm{m}',**kwargs)
        if remove_outliers:
            self.map.remove_outliers()
        return
    
    def plot(self, show_lineshape=True, h_angle=-115, v_angle=45, alpha_data=.01, 
             legend=False, legend_loc='upper left', max_line_length=180, p_coefficient=.25, **kwargs):
        if show_lineshape:
            fig = plt.figure()
            self.map.plot(h_angle=h_angle, v_angle=v_angle, **kwargs)
            if legend:
                plt.legend(loc=legend_loc)
            fig.add_subplot(333)
            self.lineshape.plot(alpha_data=alpha_data)
            plt.title('interferometer lineshape')
            plt.xlabel('arm displacement ($\mu \\mathrm{m}$)')
            _=plt.ylabel('contrast (arb)')
            
            title = self.map.model_string_with_coefficients(max_line_length=max_line_length, 
                                                                p=p_coefficient)
            fig.text(.22, .8, title, size=12)
            return fig
        else:
            return self.map.plot(**kwargs)
        
    @staticmethod
    def create_from_txt(filenames, **kwargs):
        """
        import interferometer scans from the files in filenames and then create
        and InterferometerMap object
        """
        print("""
        :::::::::::::::Creating Interferometer Scan Map:::::::::::::::
        importing scans from file and fitting the scans to gaussians.
        """)
        t0 = time.time()
        scans = InterferometerData.read_from_txt(filenames)
        t1 = time.time()
        print('Completed in ' + str(t1 - t0) + ' seconds.\n')
        return InterferometerMap(scans, **kwargs)