import sys
sys.path.append('../Statistics/code')
from statfunctions import *

import re
from datetime import datetime
import numpy as np
import time
import os
import pandas as pd

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
                 labview_gaussian_fit=None,
                 residual_std_estimate=True,
                 residual_std_estimate_window=6,
                 renormalize_error=None,
                 **kwargs):
        """
        inputs:
        date: (datetime) time when scan was taken
        x_position: (float in cm) x position of velmex translation stage
        y_position: (float in cm) y position of velmex translation stage
        arm_positions: (numpy array of floats in um) position of zaber interferometer translation stage
        fringe_contrasts: (numpy array of floats in arb units) measured fringe contrast from interferometer
        labview_gaussian_fit: (GaussianFit object) contains fit parameters from labview gaussian fit to the data
        residual_std_estimate: (Boolean) indicates whether or not to estimate the x variation in uncertainty using residuals
                               (see statfunctions.
        residual_std_estimate_window: (int) number of points to sample when performing the residual_std_estimate
        kwargs: passed to GaussianFit1D object
        """
        # generally, the labview code puts NaN as the position for an axis when
        # I am not scanning that axis. I only do scans along the x or y axis,
        # so in these cases, I am setting those parameters to zero.
        if np.isnan(x_position):
            x_position = 0
        if np.isnan(y_position):
            y_position = 0
            
        if renormalize_error is None:
            renormalize_error = not(residual_std_estimate)
        
        self.date = date
        self.x_position = x_position
        self.y_position = y_position
        self.arm_positions = arm_positions
        self.fringe_contrasts = fringe_contrasts
        self.labview_gaussian_fit = labview_gaussian_fit
        # create the regression object to fit the data to a gaussian:
        self.regr = GaussianFit1D(self.arm_positions, self.fringe_contrasts, 
                                      offset=True, x_name='arm position', y_name='fringe contrast',
                                      x_unit='$\mu$m', y_unit='arb',auto_fit=False,
                                      start_point=[self.labview_gaussian_fit.amplitude,
                                                   self.labview_gaussian_fit.center,
                                                   self.labview_gaussian_fit.width,
                                                   min(self.fringe_contrasts * (self.fringe_contrasts > 0))],
                                      residual_std_estimate=residual_std_estimate,
                                      residual_std_estimate_window=residual_std_estimate_window,
                                      renormalize_error=renormalize_error,
                                      **kwargs)
        try:
            # perform the fit, and save the coefficients, provided that the scan suceeeds
            self.regr.fit()
            self.gaussian_fit = GaussianFit(amplitude=self.regr.coefficients[0],
                                            damplitude=self.regr.dcoefficients[0],
                                            center=self.regr.coefficients[1],
                                            dcenter=self.regr.dcoefficients[1],
                                            width=self.regr.coefficients[2],
                                            dwidth=self.regr.dcoefficients[2],
                                            offset=self.regr.coefficients[3],
                                            doffset=self.regr.dcoefficients[3])
            self.regr.label = (str(self.date) + 
                               ', x=' + str(self.x_position) + 
                               'cm , y=' + str(self.y_position)  + 'cm')
        except:
            print('fit to scan failed')
            self.gaussian_fit = None
        
    def __str__(self):
        output =      {'date': str(self.date),
                       'x_position (cm)': self.x_position,
                       'y_position (cm)': self.y_position}
        try:
            output['gaussian_fit'] = str(self.gaussian_fit)
        except:
            output['labview_gaussian_fit'] = str(self.labview_gaussian_fit)
        return str(output)
    
    def __repr__(self):
        return str(self)
        
    @staticmethod
    def read_from_txt(filenames=None, path=None):
        """
        this is a static method for reading in data from a list of
        input filenames and returning a list of InterferometerScan objects
        with the corresponding data
        
        inputs:
        filenames: list of paths to create interferometer scans from.
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
        """
        show the regression plot corresponding to the gaussian fit to data.
        """
        self.regr.plot(**kwargs)
        return
        
        
class InterferometerMap(object):
    def __init__(self, scans, center_x=None, center_y=None, 
                 reverse_x=True, reverse_y=False,
                 null_peak_position=0, setpoint=45000,
                 order=4, max_sum=True, 
                 apply_filters=True, renormalize_error=True,
                 remove_outliers=False, auto_fit=True, title=None, **kwargs):
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
        auto_fit: (Boolean) automatically fit the data upon creation of the regression object
        title: a label for the map (for plotting and saving)
        **kwargs: passed to the PolynomialFit2D() regression object.
        """
        
        initial_len_scans = len(scans)
        
        # apply filters to the input scans to remove those that I consider to
        # be unsuitable:
        if apply_filters:
            # the gaussian fit failed
            scans = [scan for scan in scans 
                     if (scan.gaussian_fit is not None)]
            # the uncertainties blew up
            scans = [scan for scan in scans
                     if not(np.any(np.isinf(scan.regr.dcoefficients)))]
            # the signal to noise is too small on amplitude
            scans = [scan for scan in scans
                     if scan.gaussian_fit.amplitude >= 2 * scan.gaussian_fit.damplitude]
            # the signal to noise on teh width is too small
            scans = [scan for scan in scans
                     if (np.abs(scan.gaussian_fit.width) >= 2 * scan.gaussian_fit.dwidth)]
            
        self.number_filtered = initial_len_scans - len(scans)
        
        # output how many scans were removed by filters
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
        
        self.title = title
        
        self.aggregate_lineshape()
        if auto_fit:
            self.create_map(order, max_sum, 
                            renormalize_error=renormalize_error,
                            remove_outliers=remove_outliers, **kwargs)
        
        print('the interferometer scan map has been created.')
        
    def __len__(self):
        return len(self.scans)
        
    def aggregate_lineshape(self):
        """
        take the data from all of the scan objects,
        subtract the fitted offset, shift by the fitted center to zero,
        and scale the data by the fitted amplitude. Then take the aggregate of
        all data and perform a gaussian fit to it.
        
        this will provide a good estimate of the width of the contrast lineshape
        and is produced mostly as a check that the fits were alright.
        """
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
        # create the fit to the data
        self.lineshape = GaussianFit1D(displacements, contrasts, offset=False,
                                       x_name='arm position', x_unit='$\mu$m',
                                       y_name='contrast', y_unit='arb')
        return 
    
    def create_map(self, order, max_sum, remove_outliers=False, **kwargs):
        """
        Performs a polynomial fit to the output centers of the interferometer
        scans vs position to create a map. If the input data is 1D, it produces
        a 1D fit, if the input data is 2D, it produces a 2D fit.
        
        inputs:
        order: (int) order of the polynomial fit
        max_sum: (Boolean) whether the polynomial order should refer to the maximum of the sum
                 of the exponents, or to the maximum of the exponents for a given variable in a term.
        remove_outliers: (Boolean) performs an interative fitting method to deduce and remove outliers,
                 assuming that the data is normally distributed.
        **kwargs: passed to the regression object.
        """
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

        # if the data is 1d, switch to the 1d fitting method
        if (len(set(list(x))) == 1) or (len(set(list(y))) == 1):
            if len(set(list(x))) == 1:
                coord = y
                coord_name = '$y$'
            else:
                coord = x
                coord_name = '$x$'
            return self.create_1d_map(coord, coord_name,
                                      center, dcenter, y_name,
                                      order, remove_outliers=remove_outliers, **kwargs)
        
        self.map_vars = ['$x$','$y$']
        
        # if the data is 2d, using the 2d fitting method:
        self.map = PolynomialFit2D([x, y], center, dcenter,
                                          order=order, max_sum=max_sum,
                                          x_names=self.map_vars, x_units=['\\mathrm{cm}', '\\mathrm{cm}'],
                                          y_name=y_name, y_unit='\\mu \\mathrm{m}', **kwargs)
        if remove_outliers:
            self.map.remove_outliers()
        return
    
    def create_1d_map(self, coord, coord_name,
                      center, dcenter, y_name,
                      order, remove_outliers=False, **kwargs):
        """
        1d fitting method called when the data is for a 1d map. This method
        is called by create_map, and is fed all of the data required.
        """
        
        self.map_vars = [coord_name]

        self.map = PolynomialFit1D(coord, center, dcenter,
                                              order=order,
                                              x_name=coord_name, x_unit='$\\mathrm{cm}$',
                                              y_name=y_name, y_unit='$\\mu \\mathrm{m}$',label=self.title, **kwargs)
        if remove_outliers:
            self.map.remove_outliers()
        return
    
    def plot(self, show_lineshape=True, h_angle=-115, v_angle=45, alpha_data=.01, 
             legend=False, legend_loc='upper left', max_line_length=180, p_coefficient=.25, **kwargs):
        """
        display the 2D map, and overlay it with an aggregate lineshape plot.
        
        show_lineshape: (Boolean) show the aggregate lineshape plot overlayed
        h_angle: the horizontal viewing angle in degrees
        v_angle: the vertical viewing angle in degrees
        alpha_data: the transparency of the data
        legend: (Boolean) whether or not to show the legend
        legend_loc: (string) location of the legend
        max_line_length: (int) max number of characters to include in the fit string in the title
        p_coefficient: (float) p value threshold for including a term in the fit string in the title
        **kwargs: passed to the regression plotting method.
        """
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
    
    def get_data(self):
        """
        produce a pandas dataframe
        with the peak center data and uncertainties
        as a function of x and y.
        """
        data = self.map.data
        if len(self.map_vars) == 2:
            x = data.x[0]
            y = data.x[1]
        else:
            if self.map_vars[0] == '$x$':
                x = data.x[0]
                y = np.array([self.scans[0].y_position] * len(x))
            else:
                y = data.x[0]
                x = np.array([self.scans[0].x_position] * len(y))
        s = data.y + self.setpoint
        ds= data.dy
        table = pd.DataFrame(np.array([x, y, s, ds]).T, 
                             columns=['x (cm)',
                                      'y (cm)',
                                      's (um)',
                                      'ds (um)'])
        return table

    def save_to_txt(self, name=None, path=None):
        """
        save the data from self.get_data() pandas dataframe
        to txt file in case that is easier for future access for labmates.
        """
        if path is None:
            if 'data' in os.listdir('./'):
                directory = './data'
            else:
                directory = './'
            if name is None:
                name = 'interferometer_scan_' + self.scans[0].date.strftime(format='%Y-%m-%d') + '.txt'
            path = os.path.join(directory, name)

        table = self.get_data()
        table.to_csv(path, sep='\t')
        return