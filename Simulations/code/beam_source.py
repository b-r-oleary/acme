import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import pandas as pd
from datetime import datetime
import os
import pickle
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_filename(info=None, filename=None, file_extension='pkl', default_directory='./saved'):
    if filename is not None:
        if os.path.split(filename)[1] == '':
            default_directory = os.path.split(filename)[0]
            filename = None
    if filename is None:
        now = datetime.now().strftime('%Y-%m-%dT%H-%M')
        filename = (default_directory + '/')
        if info is not None:
            filename = filename + info
        filename = filename + now + '.' + file_extension
    else:
        directory, name = os.path.split(filename)
        if directory == '':
            if os.path.isdir(default_directory):
                directory = default_directory
            else:
                directory = ''
        name = name.split('.')
        if len(name) > 1:
            extension = name[-1]
            name = '.'.join(name[:-1])
        else:
            extension = file_extension
            name = '.'.join(name)
        filename = directory + '/' + name + '.' + extension
    return filename


class SimulationParameters(object):
    # import default parameters into this class object
    default_parameters_file = 'acme_simulation_parameters.py'
    try:
        execfile(default_parameters_file)
    except:
        default_parameters_file = './code/' + default_parameters_file
        execfile(default_parameters_file)
    
    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            self.set(k, v)
        return
                
    def set(self, varname, value):
        if varname in self.__class__.__dict__.keys():
            setattr(self, varname, value)
        else:
            raise Warning(varname + ' is not a valid parameter.')
        return
    
    def save(self, filename):
        filename = create_filename(info='simulation_parameters_', filename=filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print('saved: ' + filename)
        return


class Collimator(object):
    """
    This is an object used to represent a molecular beam collimator.
    This object trims a set of molecular trajectories in a Trajectory object
    so that the remaining trajectories satisfy the collimator constraint.
    """
    
    def __init__(self, collimator_geometry, 
                       collimator_parameters):
        """
        input:
        geometry - string (accepted: 'circle', 'rectangle'
        parameters - tuple (various format depending on the geometry)
        """
        self.geometry   = collimator_geometry
        self.parameters = collimator_parameters
        if self.geometry == 'circle':
            self.x = self.parameters[1]
        elif self.geometry == 'rectangle':
            self.x = self.parameters[2]
        else:
            raise IOError('collimator geometry is not defined.')
        
    def collimate(self, trajectory):
        if len(trajectory) == 0:
            return
        if self.geometry == 'circle':
            r0, x0, y0, z0 = self.parameters
            
            X = trajectory.position(l=x0)
            y = X[:,1]
            z = X[:,2]
            condition = np.where((y - y0)**2 + (z - z0)**2 <= r0**2)
            trajectory.trim(condition)
            
        elif self.geometry == 'rectangle':
            yl, zl, x0, y0, z0 = self.parameters

            X = trajectory.position(l=x0)
            y = X[:,1]
            z = X[:,2]
            condition = np.where(np.logical_and(
                                 np.abs(y - y0) <= yl/2.0,
                                 np.abs(z - z0) <= zl/2.0))
            trajectory.trim(condition)
            
        if self not in trajectory.collimators:
            trajectory.collimators.append(self)
        return
    
    def plot(self, N=100, color='k', alpha=.5):
        if self.geometry == 'circle':
            phi = np.linspace(0, 2*np.pi, N)
            r0, x0, y0, z0 = self.parameters
            y = r0 * np.cos(phi) + y0
            z = r0 * np.sin(phi) + z0
            x = x0 * np.ones(y.shape)
        elif self.geometry == 'rectangle':
            yl, zl, x0, y0, z0 = self.parameters
            y = np.array([-1, 1, 1, -1, -1]) * (yl/2.0) + y0
            z = np.array([-1, -1, 1, 1, -1]) * (zl/2.0) + z0
            x = x0 * np.ones(y.shape)
        plt.plot(x, z, y, color=color, alpha=alpha)
        return
    
class Trajectory(object):
    """
    This is an object that holds the information
    and methods required to calculate the trajectories of a set of
    molecules. This particular class only includes ballistic trajectories
    but could be expanded to include more complex methods.
    """
    description = 'trajectories'
    
    def __init__(self, x, v, t=0, 
                 collimators=None, beam_source=None, parameters=None):
        """
        inputs:
        x: array of 3-vector of floats, positions in x, y, z
        v: array of 3-vector of floats, velocities vx, vy, vz
        t: array of floats, time that corresponds to this position and velocity.
        collimators: a list of collimator objects that have affected this
        list of trajectories - these are registered for future reference and plotting
        purposes.
        beam_source: a reference to the beam source object that created these trajectories
        (this is useful in case the BeamSource parameters are forgotten, but the trajectories
        are saved, and we would like to use them).
        """
        self.x = np.array(x)
        self.v = np.array(v)
        self.t = np.array(t)
        
        if collimators is None:
            collimators = []
        self.collimators = collimators
        
        self.beam_source = beam_source
        self.params = parameters
        
        
    def to_dataframe(self, l=None, t=None, latex=True, units=True):
        if l is None and t is None:
            l = 0
        X = self.position(l=l, t=t)
        V = self.velocity(l=l, t=t)
        T = self.time(l=l, t=t)
        
        table = {'x':X[:,0], 'y':X[:,1], 'z':X[:,2],
                 'v_x':V[:,0], 'v_y':V[:,1], 'v_z':V[:,2],
                 't':T}
        
        table = {self._reformat_label(k, latex, units):v for k, v in table.items()}
        return pd.DataFrame(table)
        
    def __len__(self):
        if len(self.t.shape) == 0:
            return 0
        else:
            return self.t.shape[0]
        
    def __repr__(self):
        return str(self)
    
    def __str__(self):
        string = (("Array of " + self.description + "\ntrajectory statistics at x=0:\n") 
                  + str(self.statistics(l=0)))
        return string
        
    def statistics(self, l=None, t=None, latex=False, units=False):
        """
        provides a table of statistics given a specified
        beamline position l, or given a time t.
        """
        if l is None and t is None:
            l=0
        
        pos  = self.position(l=l, t=t)
        vel  = self.velocity(l=l, t=t)
        time = self.time(l=l, t=t)
        
        parameters = {'x': pos[:,0], 'y': pos[:,1], 'z': pos[:,2],
                      'v_x': vel[:,0], 'v_y': vel[:,1], 'v_z': vel[:,2],
                      't': time}
        methods = [np.mean, (lambda x: np.std(x)/np.sqrt(len(x))),
                   np.std, np.min, np.max, len]
        
        table = {k: [self._get_units(k)] + [method(v) for method in methods]
                 for k, v in parameters.items()}
        
        table = {self._reformat_label(k,latex,units):v for k, v in table.items()}
        
        method_names = ['unit', 'mean', 'dmean', 'std', 'min', 'max', 'N']
        table = pd.DataFrame(table, index=method_names)
        return table.T
    
    def histogram(self,x=None, y=None, l=None, t=None, **kwargs):
        """
        this is a short-cut for creating many possible histograms, at a
        specified beamline location l, or specified time t.
        - if x and y are not input, then it creates a full joint-scatterplot
          for each pair of variables (7 variables total: x,y,z, vx, vy, vz, t)
        - if x is input, it creates a 1d histogram with respect to that parameter
        - if x and y are input, creates a 2d histogram with respect to those parameters
        """
        table = self.to_dataframe(l=l, t=t, latex=True)
        if x is None and y is None:
            g = sns.pairplot(table, **kwargs)
            for ax in g.axes.flat:
                _ = plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
            return
        if x is not None and y is None:
            x = self._reformat_label(x)
            sns.distplot(table[x], **kwargs)
            plt.xlabel(x)
            return
        if x is not None and y is not None:
            x = self._reformat_label(x)
            y = self._reformat_label(y)
            sns.jointplot(x=x, y=y, data=table, **kwargs);
            return
        
    def _get_units(self, label):
        units = {'x':'m','y':'m','z':'m',
                 'v_x':'m/s','v_y':'m/s','v_z':'m/s','t':'s'}
        return units[label]
        
    def _reformat_label(self,label, latex=True, units=True):
        new_label = label
        if latex:
            new_label = '$' + new_label + '$'
        if units:
            unit = self._get_units(label)
            if latex:
                unit = '$\\mathrm{' + unit + '}$'
            new_label = new_label + ' (' + unit + ')'
        return new_label
    
    def trim(self, condition):
        self.x = self.x[condition]
        self.v = self.v[condition]
        self.t = np.squeeze(self.t[condition])
        return
        
    def position(self, l=None, t=None, squeeze=True):
        """
        return the position of the molecules at a given beamline
        position l, or at a given time t
        """
        raise RuntimeError('this is a virtual method that must be overwritten')
        return
            
    def velocity(self,l=None, t=None):
        """
        return velocities at a given position l, or a given time t
        (since this is just a ballistic model, the output does
         not depend on l or t, but it could in a non-ballistic model)
        """
        raise RuntimeError('this is a virtual method that must be overwritten')
        return
    
    def time(self,l=None, t=None, squeeze=True):
        """
        return times corresponding to when the trajectories cross beamline position l
        (or simply the input time if t is input for some reason).
        """
        raise RuntimeError('this is a virtual method that must be overwritten')
        return
    
    def doppler_shift(self, wavelength, direction, l=None, t=None):
        """
        returns angular frequency in Hz for the doppler shift
        at the input wavelength in m in the input direction
        """
        velocity = self.velocity(l=l, t=t)
        direction = np.array(direction)
        if len(direction) != 3:
            raise IOError(' the input direction must have length 3')
        direction = direction/np.sqrt(np.sum(direction**2))
        doppler_shift = np.dot(velocity, direction) * 2 * np.pi / wavelength
        return doppler_shift
    
    def plot(self, lmin=None, lmax=None, t=None, N=2, 
             color='navy', alpha_points=.14, alpha_lines=.01,
             plot_collimators = True, markersize=6,
             pane_color=(1.0, 1.0, 1.0, 1.0), grid_linewidth=.5, ax=None):
        if lmin is not None and lmax is not None:
            self.trajectory_plot(lmin, lmax, color=color, 
                                 N=N, alpha_points=alpha_points,
                                 alpha_lines=alpha_lines,
                                 markersize=markersize,
                                 pane_color=pane_color,
                                 grid_linewidth=grid_linewidth, ax=ax)
        elif t is not None:
            self.time_plot(t, color=color, alpha=alpha_points,
                           markersize=markersize, pane_color=pane_color,
                           grid_linewidth=grid_linewidth, ax=ax)
        if plot_collimators:
            for collimator in self.collimators:
                if (lmin is None) or (lmax is None):
                    collimator.plot()
                else:
                    if (collimator.x >= lmin) and (collimator.x <= lmax):
                        collimator.plot()
        return
    
    def trajectory_plot(self, lmin, lmax, color='lightblue', 
                        N=2, alpha_points=.1, alpha_lines=.0075,
                        points_at_collimators=True, markersize=6,
                        pane_color=(1.0, 1.0, 1.0, 1.0), grid_linewidth=.5, ax=None):
        
        lmin = min([lmin, lmax])
        lmax = max([lmin, lmax])
        
        l = np.linspace(lmin, lmax, N)
        pos = self.position(l=l)
        if ax is None:
            ax = plt.axes(projection='3d')
        for i in range(pos.shape[1]):
            ax.plot(pos[:,i,0], pos[:,i,2], pos[:,i,1], color=color, alpha=alpha_lines)
        if points_at_collimators:
            for collimator in self.collimators:
                if ((collimator.x <= lmax) and (collimator.x >= lmin)):
                    pos = self.position(l=collimator.x)
                    ax.plot(pos[:,0], pos[:,2], pos[:,1],'.',color=color,
                            alpha=alpha_points, markersize=markersize)
        self._set_axis_labels(pane_color, grid_linewidth)
        return
    
    def time_plot(self, t, color='lightblue', alpha=.5, markersize=6,
                  pane_color=(1.0, 1.0, 1.0, 1.0), grid_linewidth=.5, ax=None):
        pos = self.position(t=t, squeeze=False)
        if ax is None:
            ax = plt.axes(projection='3d')
        if not isinstance(color, (list, tuple)):
            color = [color] * pos.shape[0]
        for i in range(pos.shape[0]):
            ax.plot(pos[i,:,0], pos[i,:,2], pos[i,:,1], '.', color=color[i], 
                    alpha=alpha, markersize=markersize)
        self._set_axis_labels(pane_color, grid_linewidth)
        return
    
    def _set_axis_labels(self, pane_color, grid_linewidth):
        ax = plt.gca()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_zlabel('y (m)')
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        extent = np.max(np.abs(list(ylim) + list(zlim)))
        ax.set_ylim([-extent, extent])
        ax.set_zlim([-extent, extent])
        plt.gca().invert_zaxis()
        ax.w_xaxis.set_pane_color(pane_color)
        ax.w_yaxis.set_pane_color(pane_color)
        ax.w_zaxis.set_pane_color(pane_color)

        ax.w_xaxis.gridlines.set_lw(grid_linewidth)
        ax.w_yaxis.gridlines.set_lw(grid_linewidth)
        ax.w_zaxis.gridlines.set_lw(grid_linewidth)
        return
    
    def __add__(self,trajectory):
        if len(self) == 0:
            return trajectory
        elif len(trajectory) == 0:
            return self
        else:
            return type(self)(np.append(self.x, trajectory.x, axis=0),
                              np.append(self.v, trajectory.v, axis=0),
                              np.append(self.t, trajectory.t, axis=0),
                              collimators=self.collimators,
                              parameters=self.params)
    
    def __getitem__(self, i):
        return type(self)(self.x[i,:], self.v[i,:], self.t[i], 
                          collimators=self.collimators,
                          parameters=self.params)
    
    def save(self, filename=None):
        filename = create_filename(str(len(self)) + '_trajectories_', filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print('saved: ' + filename)
        return
    
    
class BallisticTrajectory(Trajectory):
    
    description = 'ballistic trajectories'
    
    def init(self, *args, **kwargs):
        super(BallisticTrajectory, self).__init__(*args, **kwargs)
    
    def position(self, l=None, t=None, squeeze=True):
        """
        return the position of the molecules at a given beamline
        position l, or at a given time t
        """
        # convert the l or t input to a t input:
        if t is None and l is None:
            raise IOError('you must input l or t.')
        if t is None:
            l = np.array(l)
            if len(l.shape) == 0:
                l = np.array([l])
            t = self.time(l=l)
        else:
            t = np.array(t)
            if len(t.shape) == 0:
                t = np.array([t])
        
        #formatting to make sure I can perform the array operations
        g = np.expand_dims(
            np.expand_dims(self.params.GRAVITY,0),0)
        t = np.expand_dims(t - self.t,2)
        x = np.expand_dims(self.x,0)
        v = np.expand_dims(self.v,0)
        
        pos = x + v * t + (1/2.0) * g * t**2
        
        if squeeze:
            pos = np.squeeze(pos)
        return pos
            
    def velocity(self,l=None, t=None, squeeze=True):
        """
        return velocities at a given position l, or a given time t
        (since this is just a ballistic model, the output does
         not depend on l or t, but it could in a non-ballistic model)
        """
        # convert the l or t input to a t input:
        if t is None and l is None:
            raise IOError('you must input l or t.')
        if t is None:
            l = np.array(l)
            if len(l.shape) == 0:
                l = np.array([l])
            t = self.time(l=l)
        else:
            t = np.array(t)
            if len(t.shape) == 0:
                t = np.array([t])
        
        #formatting to make sure I can perform the array operations
        g = np.expand_dims(
            np.expand_dims(self.params.GRAVITY,0),0)
        t = np.expand_dims(t - self.t,2)
        x = np.expand_dims(self.x,0)
        v = np.expand_dims(self.v,0)
        
        vel = v + g * t
        
        if squeeze:
            vel = np.squeeze(vel)
        return vel
        
    
    def time(self,l=None, t=None, squeeze=True):
        """
        """
        if l is not None:
            l = np.array(l)
            if len(l.shape) == 0:
                l = np.array([l])
                
            g = self.params.GRAVITY[0]
            v0= np.squeeze(self.v[:,0])
            x0= np.squeeze(self.x[:,0])
            t0= self.t
            
            if g == 0:
                t = np.array([
                        t0 + (x - x0)/v0
                        for x in l])
            else:
                t = np.array([
                        (g * t0 - v0 + np.sqrt(v0**2 + 2*g*(x - x0))) / g
                        for x in l])
            
            if squeeze:
                t = np.squeeze(t)
            return t
        elif t is not None:
            t = np.array(t)
            if len(t.shape) == 0:
                t = np.array([t])
            t = np.expand_dims(t,1) * np.ones((1,len(self.t)))
            if squeeze:
                t = np.squeeze(t)
            return t
        else:
            raise IOError('must input l or t')
            
            
class BeamSource(object):
    """
    This object represents the molecular beam source. It loads default
    molecular beam parameters into the namespace from the file
    "acme_simulation_parameters". Any of these parameter may be overrided
    by keyword inputs to the BeamSource object. Additionally,
    default collimators are loaded - you can add to, or delete from this collimator list.
    
    The main method is 'generate_molecules' which provides a Trajectory object with
    a list of randomly generated molecular trajectories according to the parameters laid out
    in the BeamSource.
    """
    
    def __init__(self, collimators=None, parameters=None, trajectory=BallisticTrajectory,**kwargs):
        if parameters is None:
            parameters = SimulationParameters()

        for k, v in kwargs.items():
            parameters.set(k, v)
            
        self.params      = parameters
            
        if collimators is None:
            collimators = self.default_collimators()
            
        self.collimators = collimators
        self.trajectory_class  = trajectory
                
    def default_collimators(self):
        collimators = [Collimator(self.params.CONICAL_APERTURE_GEOMETRY,
                                  self.params.CONICAL_APERTURE_PARAMETERS),
                       Collimator(self.params.FIXED_COLLIMATOR_GEOMETRY,
                                  self.params.FIXED_COLLIMATOR_PARAMETERS)]
        return collimators
    
    def molecule_source(self, N=1):
        if self.params.SOURCE_VELOCITY_DISTRIBUTION == 'normal':
            v_bar, dv = self.params.SOURCE_VELOCITY_PARAMETERS
            v = np.random.normal(size=(N,3)) * np.array(dv) + np.array(v_bar)
        else:
            raise IOError('alternative velocity distribution not yet defined')
            
        if self.params.SOURCE_POSITION_DISTRIBUTION == 'uniform circle':
            r0, x0, y0, z0 = self.params.SOURCE_POSITION_PARAMETERS
            r   = r0 * np.sqrt(np.random.uniform(size=(N,)))
            phi = 2 * np.pi * np.random.uniform(size=(N,))
            
            Y = r * np.cos(phi) + y0
            Z = r * np.sin(phi) + z0
            X = x0 * np.ones(Y.shape)
            
            x = np.array([X, Y, Z]).T
        else:
            raise IOError('alternative position distribution not yet defined')
            
        if self.params.SOURCE_EXTRACTION_TIME_PROFILE == 'normal':
            t_bar, dt = self.params.SOURCE_EXTRACTION_TIME_PARAMETERS
            t = np.random.normal(size=(N,)) * dt + t_bar
        else:
            raise IOError('alternative time extraction distribution not yet defined')
        
        return self.trajectory_class(x, v, t, parameters=self.params)
    
    def collimate(self, trajectory):
        for collimator in self.collimators:
            collimator.collimate(trajectory)
        return
    
    def generate_molecules(self, N=1, timeout=5):
        t0 = time()
        trajectory = self.molecule_source(N)
        self.collimate(trajectory)
        while len(trajectory) < N:
            new_trajectory = self.molecule_source(N)
            self.collimate(new_trajectory)
            trajectory = new_trajectory + trajectory
            if (time() - t0) >= timeout:
                raise Warning(('molecule generation timed out - ' 
                               + str(len(trajectory)) + ' molecules generated.'))
                break
        if len(trajectory) > N:
            trajectory = trajectory[0:N]
        # tag the trajectory with the beam source:
        trajectory.beam_source = self
        return trajectory