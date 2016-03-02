import pyodbc
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib
import seaborn as sns
import pandas as pd
sns.set_context("notebook", font_scale=1.25)
sns.set_style('white')
import re
import numpy as np
from textwrap import wrap
import cPickle as pickle
import os
from getpass import getpass
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy


def get_data_files(directory='./', file_extensions=['pkl'], recursive_search=True):
    
    if directory[-1] not in ['/', '\\']:
        directory = directory + '/'
        
    items = os.listdir(directory)
    files = []
    
    for item in items:
        if item.split('.')[-1] in file_extensions:
            files.append(directory + item)
        if recursive_search:
            subdirectory = directory + item
            if os.path.isdir(subdirectory):
                more_files = get_data_files(directory=subdirectory,
                                            file_extensions=file_extensions,
                                            recursive_search=True)
                files = files + more_files
                
    return files
            

def load(filename=None, directory='./'):
    if filename is None:
        files = get_data_files(directory=directory,
                               file_extensions=['pkl'],
                               recursive_search=True)
        files = pd.Series(files)
        print(files)
        index = input("""input an integer corresponding to the file that you would like to load: """)
        filename = files[index]
    with open(filename, 'rb') as f:
        output = pickle.load(f)
    print('loaded: ' + filename)
    return output
        
def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print('saved: ' + filename)
    return

def time_string_to_seconds(time):
    #the time string can use the following units with the following conversions to seconds:
    time_units={'ms':.001, 's':1, 'min':60, 'hr':3600, 
                'day':86400, 'week':604800, 'month':2592000,
                'year':31536000}
    
    variants = {'ms':'ms', 'millisecond':'ms', 
                's':'s', 'second':'s', 'sec':'s',
                'min':'min', 'minute':'min',
                'hr':'hr', 'hour':'hr',
                'day': 'day',
                'week': 'week', 'wk': 'week',
                'month': 'month',
                'year':'year', 'yr': 'year'}
    
    variants_allowed = '|'.join(variants.keys())
    
    matches    = re.findall((r'\s*?([\-\d\.]*?)\s*?(' 
                            + variants_allowed + ')s?\s*?'), time)
    total = 0
    for match in matches:
        quantity = (float(match[0]) if match[0] != '' else 1)

        try:
            unit     = variants[match[1]]
        except:
            raise IOError('your time unit ' + match[1] + ' is not recognized')

        seconds = quantity * time_units[unit]

        total += seconds
        
    return total

def get_time_range(duration='1 hour', 
                   start_time=None, 
                   end_time=None):
    
    delta_time = datetime.timedelta(0,time_string_to_seconds(duration))
    
    if start_time is None and end_time is None:
        end_time = datetime.datetime.now()
        start_time = end_time - delta_time
        
    if start_time is not None and end_time is None:
        end_time = start_time + delta_time
        
    if end_time is not None and start_time is None:
        start_time = end_time - delta_time
        
    return start_time, end_time

def create_filename(start_time=None, end_time=None, channels=None, fmt='%Y-%m-%dT%H-%M'):
    string = []
    if start_time is not None:
        string.append(start_time.strftime(fmt))
    if end_time is not None:
        string.append(end_time.strftime(fmt))
    if channels is not None:
        if isinstance(channels, str):
            channels = [channels]
        string = string + channels
    string = '_'.join(string)
    return '_'.join(string.split(' '))


class TimeSeries(object):
    """
    this is a data object that is used to store data obtained from
    the database, and used to plot and save the data to local files
    and figures.
    """
    
    def __init__(self, t=None, x=None, channel=None):
        
        self.t = t
        self.x = x
        self.channel = channel
        
    def plot(self, formatting=True, y_style='sci', 
             wrap_length=30, remove_extreme_yticks=False):
        if remove_extreme_yticks:
            mult_factor = 10**np.ceil(np.log10(np.max(self.x)))
            plt.plot(self.t, self.x / mult_factor, label=self.label())
            ylabel = self.label(units=True, wrap_length=wrap_length,
                                mult_factor=mult_factor)
        else:
            plt.plot(self.t, self.x, label=self.label())
            ylabel = self.label(units=True, wrap_length=wrap_length)
        plt.ylabel(ylabel)
        if formatting:
            self._plot_formatting(y_style=y_style, 
                                  remove_extreme_yticks=remove_extreme_yticks)
        return
        
    def _plot_formatting(self, y_style='sci', remove_extreme_yticks=False):
        plt.xlabel('time')
        ax = plt.gca()
        hfmt = dates.DateFormatter('%m/%d %H:%M')
        ax.xaxis.set_major_formatter(hfmt)
        if y_style == 'sci':
            plt.ticklabel_format(style=y_style, axis='y', scilimits=(-3,4))
        fig = plt.gcf()
        fig.autofmt_xdate()
        if remove_extreme_yticks:
            yticks = plt.gca().get_yticks()[0:-1]
            plt.gca().set_yticks(yticks)
        return
    
    def label(self, units=False, wrap_length=50, mult_factor=None):
        
        unit = []
        if mult_factor is not None:
            unit.append('%0.0E' % (mult_factor,))
        if units:
            unit.append(self.channel['UnitName'])
            
        unit = ' '.join(unit)
        
        if unit != '':
            l = (self.channel['LogChannelName'] 
                + ' (' + unit + ')')
        else:
            l = self.channel['LogChannelName']
        return '\n'.join(wrap(l, wrap_length))
    
    def save(self, filename=None):
        if filename is None:
            filename = create_filename(self.t[0], self.t[-1], [self.channel['LogChannelName']]) + '.pkl'
        save(self, filename)
        return
    
    
class TimeSeriesArray(object):
    
    def __init__(self, time_series, name=None):
        
        try:
            time_series[0]
        except:
            time_series = [time_series]
            
        self.time_series = time_series
                
        self.same_units = (len(set([ts.channel['UnitName'] 
                                   for ts in self.time_series])) == 1)
            
        self.name  = name
        self.units = (self[0].channel['UnitName'] if self.same_units else None)
        
    def __len__(self):
        return len(self.time_series)
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.time_series[i]
        elif isinstance(i, str):
            return self.time_series[self.channels().index(i)]
        else:
            raise IOError('you must input an index integer, or a valid channel name')
    
    def label(self, wrap_length=30):
        l = ''
        if self.name is not None:
            l = l + self.name
        if self.units is not None:
            l = l + ' (' + self.units + ')'
        l = '\n'.join(wrap(l,wrap_length))
        return l
    
    def plot(self, y_style='sci', subplots=None, wrap_length=None):
        
        if subplots is None:
            subplots = not(self.same_units)
        
        if subplots:
            fig = plt.figure(figsize=(7,2.75*len(self)))
            fig.subplots_adjust(hspace=0, wspace=0)
            if len(self) == 1:
                rows = 1
                columns = 1
            else:
                rows = len(self)
                columns = 1
                
        if wrap_length is None:
            if subplots:
                wrap_length = 20
            else:
                wrap_length = 30
                
        counter = 1
        for ts in self.time_series:
            if subplots:
                plt.subplot(rows, columns, counter)
                ts.plot(formatting=True, 
                        wrap_length=wrap_length, 
                        remove_extreme_yticks=True)
            else:
                ts.plot(formatting=False, wrap_length=wrap_length)
            counter += 1
        if not(subplots):
            ts._plot_formatting()
            plt.ylabel(self.label())
            plt.legend(loc='best')
        else:
            pass
            #plt.tight_layout()
        
        return
    
    def channels(self):
        return [ts.channel['LogChannelName'] for ts in self.time_series]
    
    def format_filename(self, filename=None, file_extension='pkl'):
        if filename is None:
            
            filename = ('./' + 
                        create_filename(self[0].t[0], self[0].t[-1], self.channels())
                        + '.' + file_extension)
            
        directory, f = os.path.split(filename)
        dirs = os.listdir(directory)
        if 'data' in dirs:
            if directory[-1] not in ['/','\\']:
                directory = directory + '/'
            filename = directory + 'data/' + f
        return filename
    
    def save(self, filename=None):
        filename = self.format_filename(filename, file_extension='pkl')
        save(self, filename)
        return
    
    def save_to_text(self, filename=None, fmt='%Y-%m-%dT%X.%f', precision=10):
        
        filename = self.format_filename(filename, file_extension='txt')
        
        output = ['\t'.join(
                    ['time\t' + ts.label(units=True, wrap_length=1000) 
                     for ts in self.time_series]
                            )]
        
        max_length = max([len(ts.t) for ts in self.time_series])
        for i in range(max_length):
            line = []
            for ts in self.time_series:
                try:
                    line.append(ts.t[i].strftime(fmt))
                    line.append(('%.' + str(precision) + 'g') % ts.x[i])
                except:
                    line.append('')
                    line.append('')
            line = '\t'.join(line)
            output.append(line)
        output = '\n'.join(output)

        with open(filename, 'wb') as f:
            f.write(output)
        print('saved: ' + filename)
        return
    

class TimeSeriesStates(object):
    
    def __init__(self, t, control_id, state_id, 
                 controls_list, states_list, title=None):
    
        controls_list = controls_list[controls_list['LogControlID'].isin(list(set(control_id)))]
        new_controls_list = controls_list.set_index('LogControlID')
        new_controls_list['LogControlName'] = controls_list.index
        controls_list = {i: new_controls_list.T[i] for i in new_controls_list.index}
        for k, v in controls_list.items():
            v['LogControlID'] = k
            controls_list[k] = v
        
        states_list   = states_list[states_list['LogControlStateID'].isin(list(set(state_id)))]
        new_states_list   = states_list.set_index('LogControlStateID')
        new_states_list['LogControlStateName'] = states_list.index
        states_list   = {i: new_states_list.T[i] for i in new_states_list.index}
        for k, v in states_list.items():
            v['LogControlStateID'] = k
            states_list[k] = v
        
        control      = [controls_list[c] for c in control_id]
        state        = [states_list[s] for s in state_id]
        notification = [c['LogControlName'] + ' ' + s['NotificationText'] for c, s in zip(control, state)]
        
        self.t            = t
        self.control_id   = control_id
        self.control      = control
        self.state_id     = state_id
        self.state        = state
        self.notification = notification
        
        self.states_list   = states_list
        self.controls_list = controls_list
        
        self.title = title
        
    def get_controls(self):
        return [v['LogControlName'] for v in self.controls_list.values()]
        
    def __str__(self):
        return 'TimeSeriesStates object inluding controls: ' + str(self.get_controls())
    
    def __repr__(self):
        return str(self)
        
    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, i):
        return self.get_discrete_waveform(i)
        
    def get_notifications(self, fmt=''):
        return [t.strftime('%c') + ' - ' + n for t, n in zip(self.t, self.notification)]
    
    def get_discrete_waveform(self, control_id):
        if isinstance(control_id, str):
            for k, v in self.controls_list.items():
                if v['LogControlName'] == control_id:
                    control_id = k
                    break
            if isinstance(control_id, str):
                raise IOError('please enter a valid control id or control name')
        
        inds = [i for i in range(len(self.control_id)) 
                   if self.control_id[i] == control_id]
        
        t = [self.t[i] for i in inds]
        state_ids = list(set([self.state_id[i] for i in inds]))
        if len(state_ids) == 0:
            x = np.array([])
        elif len(state_ids) == 1:
            x = np.ones(len(inds)) * .5
        else:
            x = np.array([state_ids.index(self.state_id[i])/float(len(state_ids) - 1) for i in inds])
        
        labels = [self.states_list[i]['LogControlStateName'] for i in state_ids]
        name = self.controls_list[control_id]['LogControlName']
        
        return t, x, name, labels
    
    def plot(self, size=.8, wrap_length=16, 
             label_fontsize=10, name_fontsize=14, title=None,
             show_points=True):
        counter = 0
        labels_list = []
        labels_pos  = []
        names = []
        colors = []
        for i in self.controls_list.keys():
            t, x, name, labels = self.get_discrete_waveform(i)
            if len(t) > 0:
                x = size * x + counter + (1-size)/2.0
                if show_points:
                    line = plt.step(t, x, '.', 
                                    label=name, markerfacecolor='w', markeredgewidth=.5)
                else:
                    line = plt.step(t, x, label=name)
                colors.append(line[0].get_color())
                labels_list = labels_list + ['\n'.join(wrap(l, wrap_length)) for l in labels]
                if len(labels) > 1:
                    labels_pos = labels_pos + list(np.linspace(0, 1, len(labels)) * size + counter + (1-size)/2.0)
                else:
                    labels_pos = labels_pos + [.5 + counter]
                names.append(name)
                counter += 1
            
        plt.yticks(labels_pos, labels_list, fontsize=label_fontsize)
        plt.ylim([0, counter])
        
        self._plot_formatting()
        
        xlim = plt.gca().get_xlim()
        rangex = (xlim[1] - xlim[0])/40.0
        xlim = [xlim[0] - rangex, xlim[1] + rangex]
        plt.xlim(xlim)
        x_pos = xlim[1] + rangex
        for i in range(len(names)):
            plt.text(x_pos, i + .5, names[i], color=colors[i], 
                     verticalalignment='center', fontsize=name_fontsize)
        if self.title is not None:
            plt.title(self.title + ' - Logged State Changes')
            
        return
    
    def _plot_formatting(self, fmt='%m/%d %H:%M'):
        plt.xlabel('time')
        ax = plt.gca()
        hfmt = dates.DateFormatter(fmt)
        ax.xaxis.set_major_formatter(hfmt)
        fig = plt.gcf()
        fig.autofmt_xdate()
        sns.despine(left=True, top=True, right=True)
        return
    
    def format_filename(self, filename=None, file_extension='pkl'):
        if filename is None:
            if self.title is None:
                title = 'Control State Log'
            else:
                title = self.title
            filename = ('./' + 
                        create_filename(self.t[0], self.t[-1], [self.title])
                        + '.' + file_extension)
            
        directory, f = os.path.split(filename)
        dirs = os.listdir(directory)
        if 'data' in dirs:
            if directory[-1] not in ['/','\\']:
                directory = directory + '/'
            filename = directory + 'data/' + f
        return filename
    
    def save(self, filename=None):
        filename = self.format_filename(filename, file_extension='pkl')
        save(self, filename)
        return
    
    
class AblationMap(object):
    
    def __init__(self, x, y, t, on_time, map_details=None, title=None):
        
        self.t = np.array(t)
        self.x = np.array(x)
        self.y = np.array(y)
        self.on_time = np.array(on_time)
        self.map_details = map_details
        self.title = title
        
    def __len__(self):
        return len(self.t)
    
    def __str__(self):
        table = pd.DataFrame(self.map_details).T
        table['MapID'] = table.index[0]
        table = table[['MapID','MaxTime','MinTime','TargetID','TargetName']]
        table = table.T
        string = str(table)
        string = string.split('\n')
        string = ['Ablation Mirror Positioning Map'] + string[1:]
        string = '\n'.join(string)
        return string
    
    def __repr__(self):
        return str(self)
    
    def plot_points(self, max_on_time=3600, threshold=10, 
                    extent=3.5, alpha_times=False, alpha=.2):
        range_x, range_y = self.get_range(extent=extent)
        
        if alpha_times:
            inds = np.where(
                np.logical_and(
                np.logical_not(np.isnan(self.on_time)),
                self.on_time > threshold))
            alphas = self.on_time[inds]/min([np.max(self.on_time[inds]), max_on_time])
            alphas[alphas > 1] = 1

            rgba_colors = np.zeros((len(alphas),4))
            # for red the first column needs to be one
            rgba_colors[:,0] = 1.0
            # the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas

            plt.scatter(self.x, self.y, color=rgba_colors)
        plt.scatter(self.x, self.y, 
                    c='cornflowerblue', alpha=alpha)
        
        plt.xlim(range_x)
        plt.ylim(range_y)
        
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        
        plt.xlabel('x (arb)')
        plt.ylabel('y (arb)')
        return

    def get_range(self, extent=3.5):
        weighted_std = lambda z, weights: np.sqrt(
                       np.average((z - np.average(z,weights=weights))**2, weights=weights)
                       )

        _get_range = lambda z, weights: [np.average(z, weights=weights) - weighted_std(z, weights) * extent, 
                               np.average(z, weights=weights) + weighted_std(z, weights) * extent]

        return _get_range(self.x, self.on_time), _get_range(self.y, self.on_time)

    def create_map(self, values, edge_size=.015, 
                   extent=3.5, mode='sum', background=0):

        get_len   = lambda r, es: int(np.ceil((r[1] - r[0]) / float(es)))
        get_array = lambda r, l: np.linspace(r[0], r[1], l)

        range_x, range_y = self.get_range(extent=extent)

        len_x = get_len(range_x, edge_size)
        len_y = get_len(range_y, edge_size)

        X = get_array(range_x, len_x)
        Y = get_array(range_y, len_y)

        Map = np.ones((len_x, len_y)) * background
        for i in range(len_x - 1):
            for j in range(len_y - 1):
                inds = np.where(np.logical_and(
                                np.logical_and(
                                self.x >= X[i], self.x < X[i + 1]),
                                np.logical_and(
                                self.y >= Y[j], self.y < Y[j + 1])))
                inds = inds[0]
                if len(inds) > 0:
                    if mode == 'sum':
                        Map[i, j] = np.sum(values[inds])
                    elif mode == 'max':
                        Map[i, j] = max(values[inds])
                    else:
                        raise IOError('invalid mode input.')
        return X,Y, Map.T

    def plot(self, edge_size=.015, extent=3.5):
        X, Y, Map = self.create_map(self.on_time, edge_size=edge_size, 
                                    mode='sum', background=0)
        im = plt.imshow(Map, norm=LogNorm(vmin=1, vmax=np.max(Map)), cmap=plt.cm.Blues,
                        extent=[min(X), max(X), max(Y), min(Y)],
                        interpolation='nearest')
        
        plt.xlabel('x (arb)')
        plt.ylabel('y (arb)')
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('ablation time (s)')
        return
    
    def plot_time_ago(self, edge_size=.015, extent=3.5):
        time_ago = np.array([t.value/float(10**9 * 24 * 3600) for t in self.t])
        time_ago = time_ago - min(time_ago)
        X, Y, Map = self.create_map(time_ago, edge_size=edge_size, 
                                    mode='max', background=np.nan)
        
        cmap = copy.deepcopy(plt.cm.Blues)
        masked_array = np.ma.array(Map, mask=np.isnan(Map))
        cmap.set_bad('lightgray',1.)
        
        im = plt.imshow(masked_array, cmap=cmap,
                        extent=[min(X), max(X), max(Y), min(Y)],
                        interpolation='nearest')
        
        plt.xlabel('x (arb)')
        plt.ylabel('y (arb)')
        
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('time from start (days)')
        return
    
    def print_report(self, max_on_time=3600, 
                     threshold=10, extent=3.5, 
                     edge_size=.005, save=True, filename=None):
        plt.subplot(2,2,1, aspect='equal')
        self.plot(edge_size=edge_size, extent=extent)
        plt.subplot(2,2,2, aspect='equal')
        self.plot_points(max_on_time=max_on_time, threshold=threshold, extent=extent)
        plt.subplot(2,2,3, aspect='equal')
        self.plot_time_ago(edge_size=edge_size, extent=extent)
        plt.subplot(2,2,4)
        plt.text(0,.5,str(self), va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        if save:
            filename = self.format_filename(filename, file_extension='pdf')
            plt.savefig(filename)
        return
    
    def format_filename(self, filename=None, file_extension='pkl'):
        if filename is None:
            if self.title is None:
                title = 'Ablation Mirror Map'
            else:
                title = self.title
            filename = ('./' + 
                        create_filename(self.t[0], self.t[-1], [title])
                        + '.' + file_extension)
            
        directory, f = os.path.split(filename)
        dirs = os.listdir(directory)
        if 'data' in dirs:
            if directory[-1] not in ['/','\\']:
                directory = directory + '/'
            filename = directory + 'data/' + f
        return filename
    
    def save(self, filename=None):
        filename = self.format_filename(filename, file_extension='pkl')
        save(self, filename)
        return
    
class DatabaseAccess(object):
    
    def __init__(self, **kwargs):
        self.open(**kwargs)
        
    def __str__(self):
        return 'database connection to ' + str(self.connections.keys())
    
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.connections.keys())
    
    def open(self, server='25.70.187.150',
                   databases=['LoggingConfigSQL','LoggingLogData', 'master'],
                   user_id='sa',
                   password=None):
        
        if password is None:
            password = getpass('please enter the database password: ')
        
        #initialize a dictionary of connections:
        connections={}
        #and initialize a dictionary of cursors:
        cursors={}

        #create the connections:
        for database in databases:
            connection_string = ('DRIVER={SQL Server};SERVER='+server
                                 + ';DATABASE=' + database 
                                 + ';UID=' + user_id
                                 + ';PWD=' + password)

            conn = pyodbc.connect(connection_string)
            connections[database]=conn
            cursor = conn.cursor()
            cursors[database]=cursor
            print(database + ' connection open')

        self.connections = connections
        self.cursors     = cursors
        
    def close(self):
        #close the connections 
        for key in self.connections.keys():
            try:
                self.connections[key].close()
            except:
                pass
        print('database connection closed')
        
    def restart(self, **kwargs):
        self.close()
        self.open(**kwargs)
        return
        
    def logging_channels(self, update=False):
        if not(update):
            try:
                return self._logging_channels
            except:
                pass
        else:
            command=("""
            SELECT LogChannelID,LogChannelName 
            FROM LogChannels
            """)
            self.cursors['LoggingConfigSQL'].execute(command)
            rows = self.cursors['LoggingConfigSQL'].fetchall()
            logchannels={row.LogChannelName:row.LogChannelID for row in rows}
            self._logging_channels = logchannels
            return self._logging_channels
        
    def get_table(self, table, database='LoggingConfigSQL', index=None):
        cursor = self.cursors[database]
        # first get the column names
        cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '""" + table + "'")

        rows = cursor.fetchall()
        columns = [row[0] for row in rows]

        # then get the table:
        cursor.execute("""
        SELECT *
        FROM """ + table)

        rows = cursor.fetchall()
        table = {columns[i]: [row[i] for row in rows] for i in range(len(columns))}
        table = pd.DataFrame(table)
        if index is not None:
            table = table.set_index(index)
        return table
    
    def sql_command(self, command, database=None, output=True, index=None, commit=False):
        if database is None:
            database = self.cursors.keys()[0]
        cursor = self.cursors[database]
        cursor.execute(command)
        if commit:
            self.connections[database].commit()
        if output:
            rows = cursor.fetchall()
            try:
                row = rows[0]
                names = [item[0] for item in row.cursor_description]
                output = {names[i]: [row[i] for row in rows] for i in range(len(names))}
                output = pd.DataFrame(output)
                if index is not None:
                    output = output.set_index(index)
                return output
            except:
                raise IOError('error on outputting data')
        else:
            return None
    
    def logging_channels(self, update=False):
        if not(update):
            try:
                return self._logging_channels
            except:
                pass
        
        table = self.get_table('LogChannelDetails', index='LogChannelName')
        self._logging_channels = table
        return self._logging_channels
    
    def logging_controls(self, update=False):
        if not(update):
            try:
                return self._logging_controls
            except:
                pass
        table = self.get_table('LogControlDetails', index='LogControlName')
        self._logging_controls = table
        return table
    
    def logging_groups(self, update=False):
        if not(update):
            try:
                return self._logging_groups
            except:
                pass
        table = self.get_table('LogChannelGroups', index='LogChannelGroupName')
        self._logging_groups = table
        return self._logging_groups
    
    def control_states(self, update=False):
        if not(update):
            try:
                return self._control_states
            except:
                pass
        table = self.get_table('LogControlStates', index='LogControlStateName')
        self._control_states = table
        return table
    
    def control_name(self, control_id):
        table = self.logging_controls()
        return table[table['LogControlID']==control_id].index[0]
    
    def control_id(self, control_name):
        table = self.logging_controls()
        return table.T[control_name]['LogControlID']
    
    def channel_properties(self, channel_name):
        table = self.logging_channels().T[channel_name]
        table['LogChannelName'] = channel_name
        return table
    
    def control_properties(self, control_name):
        table = self.logging_controls().T[control_name]
        table['LogControlName'] = control_name
        return table
    
    def class_controls(self, class_name):
        table = self.logging_controls()
        return table[table['LogChannelClassName'] == class_name]
    
    def group_properties(self, group_name):
        table = self.logging_groups().T[group_name]
        table['LogChannelGroupName'] = group_name
        return table
    
    def group_channels(self, group_name):
        table = self.logging_channels()
        return table[table['LogChannelGroupName'] == group_name]
    
    def get_data(self, channel=None, group=None, duration='1 hour',
                 start_time=None, end_time=None, name=None):
        # do some input checking to allow for multiple channels
        # and multiple groups.
        
        channels = []
        if channel is not None:
            try:
                channel[0]
                channels = channels + channel
            except:
                channels.append(channel)
                if group is None:
                    name = channel
     
        if group is not None:
            if isinstance(group, str):
                if name is None:
                    name = group
                group = [group]

            for g in group:
                group_channels = list(self.group_channels(g).index)
                channels = channels + group_channels
        
        time_series = []
        for channel in channels:
            ts = self.get_time_series(channel, duration=duration,
                                      start_time=start_time, end_time=end_time)
            time_series.append(ts)
        return TimeSeriesArray(time_series, name=name)
        
    
    def get_time_series(self, channel_name=None, duration='1 hour', 
                           start_time=None, end_time=None):
        """
        this method performs a database query to obtain data from
        the database and returns a TimeSeries object.
        """
        start_time, end_time = get_time_range(duration=duration, 
                                              start_time=start_time, 
                                              end_time=end_time)
        
        time_format = '%Y-%m-%dT%X'
        
        start_time_string = start_time.strftime(time_format)
        end_time_string   = end_time.strftime(time_format)
        
        channel    = self.channel_properties(channel_name)

        channel_id = channel['LogChannelID']
        
        command=(
             """SELECT TimeStamp, Data 
                FROM LogChannelData
                WHERE LogChannelID = """ + str(channel_id) +
               " AND TimeStamp>='" + start_time_string + "'" +
               " AND TimeStamp<='" + end_time_string + "'" +
               " ORDER BY TimeStamp;")
        
        cursor = self.cursors['LoggingLogData']
        cursor.execute(command)
        rows = cursor.fetchall()
        
        timestamps = [row.TimeStamp for row in rows]
        data       = [row.Data for row in rows]
        
        time_series = TimeSeries(t=timestamps, x=data, channel=channel)
        
        return time_series
    
    def get_control_data(self,
                         control=None, group=None,
                         start_time=None, end_time=None,
                         duration='1 hour', title=None):

        start_time, end_time = get_time_range(duration=duration, 
                                              start_time=start_time, 
                                              end_time=end_time)

        time_format = '%Y-%m-%dT%X'

        start_time_string = start_time.strftime(time_format)
        end_time_string   = end_time.strftime(time_format)

        if control is None and group is None:
            if title is None:
                title = 'All Controls'
        
        control_ids = []

        if control is not None:
            if isinstance(control, (str, int)):
                if (title is None and isinstance(group, str)) and (group is None):
                    title = control
                control = [control]

            for c in control:
                if isinstance(c, int):
                    control_ids.append(c)
                elif isinstance(c, str):
                    control_ids.append(int(self.control_id(c)))
                else:
                    raise IOError('invalid input for channel')

        if group is not None:
            if isinstance(group,(str, int)):
                if (title is None and isinstance(group, str)) and (control is None):
                    title = group
                group = [group]

            for g in group:
                if isinstance(g, str):
                    control_ids   = control_ids + list(self.class_controls(g)['LogControlID'])
                else:
                    raise IOError('group must correspond to LogChannelClassName')


        command = ("""
            SELECT TimeStamp, LogControlID, LogControlStateID
            FROM LogControlStateData
            WHERE TimeStamp >= '""" + start_time_string + """'
            AND TimeStamp <= '""" + end_time_string + """'
            """)

        if len(control_ids) > 0:
            control_string = control_string = '(' + ', '.join(["'" + str(c) + "'" for c in control_ids]) + ')'
            command = command + "AND LogControlID IN " + control_string + '\n'

        cursor = self.cursors['LoggingLogData']
        cursor.execute(command)
        rows = cursor.fetchall()
        TimeStamp    = [row.TimeStamp for row in rows]
        LogControlID = [row.LogControlID for row in rows]
        LogControlStateID = [row.LogControlStateID for row in rows]
        return TimeSeriesStates(TimeStamp, LogControlID, LogControlStateID,
                                self.logging_controls(), self.control_states(), title=title)
    
    def get_list_of_ablation_maps(self, update=False):
        if not(update):
            try:
                return self._list_of_ablation_maps
            except:
                pass
        
        table = self.sql_command("""
            SELECT a.MapID, a.MinTime, a.MaxTime, TargetID=b.LogControlStateID, 
                   TargetName=b.LogControlStateName, TargetDetails = CAST(b.LogControlStateInfo AS VARCHAR(MAX))
            FROM
            (
            SELECT MapID, MinTime=MIN(TimeStamp), MaxTime=MAX(TimeStamp), TargetID=MAX(TargetID)
            FROM AblationMirrorPositioning
            GROUP BY MapID
            ) a
            INNER JOIN
            LoggingConfigSQL.dbo.LogControlStates b 
            ON a.TargetID = b.LogControlStateID
        """, database='LoggingLogData', index='MapID')
        self._list_of_ablation_maps = table
        return table
    
    def get_ablation_map(self, map_id=None):
        ablation_maps = self.get_list_of_ablation_maps()
        if map_id is None:
            print ablation_maps
            map_id = int(input('please select a MapID'))
            
        map_details = ablation_maps.T[map_id]
        
        data = self.sql_command("""
            SELECT XPosition, YPosition, TimeStamp, OnTimeSeconds
            FROM AblationMirrorPositioning
            WHERE MapID = """ + str(map_id) + """
            ORDER BY TimeStamp DESC
            """,
            database="LoggingLogData")
        data = data.dropna()
        
        return AblationMap(list(data.XPosition),
                           list(data.YPosition),
                           list(data.TimeStamp),
                           list(data.OnTimeSeconds),
                           map_details)
            
    