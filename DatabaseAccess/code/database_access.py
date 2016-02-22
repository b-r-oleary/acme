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
        print files
        index = input("""input an integer corresponding to the file that you would like to load: """)
        filename = files[index]
    with open(filename, 'rb') as f:
        output = pickle.load(f)
    print 'loaded: ' + filename
    return output
        
def save(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print 'saved: ' + filename
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

def create_filename(start_time, end_time, channels, fmt='%Y-%m-%dT%H-%M'):
    string = '_'.join([start_time.strftime(fmt), 
                       end_time.strftime(fmt)]
                       + channels)
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
        print 'saved: ' + filename
        return
    
class DatabaseAccess(object):
    
    def __init__(self, server='25.70.187.150',
                       databases=['LoggingConfigSQL','LoggingLogData'],
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
            print database + ' connection open'

        self.connections = connections
        self.cursors     = cursors
        
    def __str__(self):
        return 'database connection to ' + str(self.connections.keys())
    
    def __repr__(self):
        return str(self)
    
    def __len__(self):
        return len(self.connections.keys())
        
    def close(self):
        #close the connections 
        for key in self.connections.keys():
            try:
                self.connections[key].close()
            except:
                pass
        print 'database connection closed'
        
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
    
    def logging_channels(self, update = False):
        if not(update):
            try:
                return self._logging_channels
            except:
                pass
        
        table = self.get_table('LogChannelDetails', index='LogChannelName')
        self._logging_channels = table
        return self._logging_channels
    
    def logging_groups(self, update=False):
        if not(update):
            try:
                return self._logging_groups
            except:
                pass
        table = self.get_table('LogChannelGroups', index='LogChannelGroupName')
        self._logging_groups = table
        return self._logging_groups
    
    def channel_properties(self, channel_name):
        table = self.logging_channels().T[channel_name]
        table['LogChannelName'] = channel_name
        return table
    
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
        data       =[row.Data for row in rows]
        
        time_series = TimeSeries(t=timestamps, x=data, channel=channel)
        
        return time_series