# This is a script that was used to insert the
# previously text-file-logged ablation data into
# the database:
import os
import pandas as ps
from database_access import DatabaseAccess
import numpy as np
import re
from datetime import datetime
from time import strftime, strptime

def get_files(directory):
    directory = os.listdir(directory)
    t     = []
    index = []
    size  = []
    files = []

    for f in directory:
        l = re.findall(r'^ThO Target ([\d]*) ([\d]*) ([\d]*)_([\d]*).txt', f)
        if len(l) > 0:
            l = tuple([int(i) for i in l[0]])
            t.append(datetime(l[2], l[0], l[1]))
            index.append(l[3])
            size.append(os.path.getsize('./Data Directory/' + f))
            files.append(f)

    t     = np.array(t)
    index = np.array(index)

    size = np.array(size)
    size_threshold = 600
    inds = np.where(size > size_threshold)

    t = t[inds]
    index = index[inds]
    files = [files[i] for i in inds[0]]

    inds = np.argsort(t)
    t = t[inds]
    index = index[inds]
    files = [files[i] for i in inds]
    return files

def create_table_from_file(f, threshold=0):
    table = pd.read_csv('./Data Directory/' + f, sep='\t')
    timestamp = [pd.to_datetime(date + 'T' + time, format='%m/%d/%YT%I:%M %p') for date, time in zip(table.Date, table.Time)]
    target_name = table.columns[-1]
    table = table.rename(columns={target_name: 'OnTimeSeconds',
                                  'X position': 'XPosition',
                                  'Y position': 'YPosition'})
    table['TimeStamp'] = timestamp
    on_time = table['OnTimeSeconds']
    on_time = (np.array(on_time[1:]) - np.array(on_time[:-1]))/50.0
    on_time = [np.nan] + list(on_time)

    map_id = db.sql_command('SELECT MAX(MapID) FROM AblationMirrorPositioning', 
                            database="LoggingLogData")
    map_id = int(map_id.T[0]) + 1

    target_id = 76
    table['OnTimeSeconds'] = on_time
    table = table[table['OnTimeSeconds'] > threshold]
    table['MapID'] = [map_id] * len(table)
    table['TargetID'] = [target_id] * len(table)
    table = table.dropna()
    del table['Date']
    del table['Time']
    return table

def create_sql_command(table, values_per_command=500):
    columns = '(' + ', '.join(table.columns) + ')'
    value = lambda index: ('(' + str(table['MapID'][index]) + ', '
                               + str(table['TargetID'][index]) + ', '
                               + str(table['XPosition'][index]) + ', '
                               + str(table['YPosition'][index]) + ', '
                               + str(table['OnTimeSeconds'][index]) + ', '
                               + "'" + table['TimeStamp'][index].isoformat() + "')")
    
    values = map(value, table.index)
    while len(values) > 0:
        if len(values) >= values_per_command:
            subset = values[:values_per_command]
            values = values[values_per_command:]
        else:
            subset = values
            values = []
            
    command = ("""INSERT INTO AblationMirrorPositioning
                  (MapID, TargetID, XPosition, YPosition, OnTimeSeconds, TimeStamp)
                  VALUES """ + ', '.join(subset) + ';')
    return command


db = DatabaseAccess()

files = get_files('./Data Directory/')
for f in files:
    table = create_table_from_file(f)
    if len(table) > 0:
        command = create_sql_command(table)
        db.sql_command(command, database="LoggingLogData",output=False, commit=True)
        