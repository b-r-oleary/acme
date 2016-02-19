# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:37:12 2016

@author: Brendon
"""

import sys
sys.path.insert(1,'./')

import numpy as np
from statfunctions import *

def test_damage_control():
    """
    test the function damage_control
    """
    # these are some of the types of possible inputs
    types = [-np.inf, -1, 0, 1, np.inf, np.NaN]
    
    # create numpy arrays with all combinations of these inputs
    # between the means and uncertainties:
    a  = np.array(types * len(types))
    da = np.array([types[i//len(types)] for i in range(len(types)**2)])
    
    damage_control(a, da)
    
    # these is the expected output:
    b  = np.array([0.] * 19 + [-1., 0., +1] + [0.] * 14)
    db = np.array([np.inf] * 19 + [1.] * 3 + [np.inf] * 14)
    
    # return True if it passes the test
    result = (np.all(a == b) and np.all(da == db))
    return result

def test_weighted_mean():
    """
    test my weighted_mean function
    """
    a  = np.array([np.inf, np.NaN, -1, 0, 1])
    da = np.array([1, 2, 3, 4, np.inf])
    
    y, dy = weighted_mean(a, da)
    
    return np.isclose(y,-0.64) and np.isclose(dy,2.4)
    
print(test_damage_control())
print(test_weighted_mean())