# this script includes some classes that are useful for scientific research:
#
# 1. References - keeps track of a paper, its authors, and its link
# 2. Un - keeps track of and propagates gaussian uncertainties from measurements
# 3. Data - includes data with uncertainties, notes regarding the conditions
#            under which the data was taken and the quality, and references if applicable
#

import numpy as np
from uncertainties import ufloat

class Reference(object):

    def __init__(self, bibtex=None, authors = None, year = None):
        self.bibtex = bibtex
        self.authors = authors
        self.year = year


class Data(Un):

    def __init__(self,note="",reference=""):
        self.note = note
        self.reference = reference

