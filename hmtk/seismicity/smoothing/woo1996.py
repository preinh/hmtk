# -*- coding: utf-8 -*-
'''
Implement Woo1996 smoothing method
'''

from copy import deepcopy

class SmoothedSeismicity(object):

    def __init__(self, catalogue, config):
        self.catalogue = deepcopy(catalogue)
        self.config = config
     
     
     
     
