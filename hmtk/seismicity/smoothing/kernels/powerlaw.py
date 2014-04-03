# -*- coding: utf-8 -*-

import numpy as np

def _normalization_factor(d):
    return 1

def powerlaw_kernel(r, p):
    return ( r**2 + p[0]**2 )**(-1.5)*_normalization_factor(p[0])

