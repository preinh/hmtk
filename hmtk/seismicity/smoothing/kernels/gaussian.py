# -*- coding: utf-8 -*-

import numpy as np

from hmtk.seismicity.smoothing.kernels.base import BaseSimpleKernel


class Kernel(BaseSimpleKernel):
    
    def _normalization_factor(self, d):
        return np.sqrt( 2 * np.pi ) * d 

    def value(self, r, p):
        return np.exp( -r*r / (2.0 * p*p) ) \
                * self._normalization_factor(p)

