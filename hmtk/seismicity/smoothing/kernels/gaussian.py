# -*- coding: utf-8 -*-

import numpy as np

from hmtk.seismicity.smoothing.kernels.base import BaseSimpleKernel


class Kernel(BaseSimpleKernel):
    
    def _normalization_factor(self, d):
        return np.sqrt( 2 * np.pi ) * d 

    def value(self, r, p):
        return np.exp( -r**2 / (2.0 * p[0]**2) ) \
                * self._normalization_factor(p[0])

