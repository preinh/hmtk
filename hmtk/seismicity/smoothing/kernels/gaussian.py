# -*- coding: utf-8 -*-

import numpy as np

from hmtk.seismicity.smoothing.kernels.base import BaseSimpleKernel


class Kernel(BaseSimpleKernel):
    
    def _normalization_factor(self, d):
        return np.sqrt( 2 * np.pi ) * d

    def value(self, r, d):
        nf = self._normalization_factor(d)
        return np.exp( -r*r / (2.0 * d*d) ) * nf

if __name__ == '__main__':
    k = Kernel()
    r = np.array([[2, 2], [3, 3]])
    p = np.array([1, 1])
    print r
    print r*r
    print k.value(r, p)