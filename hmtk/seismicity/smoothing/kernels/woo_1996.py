# -*- coding: utf-8 -*-
# LICENSE
#
# Copyright (c) 2014, M.Pirchiner
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>
#
# DISCLAIMER
#
# The software Hazard Modeller's Toolkit (hmtk) provided herein
# is released as a prototype implementation on behalf of
# scientists and engineers working within the GEM Foundation (Global
# Earthquake Model).
#
# It is distributed for the purpose of open collaboration and in the
# hope that it will be useful to the scientific, engineering, disaster
# risk and software design communities.
#
# The software is NOT distributed as part of GEM’s OpenQuake suite
# (http://www.globalquakemodel.org/openquake) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM’s OpenQuake software suite.
#
# Feedback and contribution to the software is welcome, and can be
# directed to the hazard scientific staff of the GEM Model Facility
# (hazard@globalquakemodel.org).
#
# The Hazard Modeller's Toolkit (hmtk) is therefore distributed WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# The GEM Foundation, and the authors of the software, assume no
# liability for use of the software.
# -*- coding: utf-8 -*-
'''
Module :mod: hmtk.seismicity.smoothing.kernels.woo_1996 imports
hmtk.seismicity.smoothing.kernels.woo_1996.Woo the IsotropicGaussianWoo
kernel with variable global bandwith H(M) = a*exp(b*M) as described 
by Gordon Woo (1996)

Woo, G. (1996) Kernel Estimation Methods for Seismic Hazard Area Source Modeling. 
Bulletin of Seismological Society of America, 86(2) 353-362

'''

import numpy as np
from hmtk.seismicity.utils import haversine
from hmtk.seismicity.smoothing.kernels.base import BaseSmoothingKernel

class Frankel_1995(BaseSmoothingKernel):
    
    def __init__(self, c, d, D=1.75):
        self.D = D
        self.H = lambda m : c*np.exp(m*d)

    def smooth_data(self):
        pass

    def kernel(self, M, r):
        h = self.H(M)
        #if r > h: return 0

        # K(M, x) = [D/2nh(M)] {h(M)/r}^2-D
        k = ( self.D / (2*np.pi*h)) * (h / r)**(2 - self.D)
        #print k
        return k
    
    
    def _seismic_rate(self, catalogue, magnitude):
        # sum 
        rate_locations = np.zeros(self.grid_locations.shape)
        
        event_locations = np.array([catalogue.data['longitude'], 
                                     catalogue.data['latitude']])
        
        h = self.H(magnitude)
        
        d = lambda x, y : np.sqrt( y^2 + x^2 )
        for x in self.grid_locations:
            
            r = [ d for d in d(x, event_locations) if d <= h ] 
        
        pass


class IsotropicGaussianWoo(BaseSmoothingKernel):
    '''
    Applies a simple isotropic Gaussian smoothing using an Isotropic Gaussian
    Kernel - taken from Woo (1996) approach
    '''

    def smooth_data(self, data, config, is_3d=False):
        '''
        Applies the smoothing kernel to the data
        :param np.ndarray data:
            Raw earthquake count in the form [Longitude, Latitude, Depth,
                Count]
        :param dict config:
            Configuration parameters must contain:
            * BandWidth: The bandwidth of the kernel (in km) (float)
            * Length_Limit: Maximum number of standard deviations

        :returns:
            * smoothed_value: np.ndarray vector of smoothed values
            * Total (summed) rate of the original values
            * Total (summed) rate of the smoothed values
        '''
        max_dist = config['Length_Limit'] * config['BandWidth']
        smoothed_value = np.zeros(len(data), dtype=float)

        for iloc in range(0, len(data)):
            dist_val = haversine(data[:, 1], data[:, 0],
                                 data[iloc, 1], data[iloc, 0])
            if is_3d:
                dist_val = np.sqrt(dist_val.flatten() ** 2.0 +
                                   (data[:, 2] - data[iloc, 2]) ** 2.0)

            id0 = np.where(dist_val <= max_dist)[0]
            w_val = (np.exp(-(dist_val[id0] ** 2.0) /
                            (config['BandWidth'] ** 2.))).flatten()
            smoothed_value[iloc] = np.sum(w_val * data[id0, 3]) / np.sum(w_val)

        return smoothed_value, np.sum(data[:, -1]), np.sum(smoothed_value)
