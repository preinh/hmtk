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

# integration method
from scipy.integrate import dblquad


def gaussian_2d(x, y, s):
    normalization_factor = 1./2./np.pi/s/s
    exponent = (-1./2.)*(x*x/s/s + y*y/s/s)
    return normalization_factor*np.exp(exponent)


def powerlaw_2d(x, y, s, decaiment = 1.5):
    normalization_factor = s/2./np.pi
    denominator = (x*x + y*y + s*s)**(decaiment)
    #return normalization_factor*(1.0/denominator)
    return normalization_factor/denominator


def infinite_2d(x, y, h, a = 1.5):
    h2 = h*h
    return ((a - 1.)/np.pi)*h2*((1 + x*x + y*y)/h2)**(-a)
# 
#     # 
#       rad = np.pi/180.0
# C H is the bandwidth of the kernel
#       H = BWIDA*EXP(BWIDB*GMAG)
#       CN = PI/(PL-1.0)
#       H2 =H*H
#       CONST = 1.0/(CN*H2)
# C CANG is the normalization for the anisotropic factor ANISO
#       CANG = 2.0*PI/( 2.0*PI + DL(IQ)*PI )
#       R2 = X*X + Y*Y
#       IF (R2.LT.0.01) X = 0.01
#       ANG1 = ATAN2(Y,X)
#       ANG2 = TH(IQ)*RAD
#       IF (ANG2.GT.PI) ANG2 = -2.0*PI + ANG2
# C ANG is the orientation of the site with respect to the lineament
#       ANG = ANG1 - ANG2
#       ANISO = (1.0+DL(IQ)*COS(ANG)*COS(ANG)) * CANG   
#       ATTEN = (1.0 + (R2/H2))**(-PL)
#       AKER = CONST*ANISO*ATTEN

    
class spatial_kernel(object):
    
    def __init__(self, 
                 x0, y0,
                 bandwidth,
                 kernel_type = 'gaussian_2d',
                 parameters = None):
        self.x0 = x0
        self.y0 = y0
        
        self.h = bandwidth

        if kernel_type == 'gaussian_2d':
            f = gaussian_2d
        elif kernel_type == 'powerlaw_2d':
            f = powerlaw_2d
        elif kernel_type == 'infinite_2d':
            f = infinite_2d
                        
        self.f = f


    def integrate(self, x_min, x_max,
                        y_min, y_max,
                        absolute_epsilon = 1e-10):

        x0 = x_min - self.x0
        xf = x_max - self.x0

        # need to be calable of 
        y0 = lambda x: y_min - self.y0
        yf = lambda x: y_max - self.y0

        value, error = dblquad(self.f, 
                               x0, xf,
                               y0, yf,
                               args=(self.h,), 
                               epsabs=absolute_epsilon)
        return value 
        


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
