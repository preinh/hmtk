# -*- coding: utf-8 -*-
# LICENSE
#
# Copyright (c) 2010-2013, GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
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
Module :mod: hmtk.seismicity.smoothing.smoothed_seismicity implements the
:class: hmtk.seismicity.smoothing.smoothed_seismicity.SmoothedSeismicity,
a general class for implementing seismicity smoothing algorithms
'''
import csv
import numpy as np

from math import log

from hmtk.seismicity.smoothing import utils

from hmtk.seismicity.smoothing.kernels.woo_1996 \
    import IsotropicGaussianWoo, Frankel_1995

from hmtk.registry import CatalogueFunctionRegistry

import matplotlib.pyplot as plt

from hmtk.seismicity.utils import haversine

class SmoothedSeismicityWoo(object):
    '''
    Class to implement an analysis of Smoothed Seismicity, including the
    grid counting of data and the smoothing.

    :param np.ndarray grid:
        Observed count in each cell [Long., Lat., Depth., Count]

    :param catalogue:
        Valid instance of the :class: hmtk.seismicity.catalogue.Catalogue

    :param bool use_3d:
        Decide if analysis is 2-D (False) or 3-D (True). If 3-D then distances
        will use hypocentral distance, otherwise epicentral distance

    :param float bval:
        b-value

    :param float beta:
        Beta value for exponential form (beta = bval * log(10.))

    :param np.ndarray data:
        Smoothed seismicity output

    :param dict grid_limits:
        Limits ot the grid used for defining the cells
    '''

    def __init__(self, grid_limits, use_3d=False, bvalue=None):
        '''
        Instatiate class with a set of grid limits
        :param grid_limits:
            It could be a float (in that case the grid is computed from the
            catalogue with the given spacing).

            Or an array of the form:
            [xmin, xmax, spcx, ymin, ymax, spcy, zmin, spcz]

        :param bool use_3d:
            Choose whether to use hypocentral distances for smoothing or only
            epicentral

        :param float bval:
            b-value for analysis
        '''
        self.grid = None
        self.catalogue = None
        self.use_3d = use_3d
        self.bval = bvalue
        if self.bval:
            self.beta = self.bval * log(10.)
        else:
            self.beta = None
        self.data = None

        self.grid_limits = grid_limits
        self.kernel = None


    def add_bandwith_values(self, min_magnitude=None, max_magnitude=None, magnitude_bin=0.5):

# rs(m) = 1.340e0.6m (Espagne, E) (9.9) 
# rs(m) = 0.048e1.55m (Norvege, N) (9.10)
       
        # mags, mean_minimum_pairwise_distance
        _data = self._get_bandwidth_data()         
        c, c_err, d, d_err = self._fit_bandwidth_data(_data)     # optimize H(m)
        self.c = c
        self.d = d
        plt.xkcd()
        H = lambda m, c, d: c * np.exp(d*m)
        m = self.catalogue.data['magnitude']
        #self.catalogue.data['bandwidth'] = H(m, c, d)
        plt.semilogy(m, H(m, c, d), color='#75B08A', label='Brazil [$h(m)=%.2fe^{%.2fm}$]'%(c,d))
        plt.semilogy(m, H(m, 1.34, 0.60), color='#F0E797', label='Spain [$h(m)=1.34e^{0.60m}$]')
        plt.semilogy(m, H(m, 0.05, 1.55), color='#FF9D84', label='Norway [$h(m)=0.05e^{1.55m}$]')
        plt.semilogy(_data['magnitude'], _data['distance'], linewidth=0, marker='o', color='#FF5460')
        plt.title("\gls{bsb2013} bandwidth function [woo1996]")
        plt.xlabel('$m$ [magnitude]')
        plt.ylabel('$h(m)$ [distance]')
        plt.legend(loc='lower right', fontsize='small')
        plt.show()

        return None



    def _get_bandwidth_data(self, magnitude_bin=0.5):

        # pegar catálogo corrigido pela completude

        # get data
        X = self.catalogue.data['magnitude']
        
        min_magnitude = self.config['min_magnitude'] if self.config['min_magnitude'] else min(X)
        max_magnitude = max(X)

        # divide bins catalog_bins
        bins = np.arange(min_magnitude, max_magnitude + magnitude_bin, magnitude_bin)

        h , m = [], []
        for b in bins:
            _i = np.logical_and(X > b, X < b + magnitude_bin)
            #print b
            #print X[_i]
            if len(X[_i]) > 0:
                # calculate distances on bin
                from hmtk.seismicity.utils import haversine
                d = haversine(self.catalogue.data['longitude'][_i], 
                              self.catalogue.data['latitude'][_i],
                              self.catalogue.data['longitude'][_i], 
                              self.catalogue.data['latitude'][_i])
                
                # média das distancias mínimas e centro do magnitude_bin
                m.append(b + magnitude_bin/ 2.)  
                h.append(np.sort(d)[:,1].mean()) 
        
        return {'distance' : h, 
                'magnitude': m }
            
        pass
    
    def _fit_bandwidth_data(self, data):
        from scipy import optimize 
        
        #powerlaw = lambda x, amp, index: amp * (x**index)
        powerlaw = lambda m, c, d: c * np.exp(m*d)
        
        #  y     = a * exp(m*d)
        #  ln(y) = ln(a) + b*m
        
        h = np.log(data['distance'])
        m = np.array(data['magnitude'])
        h_err = 0.10*np.ones(len(h))
        
        #print h, m
        
        # define our 'line' fitting function
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
        
        
        # fit exponential data
        p_init = [1.0, 1.0]
        out = optimize.leastsq(errfunc, p_init, args=(m, h, h_err), full_output=1)
        
        p_final = out[0]
        covar = out[1]
        #print p_final
        #print covar        
        
        d = p_final[1]
        c = np.exp(p_final[0])
        
        d_err = np.sqrt( covar[0][0] )
        c_err = np.sqrt( covar[1][1] ) * c
            
        return c, c_err, d, d_err

    def _seismicity_rate(self, m=None, r=None):
        
        
        
        k = Frankel_1995()
        #print 
        # sum kernel(m, r)  / completeness_time(m)
        pass 


#     def _grid2d(self, x1, x2):
#         return 

    def _create_grid(self, use3d=False):
        l = self.grid_limits

        dx = l['xspc']
        dy = l['yspc']
        dz = l['zspc']
        
        x = np.arange(l['xmin'] + dx/2., l['xmax'], dx) 
        y = np.arange(l['ymin'] + dy/2., l['ymax'], dy) 
        z = np.arange(l['zmin'] + dz/2., l['zmax'], dz)
        
#         print min(x), max(x)
#         print min(y), max(y)
#         print min(z), max(z)
        
        if not use3d:
            #spacement = [dx, dy]
            xx, yy= np.meshgrid(x, y)
            x, y = xx.flatten(), yy.flatten()
            cells = np.array(zip(x,y))
        else:
            #spacement = [dx, dy, dz]
            xx, yy, zz = np.meshgrid(x, y, z)
            x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
            cells = np.array(zip(x,y,z))
        
        return cells #, spacement
        
        
    def _get_observation_time(self, m, completeness_table, last_year):
        ct = completeness_table

        i = ct[:,1] >= m    # all values after desired mag...
        _mt = min(ct[i,1])  # min of these values
        i = np.where(ct[:,1] == _mt) # index of

        observation_time = last_year - ct[i, 0][0][0] # corresponding year

        return observation_time
    

    def run_analysis(self, catalogue, config, completeness_table=None, smoothing_kernel=IsotropicGaussianWoo):
        '''
        Runs an analysis of smoothed seismicity in the manner
        originally implemented by Frankel (1995)
 
        :param catalogue:
            Instance of the hmtk.seismicity.catalogue.Catalogue class
 
        :param dict config:
            Configuration settings of the algorithm:
            * 'Length_Limit' - Maximum number of bandwidths for use in
                               smoothing (Float)
            * 'BandWidth' - Bandwidth (km) of the Smoothing Kernel (Float)
            * 'increment' - Output incremental (True) or cumulative a-value
                            (False)
 
        :param np.ndarray completeness_table:
            Completeness of the catalogue assuming evenly spaced magnitudes
            from most recent bin to oldest bin [year, magnitude]
 
        :param smoothing_kernel:
            Smoothing kernel as instance of :class:
                hmtk.seismicity.smoothing.kernels.base.BaseSmoothingKernel
 
        :returns:
            Full smoothed seismicity data as np.ndarray, of the form
            [Longitude, Latitude, Depth, Observed, Smoothed]
        '''
        use3d = config['use3d']
        nh = config['bandwidth_h_limit']
        mag_bin = config['magnitude_bin']
        
        self.catalogue = catalogue
        self.completeness_table = completeness_table
        self.config = config
        self.add_bandwith_values()
 
        year = catalogue.end_year
        
        # get magnitude-completeness table for each magnitude
        ct, dm = utils.get_even_magnitude_completeness(completeness_table, 
                                                       catalogue, 
                                                       magnitude_increment=mag_bin)
        
        # create grid
        grid = self._create_grid(use3d=use3d)

        x = grid[:,0]
        y = grid[:,1]

        # distances from grid to epicenters
        distances = haversine(x, 
                              y, 
                              catalogue.data['longitude'],
                              catalogue.data['latitude'])
                
        M = self.catalogue.data['magnitude']
        
        min_magnitude = self.config['min_magnitude'] if self.config['min_magnitude'] else min(M)
        max_magnitude = max(M)
        
        # divide bins catalog_bins
        mags = np.arange(min_magnitude, max_magnitude + dm, dm)
        # get observation time for each magnitude
        time = year - ct[:,0] 

# TODO refactor
        # not properly a kernel but some auxiliar function
        # C and D parameters was calculated by [add_bandwith_values]
        k = Frankel_1995(self.c, self.d)
        
        # use it to max influence distance for each magnitude 
        r_max = k.H(mags) * nh
        _d = []
        self.data = []
        # for each grid point
        for i, c in enumerate(zip(x,y)):
            # get distances from cell to each epicenter
            r = distances[i,:]
            rates=[]
            # for each magnitude bin 
            for (m, t, limit) in zip(mags + dm/2, time, r_max):
                
                # indexes for filtered magnitude
                a = np.logical_and( M >= m - dm/2., M < m + dm/2.)
                # and influence kernel zone
                b = np.logical_and(r <= limit, r > 0 )
                _i = np.logical_and(a, b)
                
#                 if m <= 4.25:
#                     print r[_i]
                
                # TODO HERE could be the place to get not
                # the value @ cell's center but the
                # integral over cell if the epic are in outside of cell
                # or get value @ cell-center when it is in the cell.
                # check with helmstetter code or using woo code.
                
                # get kernel values
                _k = k.kernel(m, r[_i]) / t 
                
                # and sum
                rates.append(_k.sum())
            
            # lon, lat, depth, m0, dm, ...
            self.data.append([c[0], c[1], 0, min_magnitude, dm, rates])
 
        from matplotlib import pylab as plt
        plt.hist(_d, bins=30)
        plt.show()
        
        return self.data


    def write_to_csv(self, filename):
        '''
        Exports to simple csv
        :param str filename:
            Path to file for export
        '''
        fid = open(filename, 'wt')
        # Create header list
        header_info = ['Longitude', 'Latitude', 'Depth', 'm_min', 'm_bin', 'Rates']
        writer = csv.DictWriter(fid, fieldnames=header_info)
        headers = dict((name0, name0) for name0 in header_info)
        # Write to file
        writer.writerow(headers)
        for row in self.data:
            row_dict = {'Longitude': '%.5f' % row[0],
                        'Latitude': '%.5f' % row[1],
                        'Depth': '%.3f' % row[2],
                        'm_min': '%.2f' % row[3],
                        'm_bin': '%.2f' % row[4],
                        'Rates': "%s" % str(row[5])[1:-1].replace(",", "")}
            writer.writerow(row_dict)
        fid.close()

    def write_rates(self, filename):
        '''
        Exports to simple csv
        :param str filename:
            Path to file for export
        '''
        fid = open(filename, 'wt')
        # Create header list
        header_info = ['Longitude', 'Latitude', 'Depth', 'm_min', 'm_bin', 'a0_value']
        writer = csv.DictWriter(fid, fieldnames=header_info)
        headers = dict((name0, name0) for name0 in header_info)
        # Write to file
        writer.writerow(headers)
        for row in self.data:
            # a = log10(alpha*m_min) + b*m_min 
            _a0 = np.log10(sum(row[5])*row[3]) + self.bval*row[3]
            row_dict = {'Longitude': '%.5f' % row[0],
                        'Latitude': '%.5f' % row[1],
                        'Depth': '%.3f' % row[2],
                        'm_min': '%.2f' % row[3],
                        'm_bin': '%.2f' % row[4],
                        'a0_value': "%e" % _a0,
                        }
            writer.writerow(row_dict)
        fid.close()




SMOOTHED_SEISMICITY_METHODS = CatalogueFunctionRegistry()

@SMOOTHED_SEISMICITY_METHODS.add(
    "run",
    completeness=True,
    b_value=np.float,
    use_3d=bool,
    grid_limits=utils.Grid,
    Length_Limit=np.float,
    BandWidth=np.float,
    increment=bool)

class WooMethod(object):
    def run(self, catalogue, config, completeness=None):
        sw = SmoothedSeismicityWoo(config['grid_limits'],
                                   config['use_3d'],
                                   config['b_value'])
        return sw.run_analysis(
            catalogue, config, completeness_table=completeness)
