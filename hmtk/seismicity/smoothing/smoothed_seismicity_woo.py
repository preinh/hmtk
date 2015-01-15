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

from copy import deepcopy
from math import log

from hmtk.seismicity.smoothing import utils

from hmtk.seismicity.smoothing.kernels.woo_1996 \
    import IsotropicGaussianWoo, Frankel_1995, spatial_kernel

from hmtk.registry import CatalogueFunctionRegistry

from hmtk.seismicity.utils import haversine

from multiprocessing import Pool


def _unwraper_smoothed_rate(*arg, **kwarg):
    #print "unrwap", arg, kwarg
    return arg[0]._smooth_event_rate(*arg[1:], **kwarg)


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
        self.rates = None
        self.pense = None

    def optimize_bandwith_values(self, min_magnitude=None, max_magnitude=None, magnitude_bin=0.5):

        H = lambda m, c, d: c * np.exp(d*m)

        
        # mags, mean_minimum_pairwise_distance
        _data = self._get_bandwidth_data(magnitude_bin = magnitude_bin) 
        # optimize H(m)
        c, c_err, d, d_err = self._fit_bandwidth_data(_data)     

        self.c = c
        self.d = d
        return None
    
#         m = self.catalogue.data['magnitude']
#         self.catalogue.data['bandwidth'] = H(m, c, d)
#         
#         return None

    def _get_bandwidth_data(self, magnitude_bin=0.5):

        # pegar catálogo corrigido pela completude

        # get data
        M = self.catalogue.data['magnitude']
        
        dmag = magnitude_bin/2.
        
        min_magnitude = self.config['min_magnitude']
        min_magnitude = min_magnitude if min_magnitude else min(M)
        max_magnitude = max(M)

        # divide bins catalog_bins
        magnitudes = np.arange(min_magnitude, 
                               max_magnitude + magnitude_bin, 
                               magnitude_bin)
        #print magnitudes

        h , m = [], []
        for mag in magnitudes:
            _i = np.logical_and(M  > mag - dmag, 
                                M <= mag + dmag)
            #print b
            #print X[_i]
            if len(M[_i]) > 1:
                # calculate distances on bin
                from hmtk.seismicity.utils import haversine
                d = haversine(self.catalogue.data['longitude'][_i], 
                              self.catalogue.data['latitude'][_i],
                              self.catalogue.data['longitude'][_i], 
                              self.catalogue.data['latitude'][_i])
                
                # mean nearest distance [km] into this magnitude_bin
                _h = np.sort(d)[:,1].mean()
                
                # accumulate
                m.append(mag)
                h.append(_h) 
        
        return {'distances' : h, 
                'magnitudes': m }
            
    
    def _fit_bandwidth_data(self, data):
        from scipy import optimize 
        
        #powerlaw = lambda x, amp, index: amp * (x**index)
        #powerlaw = lambda m, c, d: c * np.exp(m*d)
        
        #  y     = a * exp(m*d)
        #  ln(y) = ln(a) + b*m
        
        h = np.log(data['distances'])
        m = np.array(data['magnitudes'])
        default_error = 0.01
        h_err = default_error*np.ones(len(h))
        
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
        
        x = np.arange(l['xmin'] + dx/2., l['xmax'] + dx/2., dx) 
        y = np.arange(l['ymin'] + dy/2., l['ymax'] + dy/2., dy) 
        z = np.arange(l['zmin'] + dz/2., l['zmax'] + dz/2., dz)
        
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
        
    def _get_degre_km_factor(self, x0=0., y0=0.):
        ephy = np.array([ 0.00, 10.18, 16.20, 20.70, 24.48, 27.88,
                         31.00, 33.97, 36.80, 39.55, 42.25, 44.92,
                         47.62, 50.28, 53.02, 55.83, 58.77, 61.85,
                         65.18, 68.87, 73.18, 78.83])
        for iphy in np.arange(1,23):
            ipol = 23 - iphy
            np.abs(y0 + 5.), ephy[ipol-1]
            if  np.abs(y0 + 5.) >= ephy[ipol-1]: break
        
        hght = float(8. - ipol)
        pi2 = 0.017453293
        rady = pi2*(6371.0 + hght)
        return rady
        
    
    def _get_observation_time(self, m, completeness_table, last_year):
        ct = completeness_table

        ## TODO retirar o try

        try:
            i = ct[:,1] > m    # all values after desired mag...
            _mt = min(ct[i,1])  # min of these values
            i = np.where(ct[:,1] == _mt)# index of
    
            observation_time = last_year - ct[i,0][0][0] + 1 # corresponding year
        except:
            observation_time = last_year - ct[-1,0] + 1

        return observation_time
    

    
    def _smooth_event_rate(self, 
                           eq_number,
                           eq_tuple, 
                           distances,
                           mags,
                           x, y):
        
        _x, _y, _z, _m, _b, _t = eq_tuple
        
        k = spatial_kernel(_x, _y, _b, kernel_type="powerlaw_2d")

#         z_max = 50
#         if _z > z_max: continue

        #d = distances[eq_number, :]
        #print d
        
        # finding the magnitude bin
        for _bin in np.arange(len(mags)):
            if _m < min(mags) \
            or _m >= max(mags):
                _bin = -1
                break
            
            if _m >= mags[_bin]  \
            and _m < mags[_bin+1]:
               break


        #print _bin
        _eq_rate = 0.0
        # for each CELL on grid
        z_bin = []
        z_cel = []
        z_rat = []
        for i_cell, (x_cell, y_cell) in enumerate(zip(x,y)):
            x_min, x_max = x_cell - self.dx /2., x_cell + self.dx /2.
            y_min, y_max = y_cell - self.dy /2., y_cell + self.dy /2.

            #if d.ravel()[i_cell] >= 6000: continue

            #print x_min, y_min, x_max, y_max            
            #print _bin, ": ", mags[_bin], _m, mags[_bin+1]
            _rate = k.integrate(x_min, x_max, y_min, y_max) / _t
            z_bin.append(_bin)
            z_cel.append(i_cell)
            z_rat.append(_rate)
            
            
#            self.rates[_bin].ravel()[i_cell] += _rate
            _eq_rate += _rate
            
        return [eq_number, _m, _eq_rate, z_bin, z_cel, z_rat]

 
    
    def _callback_smooth_event_rate(self, answer_list):
        [eq_number, magnitude, eq_rate, z_bin, z_cel, z_rat] = answer_list
        
        for b,c,r in zip(z_bin, z_cel, z_rat):
            self.rates[b].ravel()[c] += r
        
        #print bin, cell, rate
        print "ok, eq %04d, m=%2.1f, total_rate = %e"%(eq_number, magnitude, eq_rate)



    def run_analysis(self,
                     catalogue, 
                     config, 
                     smoothing_kernel = IsotropicGaussianWoo):
        '''
        Runs an analysis of smoothed seismicity in the manner
        originally implemented by Frankel (1995)

        :param catalogue:ls

            Instance of the hmtk.seismicity.catalogue.Catalogue class
            catalogue.data dictionary containing the following -
            'year' - numpy.ndarray vector of years
            'longitude' - numpy.ndarray vector of longitudes
            'latitude' - numpy.ndarray vector of latitudes
            'depth' - numpy.ndarray vector of depths
            'magnitude' - numpy.ndarray vector of magnitudes

        :param dict config:
            Configuration settings of the algorithm:
            * 'Length_Limit' - Maximum number of bandwidths for use in
                                smoothing
                               (Float)
            * 'BandWidth' - Bandwidth (km) of the Smoothing Kernel (Float)
            * 'increment' - Output incremental (True) or cumulative a-value
                            (False)
            * 'completeness_table' - Completeness of the catalogue assuming
                                     evenly spaced magnitudes from most recent
                                     bin to oldest bin [year, magnitude]

        :param smoothing_kernel:
            Smoothing kernel as instance of :class:
                hmtk.seismicity.smoothing.kernels.base.BaseSmoothingKernel

        :returns:
            Full smoothed seismicity data as np.ndarray, of the form
            [Longitude, Latitude, Depth, Observed, Smoothed]
        '''

        self.config = config

        use3d = config['use3d']
        nh = config['bandwidth_h_limit']
        mag_bin = config['magnitude_bin']
        
        # this catalogue will be changed...
        self.completeness_table = config['completeness_table']
        self.catalogue = deepcopy(catalogue)

        # remove INCOMPLETE events and get UNIFORM catalogue
        #print self.completeness_table
        self.catalogue.catalogue_mt_filter(self.completeness_table)

        last_observation = self.catalogue.end_year

        _m = self.catalogue.data['magnitude']
        _t = self.catalogue.data['year']
        
        _t2 = self.completeness_table[:,0]
        _m2 = self.completeness_table[:,1]
        
        self.optimize_bandwith_values(magnitude_bin=mag_bin)
        print "h(m) = %.2f * exp(%.2f * m)"%(self.c, self.d)
        
        # add the kernel bandwidth (magnitude dependent) to catalog
        self.catalogue.data['bandwidth'] = self.c * np.exp(self.d * _m) / 111.1 
        #print self.catalogue.data['bandwidth']
        
        
        ## TODO attention
        ct, dm = utils.get_even_magnitude_completeness(self.completeness_table, 
                                                       self.catalogue, 
                                                       magnitude_increment = mag_bin)
        
        
        import matplotlib.pyplot as plt
          
        plt.scatter(_m, _t, marker='+')
        plt.step(_m2, _t2, color='red')
        plt.show()
#         gmag = np.arange(self.config['min_magnitude'], 
#                          max(self.catalogue.data['magnitude']) + mag_bin,
#                          mag_bin)
#         
#         for _m in gmag:
#             print _m, self._get_observation_time(_m, 
#                                                  self.completeness_table, 
#                                                  last_observation)
        
        grid = self._create_grid(use3d=use3d)

        x = grid[:,0]
        y = grid[:,1]

        x0, y0 = min(x), min(y)
#   
#         for _y in np.arange(-90, 90, 10):
#             print self._get_degre_km_factor(0, _y),
        
        # they are the CELL's CENTER POINT !!!
#         print x[0], y[0]
#         print x[-1], y[-1]

        # now, avoid this implementation to follow woo's original one.
        

        M = self.catalogue.data['magnitude']

        ot = np.array([self._get_observation_time(_magnitude, 
                                         self.completeness_table, 
                                         last_observation) for _magnitude in M ])
        self.catalogue.data['obs_time'] = ot
        

        #print self.grid_limits
        self.dx = self.grid_limits['xspc']
        self.dy = self.grid_limits['yspc']
        
        self.nx = int((self.grid_limits['xmax'] - self.grid_limits['xmin']) / 
                       self.dx)
        self.ny = int((self.grid_limits['ymax'] - self.grid_limits['ymin']) / 
                       self.dy)
        
        #print self.nx, self.ny

        min_magnitude = self.config['min_magnitude']
        min_magnitude = min_magnitude if min_magnitude else min(M)
        max_magnitude = max(M)
        
        # divide bins catalog_bins
        mags = np.arange(min_magnitude, max_magnitude + dm, dm)
        self.nm = len(mags)

        self.rates = np.zeros((self.nm, self.nx, self.ny))
        print self.nx * self.ny
        
        #print "uh"
        distances = haversine(self.catalogue.data['longitude'],
                              self.catalogue.data['latitude'],
                              x, 
                              y)


        po = Pool()
        # for each EARTHQUAKE
        for i, eq_tuple in enumerate(zip(self.catalogue.data['longitude'],
                                            self.catalogue.data['latitude'],
                                            self.catalogue.data['depth'],
                                            self.catalogue.data['magnitude'],
                                            self.catalogue.data['bandwidth'],
                                            self.catalogue.data['obs_time'])):
            print 'starting eq', i
            po.apply_async(_unwraper_smoothed_rate, 
                           (self, i, eq_tuple, distances, mags, x, y), 
                           callback=self._callback_smooth_event_rate)
            
                
                
           # loop over cells
        
        # loop over earthquakes
        po.close()
        po.join()

        total = 0.
        for j, _m in enumerate(mags):
                        
            partial = self.rates[j].sum()
            total += partial

            print "m=%3.1f --> rate: %e"%(_m, partial)
          
        print "total rate(m > 3.0) = %e"%(total)      
                
#         x = grid[:,0]
#         y = grid[:,1]
#         print x, len(x)
#         print y, len(y)

        
        _d = []
        for _i in np.arange(self.nx):
            for _j in np.arange(self.ny):
                #print _i, _j, x[_i], y[_j]
                _r = [ x0 + _i*self.dx, y0 + _j*self.dy, 0., min_magnitude, dm, self.rates[:,_j,_i]]
                _d.append(_r)

        self.data = _d
        
        return self.data




############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
#
#         
#         min_magnitude = self.config['min_magnitude']
#         min_magnitude = min_magnitude if min_magnitude else min(M)
#         max_magnitude = max(M)
# 
#         
#         
#         distances = haversine(x, 
#                               y, 
#                               self.catalogue.data['longitude'],
#                               self.catalogue.data['latitude'])
#                 
#         M = self.catalogue.data['magnitude']
#         min_magnitude = self.config['min_magnitude']
#         min_magnitude = min_magnitude if min_magnitude else min(M)
#         max_magnitude = max(M)
#         
#         # divide bins catalog_bins
#         mags = np.arange(min_magnitude, max_magnitude + dm, dm)
#         #mags = mags + dm/2.
#         time = year - ct[:,0] + 1
#
#        #print len(zip(mags, time))
#        #exit()
# 
#         k = Frankel_1995(self.c, self.d)
#         
#         r_max = k.H(mags + dm/2.) * nh
#         #print len(mags), len(h), len(r_max)
#         self.data = []
#         for i, c in enumerate(zip(x,y)):
#             r = distances[i,:]
#             #print i, c, r.shape[0]
#             rates=[]
#             for (m, t, limit) in zip(mags + dm/2, time, r_max):
#                 #print m, m + dm, t
#                 
#                 a = np.logical_and( M >= m - dm/2., M < m + dm/2.)
#                 b = np.logical_and(r <= limit, r > 0 )
#                 _i = np.logical_and(a, b)
#                 
#                 _k = k.kernel(m, r[_i]) / t 
#                 
#                 rates.append(_k.sum())
#             
#             self.data.append([c[0], c[1], 0, min_magnitude, dm, rates])
#         #print self.data


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
                        'Rates': "%s" % str(row[5])[1:-1].replace(",", "").replace("\n", "")}
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
