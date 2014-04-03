# -*- coding: utf-8 -*-
#from hmtk.seismicity.smoothing.kernels.gaussian import Kernel
#from sklearn import neighbors as nn

import datetime as dt
import numpy as np

from scipy import spatial, optimize

from hmtk.seismicity.utils import haversine
from matplotlib import pylab as pl

class smoothing(object):
    
    def __init__(self, catalogue, plotting = True):
        self.catalogue = catalogue

        k = 30
        a = 200

        #given reference time
        reference_date = dt.date(2013,1,1)
        
        #given reference time
        cut_off_date = dt.date(1960,1,1)
        

        # get time (date)  vector
        _y = catalogue.data['year']
        _m = catalogue.data['month']
        _d = catalogue.data['day']
        T = np.array([ dt.date(y, m, d) for (y, m, d) in zip(_y, _m, _d) ])
        
        # just past events from _date accounts
        #_idx = T <= reference_date
        _idx = np.logical_and(T <= reference_date, T >= cut_off_date)
        T = T[_idx]

        # select lats and lons
        lx = catalogue.data['longitude'][_idx]
        ly = catalogue.data['latitude'][_idx]
        D = haversine(lx, ly, lx, ly)

        #pl.imshow(D, origin='lower')
        #pl.show()
        # for each earthquake i
        #print zip(D, T)
        
        h = []
        d = []
        for i, v in enumerate(zip(D, T)):
            distances, t0 = v[0], v[1]

            # min dist correction
            _i = distances <= 1.0
            distances[_i] = 1.0
            times = np.array([ (t0 - t).days for t in T ])

            # filter only eq before t0
            _i = times <= 0
            times = times[_i]
            distances = distances[_i]
            
            X = np.array(zip(times.ravel(), distances.ravel()))
            
            hi, di = self.coupled_nearest_neighbor(X, a, k)
            print hi, di
            
            if plotting:
                #pl.axis('equal')
                pl.scatter(times, distances, s=50, c='y', marker='x', alpha=0.7)
                pl.ylim(ymin=0)
                pl.axvline(hi)
                pl.axhline(di)
                pl.show()

            d.append(di)
            h.append(hi)
            #print i, eq_d, eq_t
            #print h_i, d_i, "sum:", h_i + a*d_i
        h = np.array(h)
        d = np.array(d)
        print h, d
    
    def _cnn_model(self, p, X, a, k):
        return p[0] + a*p[1]
    
    def _cnn_time_constraint(self, p, X, a, k):
        tree = spatial.KDTree(X)
        _d, _i = tree.query([0,0], k)
        return p[0] - max(  X[_i, 0]  )
    
    def _cnn_space_constraint(self, p, X, a, k):
        tree = spatial.KDTree(X)
        _d, _i = tree.query([0,0], k)
        print k, _d, _i
        return p[1] - max(  X[_i, 1]  )
    
    def coupled_nearest_neighbor(self, X, a, k):
        h, d = optimize.fmin_slsqp(self._cnn_model, [-10., 10.], 
                                   args=(X, a, k),
                                   ieqcons=[self._cnn_time_constraint, 
                                            self._cnn_space_constraint],
                                   full_output=False)
        return h, d
    
    
    def rate(self, r, t, p):
        h_i, d_i = self._coupled_nearest_neighbour(p)
        print h_i, d_i


    def _pilot_estimate(self, p):
        h0 = p[0]
        d0 = p[1]
        


if __name__ == '__main__':
    from hmtk.parsers.catalogue.csv_catalogue_parser import CsvCatalogueParser
    import os

    BASE_PATH = '/Users/pirchiner/dev/pshab/data_input/'
    OUTPUT_FILE = 'data_output/hmtk_bsb2013_decluster_woo_rates.csv'
    TEST_CATALOGUE = 'hmtk_bsb2013_decluster.csv'
    
    _CATALOGUE = os.path.join(BASE_PATH,TEST_CATALOGUE)
    
    # catalogue
    parser = CsvCatalogueParser(_CATALOGUE)
    catalogue = parser.read_file()
    
    catalogue.sort_catalogue_chronologically()
    
    s = smoothing(catalogue=catalogue)
    #s.rate(1, 3, p=[2,3])