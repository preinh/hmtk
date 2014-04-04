# -*- coding: utf-8 -*-
#from hmtk.seismicity.smoothing.kernels.gaussian import Kernel
#from sklearn import neighbors as nn

import datetime as dt
import numpy as np

from scipy import spatial, optimize

from hmtk.seismicity.smoothing.kernels import gaussian
from hmtk.seismicity.utils import haversine
from matplotlib import pylab as pl

class smoothing(object):
    
    def __init__(self, catalogue):
        self.catalogue = catalogue


    
    def _cnn_model(self, p, X, k, a, tree):
        return p[0] + a*p[1]
    
    def _cnn_time_constraint(self, p, X, k, a, tree):
        _d, _i = tree.query([0,0], k)
        return p[0] - max(  X[_i, 0]  )
    
    def _cnn_space_constraint(self, p, X, k, a, tree):
        _d, _i = tree.query([0,0], k)
        return p[1] - max(  X[_i, 1]  )
    
    def coupled_nearest_neighbor(self, X, k, a, tree):
        h, d = optimize.fmin_slsqp(self._cnn_model, [10., 10.], 
                                   args=(X, k, a, tree),
                                   ieqcons=[self._cnn_time_constraint, 
                                            self._cnn_space_constraint],
                                   full_output=False)
        return h, d


    def process_catalog_bandwidths(self, k, a, plotting = False):

        catalogue = self.catalogue

        #given reference time
        reference_date = dt.date(catalogue.end_year,12,31)
        
        #given reference time

        # get time (date)  vector
        _y = catalogue.data['year']
        _m = catalogue.data['month']
        _d = catalogue.data['day']
        T = np.array([ dt.date(y, m, d) for (y, m, d) in zip(_y, _m, _d) ])
        
        # just past events from _date accounts
        #_idx = T <= reference_date
#        _idx = T <= reference_date
#        T = T[_idx]

        # select lats and lons
        lx = catalogue.data['longitude']#[_idx]
        ly = catalogue.data['latitude']#[_idx]
        D = haversine(lx, ly, lx, ly)
        
        h = []
        d = []
        for i, v in enumerate(zip(D, T)):
            distances, t0 = v[0], v[1]

            #print t0
            # min dist correction
            _i = distances <= 1.0
            distances[_i] = 1.0

            days = np.array([ (t0 - t).days for t in T ])
            # filter only eq before t0

# may it don't be necessary?!
#             _i = days >= 0
#             days = np.abs(days[_i])
#             distances = distances[_i]
        
            #escape for the 
#             if len(days) <= k: 
#                 d.append(1)
#                 h.append(1)
#                 continue
            
            X = np.array(zip(days.ravel(), distances.ravel()))
            _X = X[1:]

#             if last_X.all() == X.all():
#                 print 'é igual'
#             else:
#                 print 'não é igual'
#                 last_X = _X
            tree = spatial.KDTree(_X)
            hi, di = self.coupled_nearest_neighbor(_X, a, k, tree)
            #print hi, di
            #break
        
            if plotting:
                #pl.axis('equal')
                pl.scatter(days, distances, s=50, c='y', marker='x', alpha=0.7)
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

        self.catalogue.data['h'] = h
        self.catalogue.data['d'] = d
        #return h, d






    
    def pilot_estimator(self, p):
        h0 = p[0]
        d0 = p[1]
    
    
    def rate(self, r, t, r_min, k, a, catalogue):
        
        # compute h, d 
        #h, d = self.compute_bandwidths()
        
        # may it could be computed once ?!
        #catalogue.data['h'] = self.h
        #catalogue.data['d'] = d
        
        
        
        # using only catalog events on t_i < t # past eq
        #     compute time_kernel on t
        #     compute distance_kernel on r
        #     compute normalization_factor
        #     sum their product

        # sum r_min
        # return
        
        pass
        
        
        
    
    def rate_model(self, r, t, p):

        r_min, k, a = p[0], p[1], p[2]
        
        h, d = np.ones(len(r)), np.ones(len(r))
        
        time_kernel = gaussian.Kernel()  
        time_kernel.value(t, h)
        distance_kernel = gaussian.Kernel()
        distance_kernel.value(r, d)

        normalization = 2 / (h * d * d)
        _rate = r_min + sum( normalization * \
                            time_kernel * \
                            distance_kernel )
        return _rate



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
    s.process_catalog_bandwidths(k=4, a=200)
    #s.compute_bandwidths()
    print s.catalogue.data
    #s.rate(1, 3, p=[2,3])