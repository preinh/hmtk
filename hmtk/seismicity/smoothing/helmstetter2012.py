# -*- coding: utf-8 -*-
#from hmtk.seismicity.smoothing.kernels.gaussian import Kernel
#from sklearn import neighbors as nn

import datetime as dt
import numpy as np

from scipy import spatial, optimize

from hmtk.seismicity.smoothing import utils
from hmtk.seismicity.smoothing.kernels import gaussian
from hmtk.seismicity.utils import haversine
from matplotlib import pylab as pl

class smoothing(object):
    
    def __init__(self, catalogue, grid_limits):
        self.catalogue = catalogue
        self.r, self.grid_shape = self._create_grid()



    
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


    def optimize_catalogue_bandwidths(self, k, a, plotting = False):

        catalogue = self.catalogue

        #given reference time
        reference_date = dt.date(catalogue.end_year,12,31)
        
        # get time (date)  vector
        _y = catalogue.data['year']
        _m = catalogue.data['month']
        _d = catalogue.data['day']
        T = np.array([ dt.date(y, m, d) for (y, m, d) in zip(_y, _m, _d) ])

        # select lats and lons
        lx = catalogue.data['longitude']#[_idx]
        ly = catalogue.data['latitude']#[_idx]
        D = haversine(lx, ly, lx, ly)
        
        h, d = [], []
        
        #for each earthquake i
        for i, v in enumerate(zip(D, T)):
            # variable init
            distances, origin_time = v[0], v[1]

            # minimum distance correction
            _i = distances <= 1.0
            distances[_i] = 1.0

            # get 'days' timedelta
            # TODO: ckeck in the future if dond be exclude 'past' events 
            days = np.array([ (origin_time - t).days for t in T ])
            
            # condense vector [t, r], but exclude exclude the calculation point
            X = np.array(zip(days.ravel(), distances.ravel()))[1:]

            # compute nearest neighbor tree
            tree = spatial.KDTree(X)
            
            # compute each [h, d] by CNN 
            hi, di = self.coupled_nearest_neighbor(X, a, k, tree)
        
#             if plotting:
#                 pl.scatter(days, distances, s=50, c='y', marker='x', alpha=0.7)
#                 pl.ylim(ymin=0)
#                 pl.axvline(hi)
#                 pl.axhline(di)
#                 pl.show()

            d.append(di)
            h.append(hi)

        # after process each one, 
        # store optimized h and d for given catalog
        self.catalogue.data['h'] = np.array(h)
        self.catalogue.data['d'] = np.array(d)


    
    def pilot_estimator(self, p):
        h0, d0 = p[0], p[1]
        pass

    
    def rate(self, r, t, r_min, k, a):
        # get catalogue
        catalogue = self.catalogue
        
        # get time (date)  vector
        _y = catalogue.data['year']
        _m = catalogue.data['month']
        _d = catalogue.data['day']
        T = np.array([ dt.date(y, m, d) for (y, m, d) in zip(_y, _m, _d) ])
        
        # compute time differences from t to each earthquake i on catalogue
        # select days
        # TODO: generalize for vector entry t
        days = np.array([ (t[0] - _t).days for _t in T ])
        # filter catalog t_i < t 
        # only past events
        _i = days <= 0
        days = days[_i]

        # filter lats and lons
        lx = catalogue.data['longitude'][_i]
        ly = catalogue.data['latitude'][_i]
        
        # compute distance from r to each earthquake i on catalogue
        distances = haversine(r[:,0], r[:,1], lx, ly)

        # compute hi, di
        #self.optimize_catalogue_bandwidths(k, a)
        
        # get optimized h, d
        h = self.catalogue.data['h'][_i]
        d = self.catalogue.data['d'][_i]
#         h = np.ones(len(lx))
#         d = np.ones(len(lx))
        
        # compute kernels and normalization factor
        time_kernel = gaussian.Kernel()  
        K1 = time_kernel.value(days, h)
        #print days.shape, K1.shape
        
        distance_kernel = gaussian.Kernel()
        K2 = distance_kernel.value(distances, d)
        #print distances.shape, K2.shape

        norm = 2. / (h * d * d)
        #print norm.shape
        #print K1, K2, norm
        # compute rate
        rates = [ r_min + sum(norm * K1 * K2) for K2 in K2 ] 
        return np.array(rates)

    def _create_grid(self, use3d=False):
        l = self.grid_limits

        dx = l['xspc']
        dy = l['yspc']
        dz = l['zspc']
        
        x = np.arange(l['xmin'] + dx/2., l['xmax'] + dx, dx) 
        y = np.arange(l['ymin'] + dy/2., l['ymax'] + dy, dy) 
        z = np.arange(l['zmin'] + dz/2., l['zmax'] + dz, dz)
        
        if not use3d:
            #spacement = [dx, dy]
            _shape = (len(x), len(y))
            xx, yy= np.meshgrid(x, y)
            x, y = xx.flatten(), yy.flatten()
            cells = np.array(zip(x,y))
        else:
            #spacement = [dx, dy, dz]
            _shape = (len(x), len(y), len(z))
            xx, yy, zz = np.meshgrid(x, y, z)
            x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
            cells = np.array(zip(x,y,z))

        return cells, _shape #, spacement
        

    def rate_model(self, r, t, r_min, a, k):
        self.optimize_catalogue_bandwidths(k=k, a=a)
        
        rates = self.rate(r, t, r_min=r_min, k=k, a=a)

        return rates
    
    
#     def rate_model(self, r, t, p):
# 
#         r_min, k, a = p[0], p[1], p[2]
#         
#         h, d = np.ones(), np.ones()
#         
#         time_kernel = gaussian.Kernel()  
#         time_kernel.value(t, h)
#         distance_kernel = gaussian.Kernel()
#         distance_kernel.value(r, d)
# 
#         normalization = 2 / (h * d * d)
#         _rate = r_min + sum( normalization * \
#                             time_kernel * \
#                             distance_kernel )
#         return _rate



if __name__ == '__main__':
    from hmtk.parsers.catalogue.csv_catalogue_parser import CsvCatalogueParser
    import os

    BASE_PATH = '/Users/pirchiner/dev/pshab/data_input/'
    OUTPUT_FILE = 'data_output/hmtk_bsb2013_decluster_woo_rates.csv'
    TEST_CATALOGUE = 'hmtk_bsb2013_decluster.csv'
    
    _CATALOGUE = os.path.join(BASE_PATH,TEST_CATALOGUE)
    
    # get catalogue
    parser = CsvCatalogueParser(_CATALOGUE)
    catalogue = parser.read_file()
    catalogue.sort_catalogue_chronologically()
    

    # create grid specifications
    #[xmin, xmax, spcx, ymin, ymax, spcy, zmin, spcz]
    grid_limits = utils.Grid.make_from_list(
                        [ -80, -30, 1, -37, 14, 1, 0, 30, 10])

    # create smooth class 
    s = smoothing(catalogue=catalogue, grid_limits)
    #s.compute_bandwidths()
    # grid space
    #r, grid_shape = s._create_grid(grid_limits)

    #r = np.array([ [-46, -26], [-50, -20], [-40, -10] ])
    t = np.array([ dt.date(2010, 01, 01) ])
    
    s.optimize_catalogue_bandwidths(k=3, a=20)
    rates = s.rate(r, t, r_min=0.001, k=4, a=200)

#     pl.imshow(rates.reshape(grid_shape), origin='lower')
#     pl.colorbar()
#     pl.show()    
    # make grid ?!!?
    
    #r = compute_rates()
    #print s.catalogue.data
    #s.rate(1, 3, p=[2,3])