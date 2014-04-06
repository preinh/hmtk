# -*- coding: utf-8 -*-
#from hmtk.seismicity.smoothing.kernels.gaussian import Kernel
#from sklearn import neighbors as nn

import datetime as dt
import copy
import numpy as np

from scipy import spatial, optimize

from hmtk.seismicity.smoothing import utils
from hmtk.seismicity.smoothing.kernels import gaussian
from hmtk.seismicity.utils import haversine
from matplotlib import pylab as pl

from hmtk.plotting.mapping import HMTKBaseMap 

class smoothing(object):
    
    def __init__(self, catalogue, grid_limits, completeness_table, 
                 catalogue_year_divisor = 2000,
                 target_minimum_magnitude = 4.0):
        self.catalogue = catalogue

        # divide into learning catalog
        learning_catalogue = copy.deepcopy(catalogue)
        learn_condition = catalogue.data['year'] <= catalogue_year_divisor
        learning_catalogue.select_catalogue_events( learn_condition )
        self.learning_catalogue = learning_catalogue

        # and target catalog with magnitude threshold
        target_catalogue = copy.deepcopy(catalogue)
        target_condition = catalogue.data['year'] > catalogue_year_divisor
        target_catalogue.select_catalogue_events( target_condition )
        magnitude_condition = target_catalogue.data['magnitude'] >= target_minimum_magnitude
        target_catalogue.select_catalogue_events( magnitude_condition )
        self.target_catalogue = target_catalogue

        # TODO: decluster target_catalogue
        
        # grid
        self.grid_limits = grid_limits
        # r = grid [[lon, lat]]
        self.r, self.grid_shape = self._create_grid()
        
        # catalog_incompleteness
        self.completeness_table = completeness_table
        # purge 'UNcomplete' events
        # add a weight column to each eq on catalog
        self.estimate_catalogue_incompleteness_weight(learning_catalogue)
        
        # plot catalogues
        #print target_catalogue.get_number_events()
#         map_config = {'min_lon': -80,
#                       'max_lon': -30,
#                       'min_lat': -37,
#                       'max_lat': +14,
#                       }
#         m2 = HMTKBaseMap(config=map_config, title="catalogs", dpi=90)
#         m2.add_catalogue(learning_catalogue)
#         m1 = HMTKBaseMap(config=map_config, title="catalogs", dpi=90)
#         m1.add_catalogue(target_catalogue)
        
        
        
        # h, d  =  optimize/learn 
        
        # decluster target

        # stationary rates
        # observed rates

        # likelihood
        
        # gain
    
    
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



    def estimate_catalogue_incompleteness_weight(self, catalogue):
        """
        :param r: numpy.ndarray space of earthquake coordinates [x, y], [x, y, z]
        :type r: numpy.ndarray
        :param t: datetime.datetime time of earthquake coordinate
        :type t: datetime.datetime
        :param b_value: float b_value
        :type b_value: flaot
        :param md: float minimum catalog magnitude
        :type md: float

        :returns: weight for each earthquake kernel, on (r, t)  
                 (more for ancient uncompleted space_time)
        :rtype: numpy.array
        """
        b = 1.0
        _c = completeness(self.completeness_table)
        #catalogue.data['mc'] = mc
        
        X,Y,Z,M,T = catalogue_to_XYZMT(catalogue)
        
        md = min(M)
        r = np.array(zip(X,Y))
        #print "completenesss:", len(X)
        mc = _c.magnitude_completeness(r, T)
        catalogue.select_catalogue_events( (mc > 0) )

        w = 10.0**(b*mc - md)
        #w = w [ w >= 1]
        #print w
        catalogue.data['w'] = w
        return w #print w
        



    
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

class completeness(object):
    '''
    Class to handle magnitude completeness for wheight correction


    :attribute completeness_table:
        [[year magnitude]
         [year magnitude]...]
    '''
    def __init__(self, completeness_table):
        self.completeness_table = completeness_table
        
        
    def magnitude_completeness(self, r, t):
        """
        :param r: numpy.ndarray of space coordinates [x, y], [x, y, z]
        :type r: numpy.ndarray
        :param t: datetime.datetime time coordinate
        :type t: datetime.datetime

        :returns: magnitude completeness value or 0
        :rtype: numpy.array
        """
        c = self.completeness_table
        y = c[:,0]
        m = c[:,1]
        #print t, y, m
        years = [ min( y[ (y >= t.year) ] ) if len( y[ (y >= t.year) ] ) > 0 else 0 for t in t ]
        mc = [ np.array([0]) if _i == 0 else m[(y == _i)] for _i in years ]
        
        return np.array(mc).T[0]
        

def catalogue_to_XYZMT(catalogue):
    c = catalogue
    
    X = np.array(c.data['longitude'])
    Y = np.array(c.data['latitude'])
    Z = np.array(c.data['depth'])
    M = np.array(c.data['magnitude'])
    T = np.array([ dt.date(e[0], e[1], e[2]) \
                  for e in zip(c.data['year'], 
                               c.data['month'], 
                               c.data['day']) ])

    return X, Y, Z, M, T

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

    # Time-varying completeness
    comp_table = np.array([[1990., 3.0],
                           [1980., 3.5],
                           [1970., 4.5],
                           [1960., 5.0],
                           [1900., 6.5],
                           [1800., 7.0]])
    
#    mag_completeness = completeness(comp_table)
    t = np.array([ dt.date(1991, 01, 01),
                   dt.date(1990, 01, 01) ,
                   dt.date(1979, 01, 01)  ])

#     c = comp_table
#     y = c[:,0]
#     m = c[:,1]
#     
#     years = [ min( y[ (y >= t.year) ] ) if len( y[ (y >= t.year) ] ) > 0 else 0 for t in t ]
#     mc = [ np.array([0]) if _i == 0 else m[(y == _i)] for _i in years ]
#     
#     mc = np.array(mc).T
#     
    
#    print mag_completeness.magnitude(r=[0,0], t=t)
#    exit()
    
    #print catalogue
    # create grid specifications
    #[xmin, xmax, spcx, ymin, ymax, spcy, zmin, spcz]
    grid_limits = utils.Grid.make_from_list(
                        [ -80, -30, 1, -37, 14, 1, 0, 30, 10])

    # create smooth class 
    s = smoothing(catalogue, 
                  grid_limits,
                  comp_table)
    #s.compute_bandwidths()
    # grid space
    #r, grid_shape = s._create_grid(grid_limits)

    #r = np.array([ [-46, -26], [-50, -20], [-40, -10] ])
    #t = np.array([ dt.date(2010, 01, 01) ])
    exit()
    print s.rate_model(s.r, t, r_min=0.01, k=4, a=100)

    #s.optimize_catalogue_bandwidths(k=3, a=20)
#    rates = s.rate(r, t, r_min=0.001, k=4, a=200)
    
#     pl.imshow(rates.reshape(grid_shape), origin='lower')
#     pl.colorbar()
#     pl.show()    
    # make grid ?!!?
    
    #r = compute_rates()
    #print s.catalogue.data
    #s.rate(1, 3, p=[2,3])