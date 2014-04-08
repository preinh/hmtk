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

#from hmtk.plotting.mapping import HMTKBaseMap 

class smoothing(object):
    
    def __init__(self, catalogue, 
                 grid_limits, 
                 completeness_table,
                 stationary_time_step_in_days = 100,
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
        # decluster target

        # catalog_incompleteness
        self.completeness_table = completeness_table
        # purge 'UNcomplete' events
        # add a weight column to each eq on catalog
        self.estimate_catalogue_incompleteness_weight(learning_catalogue)
        
        # plot catalogues
        #print target_catalogue.get_number_events()
        map_config = {'min_lon': -80,
                      'max_lon': -30,
                      'min_lat': -37,
                      'max_lat': +14,
                      }
#         m2 = HMTKBaseMap(config=map_config, title="catalogs", dpi=90)
#         m2.add_catalogue(learning_catalogue)
#         m1 = HMTKBaseMap(config=map_config, title="catalogs", dpi=90)
#         m1.add_catalogue(target_catalogue)
        
        
        # grid
        self.grid_limits = grid_limits
        # r = grid [[lon, lat]]
        self.r, self.grid_shape = self._create_grid()
        #print self.r.shape


# TODO temporary
# estimation of time integrator...
        _y = np.arange(2000, 1959, -2)
        _m = np.arange(12,0,-6)
        #print _m
        t = []
        for y in _y:
#             for m in _m:
#                 t.append(dt.date(y, m, 01))
            t.append(dt.date(y, 01, 01))
        #exit()
        # 90 days vector ?!?! 
        #t = np.array([ dt.date(year, 01, 01) for year in _y ])
        self.t = np.array(t)
#### temporary

        
        # h, d  =  optimize/learn (ok)
        
        # stationary rates (ok ?!) 
#             _y = np.arange(2000, 1959, -1)
#             t = np.array([ dt.date(year, 01, 01) for year in _y ])
#             print t

        # observed rates (ok?!)

        # likelihood
        
        # gain
    
    
    def _cnn_model(self, p, X, k, a, tree):
        return p[0] + a*p[1]
    
    def _cnn_time_constraint(self, p, X, k, a, tree):
        k = k if (k >= 1) else 1
        _d, _i = tree.query([0., 0.], k=np.round(k))
        return p[0] - np.max(  X[_i, 0]  )
    
    def _cnn_space_constraint(self, p, X, k, a, tree):
        k = k if (k >= 1) else 1
        _d, _i = tree.query([0.,0.], k=np.round(k))
        return p[1] - np.max(  X[_i, 1]  )

    
    def coupled_nearest_neighbor(self, X, k, a, tree):
        h, d = optimize.fmin_slsqp(self._cnn_model, [10., 10.], 
                                   args=(X, k, a, tree),
                                   ieqcons=[self._cnn_time_constraint, 
                                            self._cnn_space_constraint],
                                   full_output=False,
                                   iprint=False)
        return h, d


    def optimize_catalogue_bandwidths(self, k, a, plotting=False):
        
        # optimize only LEARNING catalogue
        catalogue = self.learning_catalogue

        #given reference time
        #reference_date = dt.date(catalogue.end_year,12,31)
        
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
            distances, event_time = v[0], v[1]

            # minimum distance correction
            _i = distances <= 1.0
            distances[_i] = 1.0

            # get 'days' timedelta
            # TODO: ckeck in the future if dond be exclude 'past' events 
            days = np.array([ (event_time - other_event_day).days for other_event_day in T ])
            
            # condense vector [t, r], but exclude exclude the calculation point
            X = np.array(zip(days.ravel(), distances.ravel()))[1:]

            # compute nearest neighbor tree
            tree = spatial.KDTree(X)
            
            # compute each [h, d] by CNN 
            hi, di = self.coupled_nearest_neighbor(X, k, a, tree)
        
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
        catalogue.data['h'] = np.array(h)
        catalogue.data['d'] = np.array(d)



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
        # get LEARNING catalogue
        catalogue = self.learning_catalogue
        
        # get time (date)  vector
        _y = catalogue.data['year']
        _m = catalogue.data['month']
        _d = catalogue.data['day']
        T = np.array([ dt.date(y, m, d) for (y, m, d) in zip(_y, _m, _d) ])
        
        ####  T  I  M  E
        # compute time differences from t to each earthquake i on catalogue
        # select days
        # TODO: generalize for vector entry t
        # but now works with one t @time
        days = np.array([ (t - _t).days for _t in T ])
        # filter catalog t_i < t 
        # only past events
        _i = days <= 0
        days = days[_i]

        ####  S  P  A  C  E
        # filter lats and lons
        lx = catalogue.data['longitude'][_i]
        ly = catalogue.data['latitude'][_i]
        
        # compute distance from r to each earthquake i on catalogue
        distances = haversine(r[:,0], r[:,1], lx, ly)

        # compute hi, di
        #self.optimize_catalogue_bandwidths(k, a)
        
        ####  B  A  N  D  W  I  D  T  H
        # get optimized h, d 
        h = catalogue.data['h'][_i]
        d = catalogue.data['d'][_i]
#         h = np.ones(len(lx))
#         d = np.ones(len(lx))
        
        ####  K  E  R  N  E  L
        # compute kernels and normalization factor
        time_kernel = gaussian.Kernel()  
        K1 = time_kernel.value(days, h)
        #print days.shape, K1.shape
        
        distance_kernel = gaussian.Kernel()
        K2 = distance_kernel.value(distances, d)
        #print distances.shape, K2.shape

        ####  W  E  I  G  H  T
        w = catalogue.data['w'][_i]
        norm = (2. * w ) / (h * d * d)
        #print norm.shape
        #print K1.shape, K1,
        #print K2.shape, K2, 
        #print norm.shape, norm
        # compute rate
        rates = np.array([ r_min + sum(norm * K1 * K2) for K2 in K2 ])
        rates[ (rates < r_min) ] = r_min
        return rates

    def _create_grid(self, use3d=False):
        l = self.grid_limits

        dx = l['xspc']
        dy = l['yspc']
        dz = l['zspc']
        
#         x = np.arange(l['xmin'] + dx/2., l['xmax'] + dx, dx) 
#         y = np.arange(l['ymin'] + dy/2., l['ymax'] + dy, dy) 
#         z = np.arange(l['zmin'] + dz/2., l['zmax'] + dz, dz)
        x = np.arange(l['xmin'], l['xmax'], dx) 
        y = np.arange(l['ymin'], l['ymax'], dy) 
        z = np.arange(l['zmin'], l['zmax'], dz)
        
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
        

    def stationary_rate_model(self, r, t, r_min, k, a, normalized=True):
        #print t
        
        # hi, di, optimization
        self.optimize_catalogue_bandwidths(k=k, a=a)
        # rate timeseries on each cell grid
        #print t
        rates = [ self.rate(r, _t, r_min=r_min, k=k, a=a) for _t in t ]
        #print t
        rates = np.array(rates).T # one timeseries array for each grid point
        
        #print t, t.shape
        #print rates.shape, rates
        # plot ?!
#         for _y in rates:
#             pl.scatter(t, _y)
#             pl.axhline(np.average(_y))
#             pl.show()
    
        # get de median of rates distribution
        rates = np.array([ np.median(rate) for rate in rates ])

        if normalized:
            nt = self.target_catalogue.get_number_events()
            rates = (rates * nt) / sum(rates)
        
#         print rates.shape, rates

        import pylab as p
        from mpl_toolkits.mplot3d import axes3d as p3

        print r_min, k, a
                  
#         fig=p.figure()
#         ax = p3.Axes3D(fig)
#         ax.scatter(r[:,0], r[:,1], rates)
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         fig.add_axes(ax)
#         p.show()


#         exit()

        return rates
        
    
    def observed_number_of_events(self, plot=False):
        c = self.target_catalogue
        x = c.data['longitude']
        y = c.data['latitude']

        l = self.grid_limits
        dx = l['xspc']
        dy = l['yspc']
        
        x_bin = np.arange(l['xmin'], l['xmax'] + dx, dx) 
        y_bin = np.arange(l['ymin'], l['ymax'] + dy, dy) 
    
#         nx = ( l['xmax'] - l['xmin'] ) / l['xspc']
#         ny = ( l['ymax'] - l['ymin'] ) / l['yspc']
#         print (nx, ny)
        heatmap, xedges, yedges = \
            np.histogram2d(x, y, bins=[x_bin, y_bin] )

#         print heatmap
#         print heatmap.shape
#         print xedges
#         print yedges
        if plot:
#             extent = [xedges[0] - dx/2., xedges[-1] - dx/2., 
#                       yedges[0] - dy/2., yedges[-1] - dy/2.]
            extent = [xedges[0], xedges[-1], 
                      yedges[0], yedges[-1]]
            # create map
            from mpl_toolkits.basemap import Basemap, shiftgrid, cm
            _f = pl.figure()
            _ax = _f.add_axes([0.1,0.1,0.8,0.8])
            _ax.set_title('EQ target catalog count')
            
            _m = Basemap(projection='cyl', 
                        llcrnrlon = l['xmin'],
                        llcrnrlat = l['ymin'],
                        urcrnrlon = l['xmax'],
                        urcrnrlat = l['ymax'],
                        suppress_ticks=False,
                        resolution='i', 
                        area_thresh=1000.,
                        ax = _ax)
            _m.drawcoastlines(linewidth=1)
            _m.drawcountries(linewidth=1)
            ax = _m.ax.imshow(heatmap.T, 
                         origin='low', 
                         extent=extent,
                         cmap=pl.cm.spectral_r,
                         interpolation='nearest',
                         #vmin = 000,
                         #vmax = 500,
                         )
            _f.colorbar(ax)
            # plot catalog
            _m.scatter(x, y, alpha=0.1, facecolor='None')
            #show
            pl.show()
        
        # remember that grid is represented by some meshgrid vector...
        shp = (self.grid_shape[0]*self.grid_shape[1], )
        return heatmap.reshape(shp) #observed targets events by grid cell

            

    def poissonian_probability(self, Np, n):
#         print Np, n
#         print Np.shape, n.shape
        p = lambda Np, n : (Np**n)*np.exp(-1*Np) / np.math.factorial(n)
        return np.array( [ p(Np, n) for (Np, n) in zip(Np, n) ] )


    def negative_log_likelihood(self, parameters):
        
        # give parameters meaning
        r_min = parameters[2]
        k = parameters[0]
        a = parameters[1]
        
        Np = self.stationary_rate_model(self.r, self.t, r_min, k, a)
        n = self.observed_number_of_events()

        p = self.poissonian_probability(Np, n)
        
        #p[ (p <= r_min) ] = r_min
        print p
        NLL = np.sum(np.log10( p ))
        print NLL
        return NLL


    def optimize_seismicity_model(self):
        
        r_constraint = lambda p: p[2] - 0.001
        k_constraint = lambda p: p[0] - 1.5
        a_constraint = lambda p: p[1] - 1.5
    
        # Run the minimizer
        initial_parameters = [10, 100, 0.001]
        results = optimize.fmin_cobyla(s.negative_log_likelihood,
                                       initial_parameters,
                                       cons=[r_constraint, 
                                             k_constraint,
                                             a_constraint],
                                       rhobeg = 10.0,
                                       rhoend = 1,
                                       maxfun = 100)
                    
        # Print the results. They should be really close to your actual values
        return results



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

    BASE_PATH = '../../../../pshab/data_input/'
    OUTPUT_FILE = '../../../../pshab/data_output/hmtk_bsb2013_decluster_woo_rates.csv'
    TEST_CATALOGUE = 'hmtk_bsb2013.csv'
    
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
    
        
#     comp_table = np.array([[2000., 2.5],
#                            [1990., 3.0],
#                            [1980., 3.5],
#                            [1970., 4.5],
#                            [1960., 5.0],
#                            [1900., 6.5],
#                            [1800., 7.0]])
    
        
    #print catalogue
    # create grid specifications
    #[xmin, xmax, spcx, ymin, ymax, spcy, zmin, spcz]
    grid_limits = utils.Grid.make_from_list(
                        [ -80, -30, 1, -37, 14, 1, 0, 30, 10])

    # create smooth class 
    s = smoothing(catalogue = catalogue, 
                  grid_limits = grid_limits,
                  completeness_table = comp_table,
                  stationary_time_step_in_days = 100,
                  catalogue_year_divisor = 2000,
                  target_minimum_magnitude = 3.5)

    print s.optimize_seismicity_model()

#     p = [0.01, 14, 200]
#     s.negative_log_likelihood(p)

#     r_constraint = lambda p: p[0] - 1.0e-8
#     k_constraint = lambda p: p[1] - 1
#     a_constraint = lambda p: p[2] - 1
# 
#     from scipy import optimize
#     # Run the minimizer
#     initial_parameters = [1, 2, 1]
#     results = optimize.fmin_slsqp(s.negative_log_likelihood,
#                                    initial_parameters,
#                                    ieqcons=[r_constraint, 
#                                             k_constraint,
#                                             a_constraint],
#                                   iter = 100)
#                 
#     # Print the results. They should be really close to your actual values
#     print results

    #s.compute_bandwidths()

    #r = np.array([ [-46, -26], [-50, -20], [-40, -10] ])
    #t = np.array([ dt.date(2010, 01, 01) ])

#     _y = np.arange(2000, 1959, -1)
#     _m = np.arange(11,0,-2)
#     print _m
#     t = []
#     for y in _y:
#         for m in _m:
#             t.append(dt.date(y, m, 01))
#     #exit()
#     # 90 days vector ?!?! 
#     #t = np.array([ dt.date(year, 01, 01) for year in _y ])
#     t = np.array(t)
    
    # grid space
    #r, grid_shape = s._create_grid(grid_limits)

    #rates = s.stationary_rate_model(s.r, t, r_min=0.01, k=4, a=100)
    #print sum(rates), rates

    #print sum_rates * nt / nl
    
    #print stationary_rates.shape
    #s.optimize_catalogue_bandwidths(k=3, a=20)
    #rates = s.rate(r, t, r_min=0.001, k=4, a=200)

    
#     pl.imshow(stationary_rates.reshape(grid_shape), origin='lower')
#     pl.colorbar()
#     pl.show()    

#     import pylab as p
#     from mpl_toolkits.mplot3d import axes3d as p3
#     
#     fig=p.figure()
#     ax = p3.Axes3D(fig)
#     ax.scatter(s.r[:,0], s.r[:,1], rates)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     fig.add_axes(ax)
#     p.show()
    
    # make grid ?!!?
    
    #r = compute_rates()
    #print s.catalogue.data
    #s.rate(1, 3, p=[2,3])