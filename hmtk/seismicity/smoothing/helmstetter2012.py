# -*- coding: utf-8 -*-

import datetime as dt
import copy
import numpy as np

from scipy import spatial, optimize

from hmtk.seismicity.smoothing import utils
from hmtk.seismicity.smoothing.kernels import gaussian
from hmtk.seismicity.utils import haversine
#from matplotlib import pylab as pl

from multiprocessing import Pool

class smoothing(object):
    
    def __init__(self, catalogue, config ):

        self.catalogue = catalogue
                 
        self.log = config['log']
        self.plot_bandwidth = config['plot_bandwidth']
        self.plot_rate_timeseries = config['plot_rate_timeseries']
        self.plot_stationary_rate = config['plot_stationary_rate']
        self.plot_target_events_count = config['plot_target_events_count'] 

        catalogue_year_divisor = config['catalogue_year_divisor']
        catalogue_year_start = config['catalogue_year_start']

        # divide into learning catalog
        learning_catalogue = copy.deepcopy(catalogue)
        learn_condition = np.logical_and(
                            catalogue.data['year'] >= catalogue_year_start,
                            catalogue.data['year'] <= catalogue_year_divisor
                            )
        learning_catalogue.select_catalogue_events( learn_condition )
        self.learning_catalogue = learning_catalogue

        # and target catalog with magnitude threshold
        target_catalogue = copy.deepcopy(catalogue)
        add_before_learning_on_target = config['add_before_learning_on_target']
        if add_before_learning_on_target:
            target_condition = np.logical_or(
                            catalogue.data['year'] < catalogue_year_start,
                            catalogue.data['year'] > catalogue_year_divisor)
            target_catalogue.select_catalogue_events( target_condition )
        else:
            target_condition = (catalogue.data['year'] > catalogue_year_divisor)
            target_catalogue.select_catalogue_events( target_condition )

        magnitude_condition = target_catalogue.data['magnitude'] >= config['target_minimum_magnitude']
        target_catalogue.select_catalogue_events( magnitude_condition )
        self.target_catalogue = target_catalogue

        # TODO: decluster target_catalogue
        # decluster target

        # catalog_incompleteness
        self.completeness_table = config['completeness_table']
        self.b_value = config['b_value']
        # purge 'UNcomplete' events
        # add a weight column to each eq on catalog
        self.estimate_catalogue_incompleteness_weight(self.learning_catalogue)
        
        # SPACE: grid limits, cells, shape
        self.grid_limits = config['grid_limits']
        self.r, self.grid_shape = self._create_grid()

        # TIME: descendent order #important
        time_step = config['stationary_time_step_in_days']
        _time_step = dt.timedelta(days=time_step)
        
        _end = dt.date(catalogue_year_divisor, 01, 01)
        _start = dt.date(int(self.learning_catalogue.data['year'].min()), 01, 01)
        _i, t = 0, []
        while _end - _i*_time_step >= _start:
            t.append(_end - _i*_time_step)
            _i += 1
        #print t
        #exit()
        self.t = np.array(t)


    def _create_grid(self, use3d=False):
        l = self.grid_limits

        dx = l['xspc']
        dy = l['yspc']
        dz = l['zspc']
        
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

        return cells, _shape 



    #### couple nearest neighbor analysis    
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


    
    def coupled_nearest_neighbor(self, X, k, a, i):
        tree = spatial.KDTree(X)
        h, d = optimize.fmin_slsqp(self._cnn_model, [10., 10.], 
                                   args=(X, k, a, tree),
                                   ieqcons=[self._cnn_time_constraint, 
                                            self._cnn_space_constraint],
                                   full_output=False,
                                   iprint=False)
        return i, h, d

    
    #### END: couple nearest neighbor analysis

    def optimize_catalogue_bandwidths(self, k, a, plotting=False):
        
        # optimize only LEARNING catalogue
        catalogue = self.learning_catalogue

        # get time and distance vector
        X,Y,Z,M,T = catalogue_to_XYZMT(catalogue)
        D = haversine(X, Y, X, Y)
        
        h = np.zeros(len(X))
        d = np.zeros(len(X))
        
        po = Pool()

        def cb_cnn(r):
            i, hi, di = r[0], r[1], r[2]   
            #print i, hi, di
            d[i] = di
            h[i] = hi

        #print h
        #for each earthquake i
        for i, v in enumerate(zip(D, T)):
            # variable init
            distances, event_time = v[0], v[1]

            # get time: days timedelta
            # TODO: ckeck in the future if dont be exclude 'past' events 
            days = np.array([ (event_time - other_time).days for other_time in T ])

            # get distances for each another earthquake
            # minimum distance correction
            _i = distances <= 1.0
            distances[_i] = 1.0
            
            # condensed vector [t, r], but exclude exclude the calculation point
            TD = np.array( zip( days.ravel(), distances.ravel() ) )[1:]

            # compute nearest neighbor tree
            
            # compute each [h, d] by CNN 
            po.apply_async(wrap_cnn, (self, TD, k, a, i, ), callback=cb_cnn)
            #hi, di = self.coupled_nearest_neighbor(TD, k, a, tree)
        
#             if self.plot_bandwidth:
#                 pl.scatter(days, distances, s=50, c='y', marker='x', alpha=0.7)
#                 pl.ylim(ymin=0)
#                 pl.axvline(hi)
#                 pl.axhline(di)
#                 pl.show()
#  
#             d.append(di)
#             h.append(hi)

        po.close()
        po.join()

        #print h
        
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
        b = self.b_value
        _c = completeness(self.completeness_table)
        
        X,Y,Z,M,T = catalogue_to_XYZMT(catalogue)
        
        md = min(M)
        location = np.array(zip(X,Y))
        
        mc = _c.magnitude_completeness(location, T)
        catalogue.select_catalogue_events( (mc > 0) )

        # learning completeness weight factor
        w = 10.0**(b*mc - md)
        catalogue.data['w'] = w
        
    
    def pilot_estimator(self, p):
        h0, d0 = p[0], p[1]
        pass

    
    def seismicity_rate(self, r, t, r_min, k, a, i):
        # get LEARNING catalogue
        catalogue = self.learning_catalogue
        
        # get time (date)  vector
        _y = catalogue.data['year']
        _m = catalogue.data['month']
        _d = catalogue.data['day']
        T = np.array([ dt.date(y, m, d) for (y, m, d) in zip(_y, _m, _d) ])
        
        ####  T  I  M  E
        # select 'days before'
        # now works with one t @time
        # future: generalize for vector entry t
        # compute time differences from t to each earthquake i on catalogue
        days = np.array([ (t - _t).days for _t in T ])
        
        # only past events
        _i = days <= 0
        days = days[_i]

        ####  S  P  A  C  E
        # filter lats and lons
        lx = catalogue.data['longitude'][_i]
        ly = catalogue.data['latitude'][_i]
        
        # compute distance from r to each earthquake i on catalogue
        distances = haversine(r[:,0], r[:,1], lx, ly)
        
        ####  B  A  N  D  W  I  D  T  H
        # get optimized h, d 
        h = catalogue.data['h'][_i]
        d = catalogue.data['d'][_i]
        
        ####  K  E  R  N  E  L
        # compute kernels and normalization factor
        time_kernel = gaussian.Kernel()  
#        print "k1"
        K1 = time_kernel.value(days, h)
        
        distance_kernel = gaussian.Kernel()
#        print "k2"
        K2 = distance_kernel.value(distances, d)

        ####  W  E  I  G  H  T
#        print "norm"
        w = catalogue.data['w'][_i]
        norm = (2. * w ) / (h * d * d)

        # compute seismicity_rate
        # K2 is bi-dimensional
#        print "rates"
        rates = np.array([ r_min + sum(norm * K1 * K2) for K2 in K2 ]) 
        rates[ (rates <= 0) ] = r_min
#        print "end"

        return i, rates


    def stationary_rate_model(self, r, t, r_min, k, a, normalized=True):
        #print t
        
        # hi, di, optimization
        if self.log: print "optimizing catalog bandwidth "
        self.optimize_catalogue_bandwidths(k=k, a=a)
        # seismicity_rate timeseries on each cell grid
        
        if self.log: print "getting smoothed seismicity seismicity_rate timeseries"
        
        rates = np.zeros( (len(t), len(r)) )
        po = Pool()
        
        def cb_sr(r):
            i, rate = r[0], r[1]
            rates[i,:] = rate

        for i, _t in enumerate(t):
            po.apply_async(wrap_seismicity_rate, (self, r, _t, r_min, k, a, i,), 
                           callback=cb_sr) 
        
        po.close()
        po.join()
        
        rates = np.array(rates).T # one timeseries array for each grid point
        
        if self.plot_rate_timeseries:
            for _y in rates:
                pl.plot(t, _y,'g-')
                pl.scatter(t, _y, c='g', marker='+')
                pl.axhline(np.median(_y), c='r')
                pl.title('Stationary Modeled Seismicity Rate in one cell')
                pl.show()
    
        # get de median of rates distribution
        rates = np.array([ np.median(rate) for rate in rates ])

        if normalized:
            nt = self.target_catalogue.get_number_events()
            rates = (rates * nt) / sum(rates)
 
        if self.plot_stationary_rate:       
            #from mpl_toolkits.mplot3d import axes3d as p3
#             l = self.grid_limits
#             dx = l['xspc']
#             dy = l['yspc']
#             
#             _X = np.arange(l['xmin'], l['xmax'] + dx, dx) 
#             _Y = np.arange(l['ymin'], l['ymax'] + dy, dy) 
#             _x, _y = np.meshgrid(_X, _Y)
            _x, _y = r[:,0], r[:,1]
            #plpmeshcolor()
            pl.scatter(_x, _y, c=rates, cmap=pl.cm.get_cmap('RdYlGn_r'), marker='s', alpha=0.7)
            pl.colorbar()
            pl.title('Stationary Seismicity Rate')
            pl.show()
            
            import csv
            fid = open('stationary_rate_bsb2013.csv', 'wt')
            # Create header list
            header_info = ['Longitude', 'Latitude', 'Rate']
            writer = csv.DictWriter(fid, fieldnames=header_info)
            headers = dict((name0, name0) for name0 in header_info)
            # Write to file
            writer.writerow(headers)
            for (__x, __y, __r) in zip(_x, _y, rates):
                row_dict = {'Longitude': '%.5f' % __x,
                            'Latitude': '%.5f' % __y,
                            'Rate': '%.3f' % __r}
                writer.writerow(row_dict)
            fid.close()


        if self.log: print "reporting parameters: r_min=%.4f, k=%d, a=%d" % (r_min, k, a)

        return rates
        
    
    def observed_number_of_events(self):
        c = self.target_catalogue
        x = c.data['longitude']
        y = c.data['latitude']

        l = self.grid_limits
        dx = l['xspc']
        dy = l['yspc']
        
        x_bin = np.arange(l['xmin'], l['xmax'] + dx, dx) 
        y_bin = np.arange(l['ymin'], l['ymax'] + dy, dy) 
    
        heatmap, xedges, yedges = \
            np.histogram2d(x, y, bins=[x_bin, y_bin] )

        if self.plot_target_events_count:
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

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
        p = lambda Np, n : (Np**n)*np.exp(-1*Np) / np.math.factorial(n)
        return np.array( [ p(Np, n) for (Np, n) in zip(Np, n) ] )

    def negative_log_likelihood(self, parameters):
        
        # give parameters meaning
        r_min = parameters[2]
        k = parameters[1]
        a = parameters[0]
        Np = self.stationary_rate_model(self.r, self.t, r_min, k, a)
        n = self.observed_number_of_events()

        p = self.poissonian_probability(Np, n)
        
        # noise protection...
        p[ (p <= 0) ] = np.abs(r_min)

        # negative likelihood
        NLL = np.sum(np.log10( p ))
        
        if self.log: print "likelyhood evaluation: ", NLL
        return NLL


    def optimize_seismicity_model(self):
        
        from openopt import MINLP 
        from openopt import GLP
        f = s.negative_log_likelihood
        
        x0=[1, 1, 1] # a, k, r_min
        #p = MINLP(f, x0, maxIter = 1e3)
        p = GLP(f, x0, maxIter = 1e4)

        p.lb = [  0,   1,  0]
        p.ub = [500, 100,  1]

        #p.contol = 1.1e-6

        p.name = 'glp_1'
        nlpSolver = 'de'
        
        # coords of discrete variables and sets of allowed values
        #p.discreteVars = {0:range(0, int(1e3)), 1:range(1, int(2e2))}
        
        # required tolerance for discrete variables, default 10^-5
        p.discrtol = 1.1e-5
        
        results = p.solve('de', plot = False)

#         r_constraint = lambda p: p[2] - 0.001
#         k_constraint = lambda p: p[1] - 1.5
#         a_constraint = lambda p: p[0] - 1.5
#  
#         if self.log: print "optimizing model"
#     
#         # Run the minimizer
#         
#         results = optimize.fmin_cobyla(s.negative_log_likelihood,
#                                        initial_parameters,
#                                        cons=[a_constraint,
#                                              k_constraint,
#                                              r_constraint],
#                                        rhobeg = 50.0,
#                                        rhoend = 1,
#                                        maxfun = 100)
                    
        # Print the results. They should be really close to your actual values
        return results


    def plot_catalogue(self, catalogue, 
                       map_config = { 'min_lon': -80,
                                      'max_lon': -30,
                                      'min_lat': -37,
                                      'max_lat': +14,
                                      },
                       title="Catalog Plot", 
                       dpi = 90 ):
        from hmtk.plotting.mapping import HMTKBaseMap 
        map = HMTKBaseMap(config=map_config, title=title, dpi=dpi)
        map.add_catalogue(catalogue)



## TODO:
    # gain
    # tests

def wrap_cnn(*karg, **kwarg):
    return karg[0].coupled_nearest_neighbor(*karg[1:], **kwarg)

def wrap_seismicity_rate(*karg, **kwarg):
    return karg[0].seismicity_rate(*karg[1:], **kwarg)



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
    
        
    # create grid specifications
    grid_limits = utils.Grid.make_from_list(
        #[xmin, xmax, spcx, ymin, ymax, spcy, zmin, zmax, spcz]
         [ -80,  -30,    1,  -37,   14,    1,    0,   30,   10])


    # configure 
    config = {'grid_limits' : grid_limits, 
              'completeness_table' : comp_table,
              'b_value' : 1.0, 
              'stationary_time_step_in_days': 365,
              'catalogue_year_start': 1950,
              'catalogue_year_divisor': 2000,
              'target_minimum_magnitude': 3.5,
              'add_before_learning_on_target': True,
              'log': True,
              'plot_bandwidth': False,
              'plot_rate_timeseries': False,
              'plot_stationary_rate': False,
              'plot_target_events_count': False,
          }

    # and create smoothing class 
    s = smoothing(catalogue = catalogue, 
                  config = config)

#     s.plot_catalogue(s.learning_catalogue, title="Learning Catalogue [1960-2000]")
#     s.plot_catalogue(s.target_catalogue, title="Target Catalogue U \ ]1960-2000[")

    print s.optimize_seismicity_model()
#     s.plot_stationary_rate = True
#     s.stationary_rate_model(s.r, s.t, r_min=0.0010, k=29, a=165)
    
#     
#     
# """
# Example of MINLP
# It is recommended to read help(NLP) before
# and /examples/nlp_1.py 
# """
# from openopt import MINLP
# from numpy import *
# 
# N = 150
# K = 50
# 
# #objective function:
# f = lambda x: ((x-5.45)**2).sum()
# 
# #optional: 1st derivatives
# df = lambda x: 2*(x-5.45)
# 
# # start point
# x0 = 8*cos(arange(N))
# 
# # assign prob:
# # 1st arg - objective function
# # 2nd arg - start point
# # for more details see 
# # http://openopt.org/Assignment 
# p = MINLP(f, x0, df=df, maxIter = 1e3)
# 
# # optional: set some box constraints lb <= x <= ub
# p.lb = [-6.5]*N
# p.ub = [6.5]*N
# # see help(NLP) for handling of other constraints: 
# # Ax<=b, Aeq x = beq, c(x) <= 0, h(x) = 0
# # see also /examples/nlp_1.py
# 
# # required tolerance for smooth constraints, default 1e-6
# p.contol = 1.1e-6
# 
# p.name = 'minlp_1'
# 
# # required field: nlpSolver - should be capable of handling box-bounds at least
# #nlpSolver = 'ralg' 
# nlpSolver = 'ipopt'
# 
# # coords of discrete variables and sets of allowed values
# p.discreteVars = {7:range(3, 10), 8:range(3, 10), 9:[2, 3.1, 9]}
# 
# # required tolerance for discrete variables, default 10^-5
# p.discrtol = 1.1e-5
# 
# #optional: check derivatives, you could use p.checkdc(), p.checkdh() for constraints
# #p.checkdf()
# 
# # optional: maxTime, maxCPUTime
# # p.maxTime = 15
# # p.maxCPUTime = 15
# 
# r = p.solve('branb', nlpSolver=nlpSolver, plot = False)
# # optim point and value are r.xf and r.ff,
# # see http://openopt.org/OOFrameworkDoc#Result_structure for more details

#     