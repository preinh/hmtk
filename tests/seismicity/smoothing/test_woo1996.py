# -*- coding: utf-8 -*-
'''
Test suite for Woo smoothed seismicity class
'''
import os
import unittest
import numpy as np
#from math import fabs

from hmtk.seismicity.catalogue import Catalogue
from hmtk.seismicity.smoothing.grid import Grid

from hmtk.seismicity.smoothing.woo1996 import SmoothedSeismicity
# from hmtk.seismicity.smoothing import utils
# from hmtk.seismicity.smoothing.smoothed_seismicity import (
#     SmoothedSeismicity, _get_adjustment, Grid)
# 
# from hmtk.seismicity.smoothing.kernels.isotropic_gaussian import \
#     IsotropicGaussian


BASE_PATH = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_FILE = 'test_smoothing_csv_data.csv'
FRANKEL_OUTPUT_FILE = 'agridc_trial.out'
FRANKEL_TEST_CATALOGUE = 'test.cc'



class TestWoo1996(unittest.TestCase):
    '''
    Class to test the implementation of the smoothed seismicity algorithm
    '''
    def setUp(self):
        self.grid_limits = []
        self.model = None

    def test_instantiation(self):
        '''
        Tests the instantiation of the class
        '''
        # Test 1: Good Grid Limits
        self.grid_limits = Grid.make_from_list(
            [35.0, 40., 0.5, 40., 45.0, 0.5, 0., 40., 20.])
        expected_dict = {'beta': None,
                         'bval': None,
                         'catalogue': None,
                         'data': None,
                         'grid': None,
                         'grid_limits': {'xmax': 40.0,
                                         'xmin': 35.0,
                                         'xspc': 0.5,
                                         'ymax': 45.0,
                                         'ymin': 40.0,
                                         'yspc': 0.5,
                                         'zmax': 40.0,
                                         'zmin': 0.0,
                                         'zspc': 20.0},
                         'kernel': None,
                         'use_3d': False}

        self.model = SmoothedSeismicity(self.grid_limits)
        self.assertDictEqual(self.model.__dict__, expected_dict)
        # Test 2 - with b-value set
        self.model = SmoothedSeismicity(self.grid_limits, bvalue=1.0)
        expected_dict['bval'] = 1.0
        expected_dict['beta'] = np.log(10.)
        self.assertDictEqual(self.model.__dict__, expected_dict)

    def test_get_2d_grid(self):
        '''
        Tests the module to count the events across a grid
        '''
        self.grid_limits = Grid.make_from_list(
            [35.0, 40., 0.5, 40., 45.0, 0.5, 0., 40., 20.])
        self.model = SmoothedSeismicity(self.grid_limits, bvalue=1.0)
        # Case 1 - all events in grid (including borderline cases)
        comp_table = np.array([[1960., 4.0]])
        lons = np.arange(35.0, 41.0, 1.0)
        lats = np.arange(40.0, 46.0, 1.0)
        mags = 5.0 * np.ones(6)
        years = 2000. * np.ones(6)
        expected_result = np.zeros(100, dtype=int)
        expected_result[[9, 28, 46, 64, 82, 90]] = 1
        np.testing.assert_array_almost_equal(expected_result,
            self.model.create_2D_grid_simple(lons, lats, years, mags,
                                             comp_table))
        self.assertEqual(np.sum(expected_result), 6)

        # Case 2 - some events outside grid
        lons = np.arange(35.0, 42.0, 1.0)
        lats = np.arange(40.0, 47.0, 1.0)
        mags = 5.0 * np.ones(7)
        years = 2000. * np.ones(7)
        np.testing.assert_array_almost_equal(expected_result,
            self.model.create_2D_grid_simple(lons, lats, years, mags,
                                             comp_table))
        self.assertEqual(np.sum(expected_result), 6)


    def test_csv_writer(self):
        '''
        Short test of consistency of the csv writer
        '''

        self.grid_limits = [35.0, 40., 0.5, 40., 45.0, 0.5, 0., 40., 20.]
        self.model = SmoothedSeismicity(self.grid_limits, bvalue=1.0)
        self.model.data = np.array([[1.0, 1.0, 10.0, 4.0, 4.0, 1.0],
                                    [2.0, 2.0, 20.0, 8.0, 8.0, 1.0]])
        self.model.write_to_csv(OUTPUT_FILE)
        return_data = np.genfromtxt(OUTPUT_FILE, delimiter=',', skip_header=1)
        np.testing.assert_array_almost_equal(return_data, self.model.data)
        os.system('rm ' + OUTPUT_FILE)


    def test_analysis_Frankel_comparison(self):
        '''
        To test the run_analysis function we compare test results with those
        from Frankel's fortran implementation, under the same conditions
        '''
        self.grid_limits = [-128., -113.0, 0.2, 30., 43.0, 0.2, 0., 100., 100.]
        comp_table = np.array([[1933., 4.0],
                               [1900., 5.0],
                               [1850., 6.0],
                               [1850., 7.0]])
        config = {'Length_Limit': 3., 'BandWidth': 50., 'increment': 0.1}
        self.model = SmoothedSeismicity(self.grid_limits, bvalue=0.8)
        self.catalogue = Catalogue()
        frankel_catalogue = np.genfromtxt(os.path.join(BASE_PATH,
                                                       FRANKEL_TEST_CATALOGUE))
        self.catalogue.data['magnitude'] = frankel_catalogue[:, 0]
        self.catalogue.data['longitude'] = frankel_catalogue[:, 1]
        self.catalogue.data['latitude'] = frankel_catalogue[:, 2]
        self.catalogue.data['depth'] = frankel_catalogue[:, 3]
        self.catalogue.data['year'] = frankel_catalogue[:, 4]
        self.catalogue.end_year = 2006
        frankel_results = np.genfromtxt(os.path.join(BASE_PATH,
                                                     FRANKEL_OUTPUT_FILE))
        # Run analysis
        output_data = self.model.run_analysis(
            self.catalogue,
            config,
            completeness_table=comp_table,
            smoothing_kernel = IsotropicGaussian())

        self.assertTrue(fabs(np.sum(output_data[:, -1]) -
                             np.sum(output_data[:, -2])) < 1.0)
        self.assertTrue(fabs(np.sum(output_data[:, -1]) - 390.) < 1.0)
