#!/usr/bin/env/python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
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
Module :mod: hmtk.seismicity.smoothing.utils implements utility functions for
smoothed seismicity analysis
'''
import collections

import numpy as np
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon

def hermann_adjustment_factors(bval, min_mag, mag_inc):
    '''
    Returns the adjustment factors (fval, fival) proposed by
    Hermann (1978)
    :param float bval:
        Gutenberg & Richter (1944) b-value

    :param np.ndarray min_mag:
        Minimum magnitude of completeness table

    :param non-negative float mag_inc:
        Magnitude increment of the completeness table
    '''

    fval = 10. ** (bval * min_mag)
    fival = 10. ** (bval * (mag_inc / 2.)) - 10. ** (-bval * (mag_inc / 2.))
    return fval, fival


def incremental_a_value(bval, min_mag, mag_inc):
    '''
    Incremental a-value from cumulative - using the version of the
    Hermann (1979) formula described in Wesson et al. (2003)

    :param float bval:
        Gutenberg & Richter (1944) b-value

    :param np.ndarray min_mag:
        Minimum magnitude of completeness table

    :param float mag_inc:
        Magnitude increment of the completeness table
    '''
    a_cum = 10. ** (bval * min_mag)
    a_inc = a_cum + np.log10((10. ** (bval * mag_inc)) -
                             (10. ** (-bval * mag_inc)))

    return a_inc


def get_weichert_factor(beta, cmag, cyear, end_year):
    '''
    Gets the Weichert adjustment factor for each the magnitude bins

    :param float beta:
        Beta value of Gutenberg & Richter parameter (b * log(10.))

    :param np.ndarray cmag:
        Magnitude values of the completeness table

    :param np.ndarray cyear:
        Year values of the completeness table

    :param float end_year:
        Last year for consideration in the catalogue

    :returns:
        Weichert adjustment factor (float)
    '''

    if len(cmag) > 1:
        # cval corresponds to the mid-point of the completeness bins
        # In the original code it requires that the magnitude bins be
        # equal sizedclass IsotropicGaussian(BaseSmoothingKernel):
        dmag = (cmag[1:] + cmag[:-1]) / 2.
        cval = np.hstack([dmag, cmag[-1] + (dmag[-1] - cmag[-2])])
    else:
        # Single completeness value so Weichert factor is unity
        return 1.0, None

    t_f = sum(np.exp(-beta * cval)) / sum((end_year - cyear + 1) *
                                          np.exp(-beta * cval))
    return t_f, cval


def check_completeness_table(completeness_table, catalogue):
    '''
    Check to ensure completeness table is in the correct format
    completeness_table = np.array([[year_, mag_i]]) for i in number of bins

    :param np.ndarray completeness_table:
        Completeness table in format [[year, mag]]

    :param catalogue:
        Instance of hmtk.seismicity.catalogue.Catalogue class

    :returns:
        Correct completeness table

    '''

    if isinstance(completeness_table, np.ndarray):
        assert np.shape(completeness_table)[1] == 2
        return completeness_table
    elif isinstance(completeness_table, list):
        # Assuming list has only two elements
        assert len(completeness_table) == 2
        return np.array([[completeness_table[0], completeness_table[1]]])
    else:
        # Accepts the minimum magnitude and earliest year of the catalogue
        return np.array([[np.min(catalogue.data['year']),
                          np.min(catalogue.data['magnitude'])]])


def get_even_magnitude_completeness(completeness_table, 
                                    catalogue=None,
                                    magnitude_increment=0.1):
    '''
    To make the magnitudes evenly spaced, render to a constant 0.1
    magnitude unit
    :param np.ndarray completeness_table:
    Completeness table in format [[year, mag]]
    :param catalogue:
    Instance of hmtk.seismicity.catalogue.Catalogue class
    :returns:
    Correct completeness table
    '''
    mmax = np.floor(10. * np.max(catalogue.data['magnitude'])) / 10.
    
    check_completeness_table(completeness_table, catalogue)
    
    cmag = np.hstack([completeness_table[:, 1], mmax + magnitude_increment])
    cyear = np.hstack([completeness_table[:, 0], completeness_table[-1, 0]])
    
    if np.shape(completeness_table)[0] == 1:
        # Simple single-valued table
        return completeness_table, 0.1
    
    for iloc in range(0, len(cmag) - 1):
        mrange = np.arange(np.floor(10. * cmag[iloc]) / 10.,
                           (np.ceil(10. * cmag[iloc + 1]) / 10.),
                           magnitude_increment)
        _ones = np.ones(len(mrange), dtype=float)
        temp_table = np.column_stack([cyear[iloc] * _ones, mrange])
        if iloc == 0:
            completeness_table = np.copy(temp_table)
        else:
            completeness_table = np.vstack([completeness_table,
                                            temp_table])
    #completeness_table = np.vstack([completeness_table,å
    # np.array([[cyear[-1], cmag[-1]]])])
    return completeness_table, magnitude_increment

# def get_even_magnitude_completeness(completeness_table, 
#                                     catalogue=None, 
#                                     magnitude_increment=0.1):
#     '''
#     To make the magnitudes evenly spaced, render to a constant 'magnitude_increment'
#     magnitude unit
# 
#     :param np.ndarray completeness_table:
#         Completeness table in format [[year, mag]]
# 
#     :param catalogue:
#         Instance of hmtk.seismicity.catalogue.Catalogue class
# 
#     :param float magnitude_increment:
#         Magnitude increment value, default 0.1
# 
#     :returns:
#         Correct completeness table
# 
#     '''
#     mmax = np.floor(10. * np.max(catalogue.data['magnitude'])) / 10.
#     check_completeness_table(completeness_table, catalogue)
#     cmag = np.hstack([completeness_table[:, 1], mmax + magnitude_increment])
#     cyear = np.hstack([completeness_table[:, 0], completeness_table[-1, 0]])
#     if np.shape(completeness_table)[0] == 1:
#         # Simple single-valued table
#         return completeness_table, magnitude_increment
# 
#     for iloc in range(0, len(cmag) - 1):
#         mrange = np.arange(np.floor(10. * cmag[iloc]) / 10.,
#                            (np.ceil(10. * cmag[iloc + 1]) / 10.),
#                            magnitude_increment)
#         temp_table = np.column_stack([
#             cyear[iloc] * np.ones(len(mrange), dtype=float),
#             mrange])
#         if iloc == 0:
#             completeness_table = np.copy(temp_table)
#         else:
#             completeness_table = np.vstack([completeness_table,
#                                             temp_table])
#     #completeness_table = np.vstack([completeness_table,
#     #    np.array([[cyear[-1], cmag[-1]]])])
#     return completeness_table, magnitude_increment


class Grid(collections.OrderedDict):
    @classmethod
    def make_from_list(cls, grid_limits):
        new = cls()
        new.update({'xmin': grid_limits[0],
                    'xmax': grid_limits[1],
                    'xspc': grid_limits[2],
                    'ymin': grid_limits[3],
                    'ymax': grid_limits[4],
                    'yspc': grid_limits[5],
                    'zmin': grid_limits[6],
                    'zmax': grid_limits[7],
                    'zspc': grid_limits[8]})
        return new

    @classmethod
    def make_from_catalogue(cls, catalogue, spacing, dilate):
        '''
        Defines the grid on the basis of the catalogue
        '''
        new = cls()
        cat_bbox = get_catalogue_bounding_polygon(catalogue)

        if dilate > 0:
            cat_bbox = cat_bbox.dilate(dilate)

        # Define Grid spacing
        new.update({'xmin': np.min(cat_bbox.lons),
                    'xmax': np.max(cat_bbox.lons),
                    'xspc': spacing,
                    'ymin': np.min(cat_bbox.lats),
                    'ymax': np.max(cat_bbox.lats),
                    'yspc': spacing,
                    'zmin': 0.,
                    'zmax': np.max(catalogue.data['depth']),
                    'zspc': np.max(catalogue.data['depth'])})

        if new['zmin'] == new['zmax'] == new['zspc'] == 0:
            new['zmax'] = new['zspc'] = 1

        return new

    def as_list(self):
        return [self['xmin'], self['xmax'], self['xspc'],
                self['ymin'], self['ymax'], self['yspc'],
                self['zmin'], self['zmax'], self['zspc']]

    def as_polygon(self):
        return Polygon([
            Point(self['xmin'], self['ymax']),
            Point(self['xmax'], self['ymax']),
            Point(self['xmax'], self['ymin']),
            Point(self['xmin'], self['ymin'])])

    def dilate(self, width):
        polygon = self.as_polygon().dilate(width)

        self.update({'xmin': np.min(polygon.lons),
                     'xmax': np.max(polygon.lons),
                     'ymin': np.min(polygon.lats),
                     'ymax': np.max(polygon.lats)})
        return self


def get_completeness_adjustment(mag, year, mmin, completeness_year, t_f, mag_inc=0.1):
    '''
    If the magnitude is greater than the minimum in the completeness table
    and the year is greater than the corresponding completeness year then
    return the Weichert factor

    :param float mag:
        Magnitude of an earthquake

    :param float year:
        Year of earthquake

    :param np.ndarray completeness_table:
        Completeness table

    :param float mag_inc:
        Magnitude increment

    :param float t_f:
        Weichert adjustment factor

    :returns:
        Weichert adjustment factor is event is in complete part of catalogue
        (0.0 otherwise)
    '''
    if len(completeness_year) == 1:
        if mag >= mmin:
            # No adjustment needed - event weight == 1
            return 1.0
        else:
            # Event should not be counted
            return False

    kval = int(((mag - mmin) / mag_inc)) + 1

    if (kval >= 1) and (year >= completeness_year[kval - 1]):
        return t_f
    else:
        return False


def get_catalogue_bounding_polygon(catalogue):
    '''
    Returns a polygon containing the bounding box of the catalogue
    '''
    upper_lon = np.max(catalogue.data['longitude'])
    upper_lat = np.max(catalogue.data['latitude'])
    lower_lon = np.min(catalogue.data['longitude'])
    lower_lat = np.min(catalogue.data['latitude'])

    return Polygon([Point(lower_lon, upper_lat), Point(upper_lon, upper_lat),
                    Point(upper_lon, lower_lat), Point(lower_lon, lower_lat)])


