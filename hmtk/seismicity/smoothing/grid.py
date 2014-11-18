# -*- coding: utf-8 -*-
'''



'''
import collections
import numpy as np

from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.polygon import Polygon


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

