# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (c) 2010-2014, GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli., L. E. Rodriguez-Abreu
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
# The software is NOT distributed as part of GEM's OpenQuake suite
# (http://www.globalquakemodel.org/openquake) and must be considered as a
# separate entity. The software provided herein is designed and implemented
# by scientific staff. It is not developed to the design standards, nor
# subject to same level of critical review by professional software
# developers, as GEM's OpenQuake software suite.
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

#!/usr/bin/env/python

'''
Module hmtk.plotting.catalogue.map is a graphical
function for plotting the spatial distribution of events
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LogNorm, Normalize
from openquake.hazardlib.geo.point import Point
from openquake.hazardlib.geo.line import Line
from openquake.hazardlib.geo.polygon import Polygon
from openquake.hazardlib.geo import utils
from hmtk.sources.area_source import mtkAreaSource
from hmtk.sources.point_source import mtkPointSource
from hmtk.sources.simple_fault_source import mtkSimpleFaultSource

DEFAULT_SYMBOLOGY = [(-np.inf, 1., 'o', 0.1), # M < 1
                     (1., 2., 'o',      0.3), # 1 < M < 2
                     (2., 3.,'o',       0.6), # 2 < M < 3
                     (3., 4.,'o',       1.8), # 3 < M < 4
                     (4., 5.,'o',      5), # 4 < M < 5
                     (5., 6.,'o',      15), # 5 < M < 6
                     (6., 7.,'o',      30), # 6 < M < 7
                     (7., 8.,'o',      70), # 7 < M < 8
                     (8., 9.,'o',     100), # 8 < M < 9
                     (9., np.inf,'o', 150)] # 9 < M < 10

LEGEND_OFFSET=(0, 0)
PORTRAIT_ASPECT = (6, 8)
LANDSCAPE_ASPECT = (8, 6)

def _fault_polygon_from_mesh(source):
    """

    """
    # Mesh
    upper_edge = np.column_stack([source.geometry.mesh.lons[1],
                                  source.geometry.mesh.lats[1],
                                  source.geometry.mesh.depths[1]])
    lower_edge = np.column_stack([source.geometry.mesh.lons[-1],
                                  source.geometry.mesh.lats[-1],
                                  source.geometry.mesh.depths[-1]])
    return np.vstack([upper_edge, np.flipud(lower_edge), upper_edge[0, :]])

class HMTKBaseMap(object):
    '''
    Class to plot the spatial distribution of events based in the Catalogue
    imported from hmtk.
    '''
    def __init__(self, config, title, dpi=300):
        """
        :param dict config:
            Configuration parameters of the algorithm, containing the
            following information -
                'min_lat' Minimum value of latitude (in degrees, float)
                'max_lat' Minimum value of longitude (in degrees, float)
                (min_lat, min_lon) Defines the inferior corner of the map

                'min_lon' Maximum value of latitude (in degrees, float)
                'max_lon' Maximum value of longitude (in degrees, float)
                (min_lon, max_lon) Defines the upper corner of the map
        :param str title:
            Title string
        """
        self.config = config
        self.title = title
        self.dpi = dpi
        self.fig = None
        self.m = None
        self._build_basemap()

    def _build_basemap(self):
        '''
        Creates the map according to the input configuration
        '''
        if self.config['min_lon'] >= self.config['max_lon']:
            raise ValueError('Upper limit of long is smaller than lower limit')

        if self.config['min_lon'] >= self.config['max_lon']:
            raise ValueError('Upper limit of long is smaller than lower limit')
        # Corners of the map
        lowcrnrlat = self.config['min_lat']
        lowcrnrlon = self.config['min_lon']
        uppcrnrlat = self.config['max_lat']
        uppcrnrlon = self.config['max_lon']
        if not 'resolution' in self.config.keys():
            self.config['resolution'] = 'l'

        lat0 = lowcrnrlat + ((uppcrnrlat - lowcrnrlat) / 2)
        lon0 = lowcrnrlon + ((uppcrnrlon - lowcrnrlon) / 2)

        if not self.config.get('embeded', False):
            if (uppcrnrlat - lowcrnrlat) >= (uppcrnrlon - lowcrnrlon):
                fig_aspect = PORTRAIT_ASPECT
            else:
                fig_aspect = LANDSCAPE_ASPECT
            self.fig = plt.figure(num=None,
                                  figsize=fig_aspect,
                                  dpi=self.dpi,
                                  facecolor='w',
                                  edgecolor='k')

        if self.title:
            plt.title(self.title, fontsize=14)
        parallels = np.arange(-90., 90., 10.)
        meridians = np.arange(0., 360., 10.)

        # Build Map
        self.m = Basemap(
            llcrnrlon=lowcrnrlon, llcrnrlat=lowcrnrlat,
            urcrnrlon=uppcrnrlon, urcrnrlat=uppcrnrlat,
            projection='cyl', resolution=self.config['resolution'],
            area_thresh=1000.0, lat_0=lat0, lon_0=lon0)
        self.m.drawmapboundary()
        self.m.drawcoastlines(color='0.9')
        self.m.drawstates(color='0.95')
        self.m.drawcountries(color='0.8')
        
        self.m.drawmeridians(meridians,labels=[1,0,0,1],linewidth=0.0, fontsize=10)
        self.m.drawparallels(parallels,labels=[1,0,0,1],linewidth=0.0, fontsize=10)
        #plt.gca().tick_params(labelsize=10)

    def savemap(self, filename, filetype='png', papertype="a4"):
        """
        Save the figure
        """
        self.fig.savefig(filename,
                         dpi=self.dpi,
                         format=filetype,
                         papertype=papertype)

    def add_catalogue(self, catalogue, overlay=False, marker='o', linewidth=1.5, alpha=0.8, **kwargs):
        '''

        :param catalogue:
            Earthquake catalogue as instance of
            :class: hmtk.seismicity.catalogue.Catalogue

        :param dict config:
            Configuration parameters of the algorithm, containing the
            following information -
                'min_lat' Minimum value of latitude (in degrees, float)
                'max_lat' Minimum value of longitude (in degrees, float)
                (min_lat, min_lon) Defines the inferior corner of the map

                'min_lon' Maximum value of latitude (in degrees, float)
                'max_lon' Maximum value of longitude (in degrees, float)
                (min_lon, max_lon) Defines the upper corner of the map

        :returns:
            Figure with the spatial distribution of the events.
        '''
        # Magnitudes bins and minimum marrker size
        #min_mag = np.min(catalogue.data['magnitude'])
        #max_mag = np.max(catalogue.data['magnitude'])
        min_loc = np.where(np.array([symb[0] for symb in DEFAULT_SYMBOLOGY]) <
                           np.min(catalogue.data['magnitude']))[0][-1]
        max_loc = np.where(np.array([symb[1] for symb in DEFAULT_SYMBOLOGY]) >
                           np.max(catalogue.data['magnitude']))[0][1]
        symbology = DEFAULT_SYMBOLOGY[min_loc:max_loc]

        _d = catalogue.data['depth']
        d_min = np.min(_d)
        d_max = np.max(_d)

        symbology = DEFAULT_SYMBOLOGY[min_loc:max_loc]
#         color = color[min_loc:max_loc]

        legend_list = []
        leg_handles = []
        for sym in symbology:
            # Create legend string
            if np.isinf(sym[0]):
                leg_str = 'M < %5.1' % sym[1]
            elif np.isinf(sym[1]):
                leg_str = '$M \leq %5.1f$' % sym[0]
            else:
                leg_str = '$%5.1f \leq M < %5.1f$' %(sym[0], sym[1])
            idx = np.logical_and(catalogue.data['magnitude'] >= sym[0],
                                 catalogue.data['magnitude'] < sym[1])
            mag_size = 1.2 * np.min([sym[0] + 0.5, sym[1] - 0.5])
            x, y = self.m(catalogue.data['longitude'][idx],
                           catalogue.data['latitude'][idx])
            self.m.scatter(x, y,
                        s = 3*sym[3],
                        c = catalogue.data['depth'][idx]+2,
                        marker=marker,
                        facecolor='none',
                        linewidth=linewidth,
                        cmap=plt.cm.get_cmap('jet_r'),
                        alpha=alpha,
                        label=leg_str,
                        vmin=d_min,
                        vmax=d_max,
                        zorder=8,
                        **kwargs)


        plt.legend(fontsize=8)
        _cb = self.m.colorbar(location='bottom',
                              extend='max',
                              pad="5%")
        _cb.ax.tick_params(labelsize=10)
        _cb.set_label("EQ Depth [km]", fontsize=10)

        if self.title:
            plt.title(self.title, fontsize=12)
        if not overlay:
            plt.show()

    def _plot_area_source(self, source, border='k-', border_width=1.0, plot_label=True, alpha=0.6):
        """
        Plots the area source
        :param source:
            Area source as instance of :class: mtkAreaSource
        :param str border:
            Line properties of border (see matplotlib documentation for detail)
        :param float border_width:
            Line width of border (see matplotlib documentation for detail)
        :param bool plot_label:
            If True, the source will labeled with source.name
        """
        if source.mfd != None:
            a_val = source.mfd.a_val
            b_val = source.mfd.b_val
            m_max = source.mfd.max_mag
        else:
            a_val = 0.1
            b_val = 1
            m_max = 10


        x, y = self.m(source.geometry.lons, source.geometry.lats)

        # repeating the start point
        x = np.hstack((x,x[0]))
        y = np.hstack((y,y[0]))

        self.m.plot(x, y, border, color='#4a789c', linewidth=border_width, zorder=8, alpha=alpha)

        if plot_label:
            bb = utils.get_spherical_bounding_box(source.geometry.lons, source.geometry.lats)
            x_0, y_0 = utils.get_middle_point(bb[0], bb[2], bb[1], bb[3])
            text = "a=%.2f, b=%.2f, mmax=%.1f"%(a_val, b_val, m_max)
#            print x_0,y_0, source.name
            _ax = plt.gca()
            if source.name=='co_norte':
                _ax.annotate(source.name, xy=(x_0,y_0+3),
                        fontsize=8,#fontweight='bold',
                        ha='center',va='center',color='#4a789c',
                        zorder=8,
                        alpha=alpha)
                _ax.annotate(text, xy=(x_0,y_0-1.2+3),
                        fontsize=8,#fontweight='bold',
                        ha='center',va='center',color='#4a789c',
                        zorder=8,
                        alpha=alpha)
            else:
                _ax.annotate(source.name, xy=(x_0,y_0),
                        fontsize=8,#fontweight='bold',
                        ha='center',va='center',color='#4a789c',
                        zorder=8,
                        alpha=alpha)
                _ax.annotate(text, xy=(x_0,y_0-1.2),
                        fontsize=8,#fontweight='bold',
                        ha='center',va='center',color='#4a789c',
                        zorder=8,
                        alpha=alpha)
        #_ax.legend()

    def _plot_point_source(self, source, point_marker='ks', point_size=2.0):
        """
        Plots the area source
        :param source:
            Area source as instance of :class: mtkPointSource
        :param str point_marker:
            Marker style for point (see matplotlib documentation for detail)
        :param float marker size for point:
            Line width of border (see matplotlib documentation for detail)
        """
        x, y = self.m(source.geometry.longitude, source.geometry.latitude)
        self.m.plot(x, y, point_marker, markersize=point_size)

    def _plot_simple_fault(self, source, border='k-', border_width=1.0):
        """
        Plots the simple fault source as a composite of the fault trace
        and the surface projection of the fault.
        :param source:
            Fault source as instance of :class: mtkSimpleFaultSource
        :param str border:
            Line properties of border (see matplotlib documentation for detail)
        :param float border_width:
            Line width of border (see matplotlib documentation for detail)
        """
        # Get the trace
        trace_lons = np.array([pnt.longitude
                               for pnt in source.fault_trace.points])
        trace_lats = np.array([pnt.latitude
                               for pnt in source.fault_trace.points])
        surface_projection = _fault_polygon_from_mesh(source)
        # Plot surface projection first
        x, y = self.m(surface_projection[:, 0], surface_projection[:, 1])
        self.m.plot(x, y, border, linewidth=border_width)
        # Plot fault trace
        x, y = self.m(trace_lons, trace_lats)
        self.m.plot(x, y, border, linewidth=1.3 * border_width)

    def add_source_model(self, model, area_border='k-', border_width=1.0,
            point_marker='ks', point_size=2.0, overlay=False):
        """
        Adds a source model to the map
        :param model:
            Source model of mixed typologies as instance of :class:
            hmtk.sources.source_model.mtkSourceModel
        """

        for source in model.sources:
            if isinstance(source, mtkAreaSource):
                self._plot_area_source(source, area_border, border_width)
            elif isinstance(source, mtkPointSource):
                self._plot_point_source(source, point_marker, point_size)
            elif isinstance(source, mtkSimpleFaultSource):
                self._plot_simple_fault(source, area_border, border_width)
            else:
                pass
        if not overlay:
            plt.show()

    def add_colour_scaled_points(self, longitude, latitude, data, shape='s',
            alpha=1.0, size=20, norm=None, overlay=False, linewidth=0.0, **kwargs):
        """
        Overlays a set of points on a map with a fixed size but colour scaled
        according to the data
        :param np.ndarray longitude:
            Longitude
        :param np.ndarray latitude:
            Latitude
        :param np.ndarray data:
            Data for plotting
        :param str shape:
            Marker style
        :param float alpha:
            Sets the transparency of the marker (0 for transparent, 1 opaque)
        :param int size:
            Marker size
        :param norm:
            Normalisation as instance of :class: matplotlib.colors.Normalize
        """
        if not norm:
            norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        x, y, = self.m(longitude, latitude)
        self.m.scatter(x, y,
                       marker=shape,
                       s=size,
                       c=data,
                       norm=norm,
                       alpha=alpha,
                       linewidth=linewidth,
                       zorder=4,
                       **kwargs)

        _cb = self.m.colorbar(pad="10%")
        _cb.ax.tick_params(labelsize='small')
        _cb.set_label("cluster #", fontsize='small')

        if not overlay:
            plt.show()

    def add_size_scaled_points(self, longitude, latitude, data, shape='o',
            logplot=False, alpha=1.0, colour='b', smin=2.0, sscale=2.0,
            overlay=False, **kwargs):
        """
        Plots a set of points with size scaled according to the data
        :param bool logplot:
            Choose to scale according to the logarithm (base 10) of the data
        :param float smin:
            Minimum scale size
        :param float sscale:
            Scaling factor
        """
        if logplot:
            data = np.log10(data.copy())

        x, y, = self.m(longitude, latitude)
        self.m.scatter(x, y,
                       marker=shape,
                       s=(smin + data ** sscale),
                       c=colour,
                       alpha=alpha,
                       zorder=4,
                       **kwargs)
        if not overlay:
            plt.show()
