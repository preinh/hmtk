#!/usr/bin/env/python

"""
Collection of tools for plotting descriptive statistics of a catalogue
"""
import os
from datetime import date
#from dateutil import parser#, relativedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from math import log10

# Default the figure size
DEFAULT_SIZE = (8., 6.)


def build_filename(filename, filetype='png', resolution=300):
    """
    Uses the input properties to create the string of the filename
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    filevals = os.path.splitext(filename)
    if filevals[1]:
        filetype = filevals[1][1:]
    if not filetype:
        filetype = 'png'

    filename = filevals[0] + '.' + filetype

    if not resolution:
        resolution = 300
    return filename, filetype, resolution

def _save_image(filename, filetype='png', resolution=300):
    """
    If filename is specified, saves the image
    :param str filename:
        Name of the file
    :param str filetype:
        Type of file
    :param int resolution:
        DPI resolution of the output figure
    """
    if filename:
        filename, filetype, resolution = build_filename(filename,
                                                        filetype,
                                                        resolution)
        plt.savefig(filename, dpi=resolution, format=filetype)
    else:
        pass
    return

def _get_catalogue_bin_limits(catalogue, dmag):
    """
    Returns the magnitude bins corresponing to the catalogue
    """
    mag_bins = np.arange(
        float(np.floor(np.min(catalogue.data['magnitude']))) - dmag,
        float(np.ceil(np.max(catalogue.data['magnitude']))) + dmag,
        dmag)
    counter = np.histogram(catalogue.data['magnitude'], mag_bins)[0]
    idx = np.where(counter > 0)[0]
    mag_bins = mag_bins[idx[0]:idx[-1] + 3]
    return mag_bins

def plot_depth_histogram(catalogue, bin_width,  normalisation=False,
        bootstrap=None, filename=None, filetype='png', dpi=300, figsize=DEFAULT_SIZE, **kwargs):
    """
    Creates a histogram of the depths in the catalogue
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float bin_width:
        Width of the histogram for the depth bins
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample depth uncertainty choose number of samples
    """
    plt.figure(figsize=figsize)
    # Create depth range
    if len(catalogue.data['depth']) == 0:
        raise ValueError('No depths reported in catalogue!')
    depth_bins = np.arange(0.,
                           np.max(catalogue.data['depth']) + bin_width,
                           bin_width)
    depth_hist = catalogue.get_depth_distribution(depth_bins,
                                                  normalisation,
                                                  bootstrap)
    plt.barh(depth_bins[:-1],
            depth_hist,
            height=0.95 * bin_width,
#            edgecolor='k',
#            orientation='horizontal',
#             color='#5fbdce',
#             alpha=0.6,
            **kwargs)

    plt.gca().invert_yaxis()

    plt.ylabel('Depth (km)', fontsize='medium')
    if normalisation:
        plt.xlabel('Probability Density Function', fontsize='medium')
    else:
        plt.xlabel('Count', fontsize='medium')
    plt.title('Depth Histogram', fontsize='large')

    _save_image(filename, filetype, dpi)
    plt.show()
    return

def plot_weekday_histogram(catalogue, normalisation=True,
        filename=None, filetype='png', dpi=300, figsize=DEFAULT_SIZE,
        **kwargs):
    """
    Creates a histogram of the depths in the catalogue
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float bin_width:
        Width of the histogram for the depth bins
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample depth uncertainty choose number of samples
    """
    plt.figure(figsize=figsize)

#     print min(catalogue.data['day']), max(catalogue.data['day'])
#     print min(catalogue.data['month']), max(catalogue.data['month'])

    t = zip(catalogue.data['year'],  catalogue.data['month'], catalogue.data['day'])
    d = [ date(t[0], t[1], t[2]).isoweekday() for t in t ]

    bins = np.arange(0.5, 7 +1, 1)

    plt.hist(d, bins=bins, normed=normalisation, **kwargs)
    plt.hlines((1/7.),xmin=0, xmax=8, linestyles='dashed', **kwargs)

    plt.xlabel('Weekdays (monday=1)', fontsize='medium')
    plt.xlim(.5, 7.5)
    if normalisation:
        plt.ylabel('Probability Density Function', fontsize='medium')
    else:
        plt.ylabel('Count')
    plt.title('Weekdays Histogram', fontsize='large')

    _save_image(filename, filetype, dpi)
    plt.show()
    return


def plot_hour_histogram(catalogue, normalisation=True,
        filename=None, filetype='png', dpi=300, figsize=DEFAULT_SIZE,
        **kwargs):
    """
    Creates a histogram of the depths in the catalogue
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float bin_width:
        Width of the histogram for the depth bins
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample depth uncertainty choose number of samples
    """
    plt.figure(figsize=figsize)

    bins = np.arange(0.5, 24, 1)
    plt.hist(catalogue.data['hour'], bins=bins, normed=normalisation, **kwargs)
    plt.hlines((1/24.),xmin=0, xmax=24, linestyles='dashed', **kwargs)

    plt.xlabel('Hour of Day', fontsize='medium')
    plt.xlim(.5,23.5)
    if normalisation:
        plt.ylabel('Probability Density Function', fontsize='medium')
    else:
        plt.ylabel('Count', fontsize='medium')
    plt.title('Earthquake Hours Histogram', fontsize='large')

    _save_image(filename, filetype, dpi)
    plt.show()
    return


def plot_rate(catalogue, normalisation = False, cumulative=False,
              new_figure=True, overlay=False,
              filename=None, filetype='png', dpi=300, figsize=DEFAULT_SIZE,
              **kwargs):
    """
    Creates a histogram of the depths in the catalogue
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float bin_width:
        Width of the histogram for the depth bins
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample depth uncertainty choose number of samples
    """
    if new_figure:
        plt.figure(figsize=figsize)

    Y = catalogue.data['year']
    max_year = np.max(Y)
    min_year = np.min(Y)


    bins = np.arange(min_year - .5, max_year + 1.5, 1)

    plt.hist(catalogue.data['year'], bins=bins,
             histtype='step',
             normed = normalisation,
             cumulative = cumulative,
             **kwargs)

#    plt.plot(_e, _h, marker=None, linestyle='-', **kwargs)

    plt.xlim(min_year, max_year)

    plt.xlabel('Year', fontsize='medium')
    #plt.xlim(.5,23.5)
    if normalisation:
        plt.ylabel('Percent', fontsize='medium')
    else:
        plt.ylabel('Count', fontsize='medium')
    plt.title('Earthquakes Records', fontsize='large')

    _save_image(filename, filetype, dpi)
    if not overlay:
        plt.show()

    return



def plot_magnitude_depth_density(catalogue, mag_int, depth_int, logscale=False,
        normalisation=False, bootstrap=None, filename=None, filetype='png',
        dpi=300):
    """
    Creates a density plot of the magnitude and depth distribution
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float mag_int:
        Width of the histogram for the magnitude bins
    :param float depth_int:
        Width of the histogram for the depth bins
    :param bool logscale:
        Choose to scale the colours in a log-scale (True) or linear (False)
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample magnitude and depth uncertainties choose number of samples
    """
    if len(catalogue.data['depth']) == 0:
        raise ValueError('No depths reported in catalogue!')
    depth_bins = np.arange(0.,
                           np.max(catalogue.data['depth']) + depth_int,
                           depth_int)
    mag_bins = _get_catalogue_bin_limits(catalogue, mag_int)
    mag_depth_dist = catalogue.get_magnitude_depth_distribution(mag_bins,
                                                                depth_bins,
                                                                normalisation,
                                                                bootstrap)
    vmin_val = np.min(mag_depth_dist[mag_depth_dist > 0.])
    # Create plot
    if logscale:
        normaliser = LogNorm(vmin=vmin_val, vmax=np.max(mag_depth_dist))
    else:
        normaliser = Normalize(vmin=0, vmax=np.max(mag_depth_dist))
    plt.figure(figsize=DEFAULT_SIZE)
    plt.pcolor(mag_bins[:-1],
               depth_bins[:-1],
               mag_depth_dist.T,
               norm=normaliser)
    plt.xlabel('Magnitude', fontsize='medium')
    plt.ylabel('Depth (km)', fontsize='madium')
    plt.xlim(mag_bins[0], mag_bins[-1])
    plt.ylim(depth_bins[0], depth_bins[-1])
    plt.colorbar()
    if normalisation:
        plt.title('Magnitude-Depth Density', fontsize='large')
    else:
        plt.title('Magnitude-Depth Count', fontsize='large')

    _save_image(filename, filetype, dpi)
    plt.show()
    return

def plot_magnitude_time_scatter(catalogue, plot_error=False, filename=None,
        filetype='png', dpi=300, figsize=DEFAULT_SIZE, fmt_string='+',
        overlay=False, completeness_table=None, **kwargs):
    """
    Creates a simple scatter plot of magnitude with time
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param bool plot_error:
        Choose to plot error bars (True) or not (False)
    :param str fmt_string:
        Symbology of plot
    """
    if not overlay:
        plt.figure(figsize=figsize)

    dtime = catalogue.get_decimal_time()
    if len(catalogue.data['sigmaMagnitude']) == 0:
        print 'Magnitude Error is missing - neglecting error bars!'
        plot_error = False

    if plot_error:
        plt.errorbar(dtime,
                     catalogue.data['magnitude'],
                     xerr=None,
                     yerr=catalogue.data['sigmaMagnitude'],
                     fmt=fmt_string,
                     **kwargs)
    else:
        plt.scatter(dtime, catalogue.data['magnitude'], **kwargs)

    if not completeness_table is None:
        #plt.step(completeness_table[1:,0], completeness_table[:-1,1], linewidth=2)
        plt.step(completeness_table[:,0], completeness_table[:,1], where='post', linewidth=2)

    plt.xlabel('Year', fontsize='medium')
    plt.ylabel('Magnitude', fontsize='medium')
    plt.title('Magnitude-Time distribution', fontsize='large')

    _save_image(filename, filetype, dpi)
    if not overlay:
        plt.show()
    return

def plot_magnitude_time_density(catalogue, mag_int, time_int,
        normalisation=False, bootstrap=None, filename=None, filetype='png',
        dpi=300, figsize=DEFAULT_SIZE):
    """
    Creates a plot of magnitude-time density
    :param catalogue:
        Earthquake catalogue as instance of :class:
        hmtk.seismicity.catalogue.Catalogue
    :param float mag_int:
        Width of the histogram for the magnitude bins
    :param float time_int:
        Width of the histogram for the time bin (in decimal years)
    :param bool normalisation:
        Normalise the histogram to give output as PMF (True) or count (False)
    :param int bootstrap:
        To sample magnitude and depth uncertainties choose number of samples
    """
    plt.figure(figsize=figsize)
    # Create the magnitude bins
    if isinstance(mag_int, np.ndarray) or isinstance(mag_int, list):
        mag_bins = mag_int
    else:
        mag_bins = np.arange(
            np.min(catalogue.data['magnitude']),
            np.max(catalogue.data['magnitude']) + mag_int / 2.,
            mag_int)
    # Creates the time bins
    if isinstance(time_int, np.ndarray) or isinstance(time_int, list):
        time_bins = time_int
    else:
        time_bins = np.arange(
            float(np.min(catalogue.data['year'])),
            float(np.max(catalogue.data['year'])) + 1.,
            float(time_int))
    # Get magnitude-time distribution
    mag_time_dist = catalogue.get_magnitude_time_distribution(
        mag_bins,
        time_bins,
        normalisation,
        bootstrap)
    # Get smallest non-zero value
    vmin_val = np.min(mag_time_dist[mag_time_dist > 0.])
    # Create plot
    plt.pcolor(time_bins[:-1],
               mag_bins[:-1],
               mag_time_dist.T,
               norm=LogNorm(vmin=vmin_val, vmax=np.max(mag_time_dist)),
               cmap = 'YlOrRd',
               )
    plt.xlabel('Time (year)', fontsize='medium')
    plt.ylabel('Magnitude', fontsize='medium')
    plt.xlim(time_bins[0], time_bins[-1])
    plt.colorbar()
    if normalisation:
        plt.title('Magnitude-Time Density', fontsize='large')
    else:
        plt.title('Magnitude-Time Count', fontsize='large')

    _save_image(filename, filetype, dpi)
    plt.show()
    return

def get_completeness_adjusted_table(catalogue, completeness, dmag, end_year):
    """
<<<<<<< HEAD
    Counts the number of eq
=======
    Counts the number of earthquakes in each magnitude bin and normalises
    the rate to annual rates, taking into account the completeness
>>>>>>> master
    """
    inc = 1E-7
    # Find the natural bin limits
    mag_bins = _get_catalogue_bin_limits(catalogue, dmag)
    obs_time = end_year - completeness[:, 0] + 1.
    obs_rates = np.zeros_like(mag_bins)
    n_comp = np.shape(completeness)[0]
    for iloc in range(0, n_comp, 1):
        low_mag = completeness[iloc, 1]
        comp_year = completeness[iloc, 0]
        if iloc == n_comp - 1:
            idx = np.logical_and(
                catalogue.data['magnitude'] >= low_mag - (dmag / 2.),
                catalogue.data['year'] >= comp_year)
            high_mag = mag_bins[-1] + dmag
            obs_idx = mag_bins >= (low_mag - dmag / 2.)
        else:
            high_mag = completeness[iloc + 1, 1]
            mag_idx = np.logical_and(
                catalogue.data['magnitude'] >= low_mag - (dmag / 2.),
                catalogue.data['magnitude'] < high_mag)

            idx = np.logical_and(mag_idx,
                                 catalogue.data['year'] >= comp_year - inc)
            obs_idx = np.logical_and(mag_bins >= low_mag,
                                     mag_bins < high_mag + dmag)


        temp_rates = np.histogram(catalogue.data['magnitude'][idx],
                                  mag_bins[obs_idx])[0]
        temp_rates = temp_rates.astype(float) / obs_time[iloc]
        if iloc == n_comp - 1:
            obs_rates[np.where(obs_idx)[0]] = temp_rates
        else:
            obs_rates[obs_idx[:-1]] = temp_rates

    selector = np.where(obs_rates > 0.)[0]
    mag_bins = mag_bins[selector[0]:selector[-1]]
    obs_rates = obs_rates[selector[0]:selector[-1]]

    # Get cumulative rates
    cum_rates = np.array([sum(obs_rates[iloc:])
                                for iloc in range(0, len(obs_rates))])
    out_idx = cum_rates > 0.
    return np.column_stack([mag_bins[out_idx],
                            obs_rates[out_idx],
                            cum_rates[out_idx],
                            np.log10(cum_rates[out_idx])])

def plot_observed_recurrence(catalogue, completeness, dmag, end_year=None,
        filename=None, filetype='png',
        title="Recurrence [#eq/year]", dpi=300, figsize=DEFAULT_SIZE,
        overlay=False, color=['b','r'], **kwargs):
    """
    Plots the observed recurrence taking into account the completeness
    """
    # Get completeness adjusted recurrence table
    if isinstance(completeness, float):
        # Unique completeness
        completeness = np.array([[np.min(catalogue.data['year']),
                                  completeness]])
    if not end_year:
        end_year = np.max(catalogue.data['year'])
    recurrence = get_completeness_adjusted_table(catalogue,
                                                 completeness,
                                                 dmag,
                                                 end_year)
    if not overlay:
        plt.figure(figsize=figsize)

    plt.semilogy(recurrence[:, 0], recurrence[:, 1], '<', color=color[0], label='Incremental', **kwargs)
    plt.semilogy(recurrence[:, 0], recurrence[:, 2], '>', color=color[1], label='Cumulative', **kwargs)
    #plt.xlim(0., max(recurrence[:, 0] + 1))
    plt.xlim(min(recurrence[:, 0] - 0.5), max(recurrence[:, 0] + 0.5))
    #plt.ylim(1e-4, 1e10)
    #plt.ylim(1e-2, 1e2)
    plt.xlabel('Magnitude', fontsize='small')
    plt.ylabel('Annual Rate', fontsize='small')
    plt.legend(fontsize='small')
    plt.title(title, fontsize='medium')

    _save_image(filename, filetype, dpi)
    if not overlay:
        plt.show()
