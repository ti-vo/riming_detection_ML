import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from numba import jit
import matplotlib
import datetime

# convert to celsius
def toC(datalist):
    return datalist[0]['var']-273.15, datalist[0]['mask']
def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def ts_to_dt(ts):
    """unix timestamp to dt"""
    return datetime.datetime.utcfromtimestamp(ts)

def datetime64_to_unixtime(datetime64):
    """wrong results"""
    b = (datetime64.astype('uint64')/1e6).astype('uint32')
    return b

def crop_to_timestamp(dict, ts):
    t_ind = np.where(np.isin(dict['ts'], ts))[0]
    out = copy.deepcopy(dict)
    out['var'] = out['var'][t_ind, :]
    out['mask'] = out['mask'][t_ind, :]
    out['ts'] = out['ts'][t_ind]
    return out

@jit(nopython=True, fastmath=True)
def radar_moment_calculation(signal, vel_bins):
    """
    Based on Willi's SpectraProcessing routine for LIMRAD94
    Calculation of radar moments: reflectivity, mean Doppler velocity, spectral width,
        skewness, and kurtosis of one Doppler spectrum. Optimized for the use of Numba.

    Args:
        - signal (float array): detected signal from a Doppler spectrum in linear units, not normalized by velocity
        - vel_bins (float array): extracted velocity bins of the signal (same length as signal)

    Returns:
        dict containing

            - Ze_lin (float array): reflectivity (0.Mom) over range of velocity bins [mm6/m3]
            - VEL (float array): mean velocity (1.Mom) over range of velocity bins [m/s]
            - sw (float array):: spectrum width (2.Mom) over range of velocity bins [m/s]
            - skew (float array):: skewness (3.Mom) over range of velocity bins
            - kurt (float array):: kurtosis (4.Mom) over range of velocity bins
    """

    Ze_lin = np.sum(signal)  # linear full spectrum Ze [mm^6/m^3], scalar
    pwr_nrm = signal / Ze_lin  # determine normalized power (NOT normalized by Vdop bins)

    vel = np.sum(vel_bins * pwr_nrm)
    vel_diff = vel_bins - vel
    vel_diff2 = vel_diff * vel_diff
    sw = np.sqrt(np.abs(np.sum(pwr_nrm * vel_diff2)))
    sw2 = sw * sw
    skew = np.sum(pwr_nrm * vel_diff * vel_diff2 / (sw * sw2))
    kurt = np.sum(pwr_nrm * vel_diff2 * vel_diff2 / (sw2 * sw2))
#    VEL = VEL - DoppRes / 2.0

    return Ze_lin, vel, sw, skew, kurt

@jit(nopython=True, fastmath=True)
def estimate_noise_hs74(spectrum, navg=1, std_div=6.0, nnoise_min=1):
    """REFERENCE TO ARM PYART GITHUB REPO: https://github.com/ARM-DOE/pyart/blob/master/pyart/util/hildebrand_sekhon.py

    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.
    Args:
        spectrum (array): Doppler spectrum in linear units.
        navg (int, optional):  The number of spectral bins over which a moving average has been
            taken. Corresponds to the **p** variable from equation 9 of the
            article. The default value of 1 is appropriate when no moving
            average has been applied to the spectrum.
        std_div (float, optional): Number of standard deviations above mean noise floor to specify the
            signal threshold, default: threshold=mean_noise + 6*std(mean_noise)
        nnoise_min (int, optional): Minimum number of noise samples to consider the estimation valid.

    Returns:
        mean (float): Mean of points in the spectrum identified as noise.
        threshold (float): Threshold separating noise from signal. The point in the spectrum with
            this value or below should be considered as noise, above this value
            signal. It is possible that all points in the spectrum are identified
            as noise. If a peak is required for moment calculation then the point
            with this value should be considered as signal.
        var (float): Variance of the points in the spectrum identified as noise.
        nnoise (int): Number of noise points in the spectrum.
    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.
    """
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)  # default to all points in the spectrum as noise

    rtest = 1 + 1 / navg
    sum1 = 0.
    sum2 = 0.
    for i, pwr in enumerate(sorted_spectrum):
        npts = i + 1
        if npts < nnoise_min:
            continue

        sum1 += pwr
        sum2 += pwr * pwr

        if npts * sum2 < sum1 * sum1 * rtest:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise.
            sum1 -= pwr
            sum2 -= pwr * pwr
            break

    mean = sum1 / nnoise
    var = sum2 / nnoise - mean * mean

    threshold = mean + np.sqrt(var) * std_div

    return mean, threshold, var, nnoise

def compute_width(spectra, **kwargs):
    """
    wrapper for width_fast; compute Doppler spectrum edge width
    Args:
        spectra: larda dictionary of spectra
        **kwargs: To be passed on to self.width_fast:
            thresh_1, thresh_2
    Returns:

    """
    widths_array = width_fast(spectra, **kwargs)
    vel_res = abs(np.median(np.diff(spectra['vel'])))
    width = vel_res*widths_array
    np.putmask(width, (width < 0), np.nan)

    return width


def width_fast(spectra, **kwargs):
    """
    implementation without loops; compute Doppler spectrum edge width

    Args:
        spectra (np.ndarray): 3D array of time, height, velocity

    Returns:
        2D array of widths (measured in number of bins, has to be multiplied by Doppler velocity resolution)

    """
    if spectra.__class__ == dict:
        spectra = spectra['var']
    # define the threshold above which edges are found as the larger one of either
    # 0.05% of the peak reflectivity, or the minimum of the spectrum + 6 dBZ
    thresh_1 = 6 if not 'thresh1' in kwargs else kwargs['thresh1']
    thresh_2 = 0.0005 if not 'thresh2' in kwargs else kwargs['thresh2']
    bin_number = spectra.shape[2]

    thresh = np.maximum(thresh_2*np.nanmax(spectra, axis=2), np.nanmin(spectra, axis=2) * 10**(thresh_1/10))

    # find the first bin where spectrum is larger than threshold
    first = np.argmax(spectra > np.repeat(thresh[:, :, np.newaxis], bin_number, axis=2), axis=2)
    # same with reversed view of spectra
    last = bin_number - np.argmax(spectra[:, :, ::-1] > np.repeat(thresh[:, :, np.newaxis], bin_number, axis=2),
                                  axis=2)
    width = last - first
    np.putmask(width, width == bin_number, -9999)
    return width


def air_density_correction(MDV_container, pressure="standard", temperature="standard"):
    """"correcting fall velocities with respect to air density.
    Per default, the standard atmosphere is used for correction, but for pressure, also a vector containing
    p values can be supplied.
    Args
       MDV_container: larda data container, "var" is array of fall velocities to be corrected and "rg" is range in m
        **pressure: if set to "standard" use standard atmospheric pressure gradient to correct MDV
         """
    g = 9.80665  # gravitational acceleration
    R = 287.058  # specific gas constant of dry air

    def get_density(pressure, temperature):
        R = 287.058
        density = pressure / (R * temperature)
        return density

    def cal(p0, t0, L, h0, h1):
        if L != 0:
            t1 = t0 + L * (h1 - h0)
            p1 = p0 * (t1 / t0) ** (-g / L / R)
        else:
            t1 = t0
            p1 = p0 * math.exp(-g / R / t0 * (h1 - h0))
        return t1, p1

    def isa(altitude):
        """international standard atmosphere
        numbers from https://en.wikipedia.org/wiki/International_Standard_Atmosphere
        code: https://gist.github.com/buzzerrookie/5b6438c603eabf13d07e"""

        L = [-0.0065, 0]  # Lapse rate in troposphere, tropopause
        h = [11000, 20000]  # height of troposphere, tropopause
        p0 = 108900  # base pressure
        t0 = 292.15  # base temperature, 19 degree celsius
        prevh = -611
        if altitude < 0 or altitude > 20000:
            AssertionError("altitude must be in [0, 20000]")
        for i in range(0, 2):
            if altitude <= h[i]:
                temperature, pressure = cal(p0, t0, L[i], prevh, altitude)
            else:
                t0, p0 = cal(p0, t0, L[i], prevh, h[i])
                prevh = h[i]
        density = get_density(pressure, temperature)
        return pressure, density, temperature

    if pressure == "standard" and temperature == "standard":
        d = [isa(range)[1] for range in list(MDV_container['rg'])]

    elif pressure == "standard" and temperature != "standard":
        p = [isa(range)[0] for range in list(MDV_container['rg'])]
        t = temperature
        d = [get_density(pi, ti) for pi, ti in (p, t)]

    elif temperature == "standard" and pressure != "standard":
        t = [isa(range)[2] for range in list(MDV_container['rg'])]
        p = pressure
        d = [get_density(pi, ti) for pi, ti in (p, t)]

    # Vogel and Fabry, 2018, Eq. 1
    rho_0 = 1.2  # kg per m^3
    corr_fac = [np.sqrt(di / rho_0) for di in d]
    MDV_corr = MDV_container['var'] * np.asarray(corr_fac)

    return MDV_corr


def est_edr(sigma_T, theta=0.5, R=3000, wl=8e-3, C=0.5, U=10, t=2):
    L_s = U*t + 2* R* np.sin((theta/360)*2*np.pi)
    L_l = wl/2
    epsilon = 2*np.pi * (2/3 * 1/C * sigma_T**2 * (1/ (L_s**(2/3) - L_l**(2/3))))**(3/2)
    return epsilon


def get_target_classification(categorize_bits):
    """
    Function copied from cloudnetpy to get classification from cloudnet categorize bits given in
     lv1 netcdf files
    :param categorize_bits: dictionary containing category and quality bits (ndarrays)
    :return: classification
    """
    bits = categorize_bits["category_bits"]['var']
    clutter = categorize_bits["quality_bits"]['var'] == 2
    classification = np.zeros(bits.shape, dtype=int)

    classification[(bits == 1)] = 1 # 1 liquid cloud droplets only
    classification[(bits == 2)] = 2 # isbit(2, 1) drizzle or rain - falling bit is bit 1
    classification[bits == 3] = 3 # 0+1 isbit(3, 1) and isbit(3, 0) bits 0 and 1, 3 drizzle and liquid cloud
    classification[bits == 6] = 4 # ice (falling and cold) bits 1 and 2: 2+4=6
    classification[bits == 7] = 5 # 0,1,2
    classification[(bits == 8)] = 6 #3 only, melting ice
    classification[(bits == 9)] = 7 # 3 and 0 melting ice and droplets
    classification[(bits == 16)] = 8 # bit 4 is 16 this means it's class 8, so confusing: aerosol
    classification[(bits == 32) & ~clutter] = 9 # bit 5 is 32: insects
    classification[(bits == 48) & ~clutter] = 10 # 4 and 5 = 48: insects and aerosol
    classification[clutter & (~(bits == 16))] = 0 # clutter and no bit 4 (=16, aerosol)

    return classification


def plot_bin_average(data_container1, data_container2,  **kwargs):
    """scatter plot for variable comparison between two devices or variables

    Args:
        data_container1 (dict): container 1st device
        data_container2 (dict): container 2nd device
        x_lim (list): limits of var used for x axis
        y_lim (list): limits of var used for y axis
        c_lim (list): limits of var used for color axis
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **fig_size (list): size of the figure in inches
        **font_size (int): default: 15
        **font_weight (int): default: semibold
        **Nbins (int) : number of bins for histograms
        **fig
        **ax

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """

    fig_size = np.repeat(min(kwargs['fig_size']), 2) if 'fig_size' in kwargs else [6, 6]
    fontsize = kwargs['font_size'] if 'font_size' in kwargs else 12
    fontweight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'

    var1_tmp = data_container1
    var2_tmp = data_container2

    combined_mask = np.logical_or(var1_tmp['mask'], var2_tmp['mask'])
    var1 = var1_tmp['var'][~combined_mask].ravel()  # +4.5
    var2 = var2_tmp['var'][~combined_mask].ravel()

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [np.nanmin(var1), np.nanmax(var1)]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [np.nanmin(var2), np.nanmax(var2)]
    try:
        Nbins = kwargs['Nbins'] if 'Nbins' in kwargs else int(round((np.nanmax(var1) - np.nanmin(var1)) /
                                                                    (2 * (np.nanquantile(var1, 0.75) -
                                                                          np.nanquantile(var1, 0.25)) * len(var1) ** (-1 / 3))))
    except OverflowError:
        print(f'var1 {var1_tmp["name"]}: len is {len(var1)}, '
              f'IQR is {np.nanquantile(var1, 0.75)} - {np.nanquantile(var1, 0.25)},'
              f'max is {np.nanmax(var1)}, min is {np.nanmin(var1)}')
        Nbins = 100
    # Freedman-Diaconis rule: h=2×IQR×n−1/3. number of bins is (max−min)/h, where n is the number of observations
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram

    # create histogram plot
    bin_means, edges, number = scipy.stats.binned_statistic(var1, var2, bins=Nbins)

    if 'fig' in kwargs:
        fig = kwargs['fig']
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(1, figsize=fig_size)

    ax.hlines(bin_means, edges[:-1], edges[1:])
    ax.set_xlabel('{} {} [{}]'.format(var1_tmp['system'], var1_tmp['name'], var1_tmp['var_unit']),
                  fontweight=fontweight, fontsize=fontsize)
    ax.set_ylabel('{} {} [{}]'.format(var2_tmp['system'], var2_tmp['name'], var2_tmp['var_unit']),
                  fontweight=fontweight, fontsize=fontsize)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return fig, ax


def rimed_mass_fraction_dmitri(vd_mean_pcor_sealevel, frequency=35):
    """
    Function to compute rime mass fraction from mean Doppler velocity

    :param vd_mean_pcor_sealevel: Mean Doppler velocity, corrected for air density (sea level)
    :param frequency: radar frequency (GHz); defaults to 35. For different frequencies, different p1 to p5 factors are
    used in the polynomial fit
    :return: Rimed mass fraction derived from Dmitri Moisseev's fit (Kneifel & Moisseev 2020)
    """
    if frequency == 35:
        p1 = 0.0791
        p2 = -0.5965
        p3 = 1.362
        p4 = -0.5525
        p5 = -0.0514

    elif frequency == 94:
        p1 = 0.0961
        p2 = -0.6073
        p3 = 1.032
        p4 = 0.2212
        p5 = -0.4358

    else:
        NameError("frequency not in considered cases (35 or 94 GHz)")

    vd_mean_pcor_pos = -1 * vd_mean_pcor_sealevel

    # Stefan: Basically I would anyways only use MDV>1.5 m/s for the analysis,
    # because a rimed dendrite cannot be distinguished from a non-rimed snowflake
    # when both fall at a speed of 1 m/s

    np.putmask(vd_mean_pcor_pos, vd_mean_pcor_pos < 1.5, np.nan)

    rmf_dmitri = p1 * vd_mean_pcor_pos**4 + p2 * vd_mean_pcor_pos**3 + p3 * vd_mean_pcor_pos**2 \
                 + p4 * vd_mean_pcor_pos + p5

    # filter out negative rimed mass fractions (vd > ca. - 0.65 m / s)
    neg_rf, neg_rf_dmitri_ind = np.where(rmf_dmitri < 0)
    rmf_dmitri[neg_rf_dmitri_ind] = np.nan

    # some very few rain events are misclassified by CLOUDNET as ice and also Dmitri's
    # polynomial fit actually bends up after 3 m/s, so we could just set everything falling faster
    # to a rimed mass fraction of 1.
    rf = np.where(vd_mean_pcor_pos > 2.5, 0.85, rmf_dmitri)

    return rf


def rolling_window_lastaxis(a, window):
    """
   Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>
    Args:
        a:
        window:

    Returns:

    """
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long (longer than dimension of last axis).")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot_scatter_modified(data_container1, data_container2, data_container3, **kwargs):
    """scatter plot for variable comparison between two devices or variables

    Args:
        data_container1 (dict): container 1st device
        data_container2 (dict): container 2nd device
        x_lim (list): limits of var used for x axis
        y_lim (list): limits of var used for y axis
        c_lim (list): limits of var used for color axis
        **identity_line (bool): plot 1:1 line if True
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **custom_offset_lines (float): plot 4 extra lines for given distance
        **info (bool): print slope, interception point and R^2 value
        **fig_size (list): size of the figure in inches
        **font_size (int): default: 15
        **font_weight (int): default: semibold
        **colorbar (bool): if True, add a colorbar to the scatterplot
        **color_by (dict): data container 3rd device
        **scale (string): 'lin' or 'log' --> if you get a ValueError from matplotlib.colors
                          try setting scale to lin, log does not work for negative values!
        **cmap (string) : colormap
        **formstring (string): format string to use as key word argument to fig.colorbar, overwrites the default values
                               for 'lin' or 'log' scale. E.g. use "%.2f" for displaying two digits
        **Nbins (int) : number of bins for histograms

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    fig_size = np.repeat(min(kwargs['fig_size']), 2) if 'fig_size' in kwargs else [6, 6]
    fontsize = kwargs['font_size'] if 'font_size' in kwargs else 12
    labelsize = kwargs['label_size'] if 'label_size' in kwargs else 12
    fontweight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'

    var1_tmp = data_container1
    var2_tmp = data_container2
    var3_tmp = data_container3

    combined_mask = np.logical_or(var1_tmp['mask'], np.logical_or(var2_tmp['mask'], var3_tmp['mask']))
    colormap = kwargs['cmap'] if 'cmap' in kwargs else 'viridis'

    var1 = var1_tmp['var'][~combined_mask].ravel()  # +4.5
    var2 = var2_tmp['var'][~combined_mask].ravel()
    var3 = var3_tmp['var'][~combined_mask].ravel()

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [np.nanmin(var1), np.nanmax(var1)]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [np.nanmin(var2), np.nanmax(var2)]
    fig_size[0] = fig_size[0] + 2 if 'colorbar' in kwargs and kwargs['colorbar'] else fig_size[0]
    try:
        Nbins = kwargs['Nbins'] if 'Nbins' in kwargs else int(round((np.nanmax(var1) - np.nanmin(var1)) /
                                                                    (2 * (np.nanquantile(var1, 0.75) -
                                                                          np.nanquantile(var1, 0.25)) * len(var1) ** (-1 / 3))))
    except OverflowError:
        print(f'var1 {var1_tmp["name"]}: len is {len(var1)}, '
              f'IQR is {np.nanquantile(var1, 0.75)} - {np.nanquantile(var1, 0.25)},'
              f'max is {np.nanmax(var1)}, min is {np.nanmin(var1)}')
        Nbins = 100
    # Freedman-Diaconis rule: h=2×IQR×n−1/3. number of bins is (max−min)/h, where n is the number of observations
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram

    # create histogram plot
    H, xedges, yedges = np.histogram2d(var1, var2, bins=Nbins, range=[x_lim, y_lim])
    H1 = H < 10

    print("Coloring scatter plot by {}...\n".format(var3_tmp['name']))
    # overwrite H
    H = np.zeros(H.shape)
    x_coords = np.digitize(var1, xedges)
    y_coords = np.digitize(var2, yedges)
    # find unique bin combinations = pixels in scatter plot
    # sort x and y coordinates using lexsort
    # lexsort sorts by multiple columns, first by y_coords then by x_coords

    newer_order = np.lexsort((x_coords, y_coords))
    x_coords = x_coords[newer_order]
    y_coords = y_coords[newer_order]
    var3 = var3[newer_order]
    first_hit_y = np.searchsorted(y_coords, np.arange(1, Nbins + 2))
    first_hit_y.sort()
    first_hit_x = [np.searchsorted(x_coords[first_hit_y[j]:first_hit_y[j + 1]], np.arange(1, Nbins + 2))
                   + first_hit_y[j] for j in np.arange(Nbins)]

    for x in range(Nbins):
        for y in range(Nbins):
            H[y, x] = np.nanmedian(var3[first_hit_x[x][y]: first_hit_x[x][y + 1]])
    np.putmask(H, H1, np.nan)
    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(1, figsize=fig_size)

    c_lim = kwargs['c_lim'] if 'c_lim' in kwargs else [1, round(np.nanmax(H), int(np.log10(max(np.nanmax(H), 10.))))]

    if 'scale' in kwargs and kwargs['scale'] == 'lin':
        formstring = "%.0f"
        pcol = ax.pcolormesh(X, Y, np.transpose(H), vmin=c_lim[0], vmax=c_lim[1], cmap=colormap)
    else:
        formstring = "%.2E"
        pcol = ax.pcolormesh(X, Y, np.transpose(H), norm=matplotlib.colors.LogNorm(vmin=c_lim[0], vmax=c_lim[1]),
                             cmap=colormap)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('{} {} [{}]'.format(var1_tmp['system'], var1_tmp['name'], var1_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.set_ylabel('{} {} [{}]'.format(var2_tmp['system'], var2_tmp['name'], var2_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    if 'colorbar' in kwargs and kwargs['colorbar']:
        cmap = copy.copy(plt.get_cmap(colormap))
        cmap.set_under('white', 1.0)
        formstring = kwargs['formstring'] if 'formstring' in kwargs else formstring
        cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8, format=formstring)
        cbar.set_label(label="median {} [{}]".format(var3_tmp['name'], var3_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
        cbar.mappable.set_clim(c_lim)
        cbar.aspect = 50

    if 'title' in kwargs:
            ax.set_title(kwargs['title'], fontweight=fontweight, fontsize=fontsize)

    plt.grid(b=True, which='both', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    # ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    if 'colorbar' in kwargs and kwargs['colorbar']:
        cbar.ax.tick_params(axis='both', which='major', labelsize=labelsize, width=2, length=4)

    return fig, ax


def plot_scatter_foo_modified(data_container1, data_container2, **kwargs):
    """scatter plot for variable comparison between two devices or variables
    TODO merge this funciton into plot_scatter_modified

    Args:
        data_container1 (dict): container 1st device
        data_container2 (dict): container 2nd device
        x_lim (list): limits of var used for x axis
        y_lim (list): limits of var used for y axis
        c_lim (list): limits of var used for color axis
        **identity_line (bool): plot 1:1 line if True
        **z_converter (string): convert var before plotting use eg 'lin2z'
        **var_converter (string): alternate name for the z_converter
        **custom_offset_lines (float): plot 4 extra lines for given distance
        **info (bool): print slope, interception point and R^2 value
        **fig_size (list): size of the figure in inches
        **font_size (int): default: 15
        **font_weight (int): default: semibold
        **colorbar (bool): if True, add a colorbar to the scatterplot
        **color_by (dict): data container 3rd device
        **scale (string): 'lin' or 'log' --> if you get a ValueError from matplotlib.colors
                          try setting scale to lin, log does not work for negative values!
        **cmap (string) : colormap
        **formstring (string): format string to use as key word argument to fig.colorbar, overwrites the default values
                               for 'lin' or 'log' scale. E.g. use "%.2f" for displaying two digits
        **Nbins (int) : number of bins for histograms

    Returns:
        tuple with

        - **fig**: matplotlib figure
        - **ax**: axis
    """
    fig_size = np.repeat(min(kwargs['fig_size']), 2) if 'fig_size' in kwargs else [6, 6]
    fontsize = kwargs['font_size'] if 'font_size' in kwargs else 12
    labelsize = kwargs['label_size'] if 'label_size' in kwargs else 12
    fontweight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'

    var1_tmp = data_container1
    var2_tmp = data_container2

    combined_mask = np.logical_or(var1_tmp['mask'], var2_tmp['mask'])
    colormap = kwargs['cmap'] if 'cmap' in kwargs else 'viridis'

    var1 = var1_tmp['var'][~combined_mask].ravel()  # +4.5
    var2 = var2_tmp['var'][~combined_mask].ravel()

    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [np.nanmin(var1), np.nanmax(var1)]
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [np.nanmin(var2), np.nanmax(var2)]
    fig_size[0] = fig_size[0] + 2 if 'colorbar' in kwargs and kwargs['colorbar'] else fig_size[0]
    try:
        Nbins = kwargs['Nbins'] if 'Nbins' in kwargs else int(round((np.nanmax(var1) - np.nanmin(var1)) /
                                                                    (2 * (np.nanquantile(var1, 0.75) -
                                                                          np.nanquantile(var1, 0.25)) * len(var1) ** (-1 / 3))))
    except OverflowError:
        print(f'var1 {var1_tmp["name"]}: len is {len(var1)}, '
              f'IQR is {np.nanquantile(var1, 0.75)} - {np.nanquantile(var1, 0.25)},'
              f'max is {np.nanmax(var1)}, min is {np.nanmin(var1)}')
        Nbins = 100
    # Freedman-Diaconis rule: h=2×IQR×n−1/3. number of bins is (max−min)/h, where n is the number of observations
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram

    # create histogram plot
    H, xedges, yedges = np.histogram2d(var1, var2, bins=Nbins, range=[x_lim, y_lim])
    H1 = H < 10

    print("Coloring scatter plot by frequency of occurrence")
    np.putmask(H, H1, np.nan)
    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(1, figsize=fig_size)

    c_lim = kwargs['c_lim'] if 'c_lim' in kwargs else [1, round(np.nanmax(H), int(np.log10(max(np.nanmax(H), 10.))))]

    if 'scale' in kwargs and kwargs['scale'] == 'lin':
        formstring = "%.0f"
        pcol = ax.pcolormesh(X, Y, np.transpose(H), vmin=c_lim[0], vmax=c_lim[1], cmap=colormap)
    else:
        formstring = "%.2E"
        pcol = ax.pcolormesh(X, Y, np.transpose(H), norm=matplotlib.colors.LogNorm(vmin=c_lim[0], vmax=c_lim[1]),
                             cmap=colormap)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('{} {} [{}]'.format(var1_tmp['system'], var1_tmp['name'], var1_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.set_ylabel('{} {} [{}]'.format(var2_tmp['system'], var2_tmp['name'], var2_tmp['var_unit']), fontweight=fontweight, fontsize=fontsize)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    if 'colorbar' in kwargs and kwargs['colorbar']:
        cmap = copy.copy(plt.get_cmap(colormap))
        cmap.set_under('white', 1.0)
        formstring = kwargs['formstring'] if 'formstring' in kwargs else formstring
        cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01, shrink=0.8,
                            format=formstring)
        cbar.set_label(label="Frequency of occurrence", fontweight=fontweight, fontsize=fontsize)
        cbar.mappable.set_clim(c_lim)
        cbar.aspect = 50

    if 'title' in kwargs:
            ax.set_title(kwargs['title'], fontweight=fontweight, fontsize=fontsize)

    plt.grid(b=True, which='both', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    # ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=labelsize, width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    if 'colorbar' in kwargs and kwargs['colorbar']:
        cbar.ax.tick_params(axis='both', which='major', labelsize=labelsize, width=2, length=4)

    return fig, ax
