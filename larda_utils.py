import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import scipy

# convert to celsius
def toC(datalist):
    return datalist[0]['var']-273.15, datalist[0]['mask']


def crop_to_timestamp(dict, ts):
    t_ind = np.where(np.isin(dict['ts'], ts))[0]
    out = copy.deepcopy(dict)
    out['var'] = out['var'][t_ind, :]
    out['mask'] = out['mask'][t_ind, :]
    out['ts'] = out['ts'][t_ind]
    return out


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
    spectra_range = spectra['rg']
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
    labelsize = kwargs['label_size'] if 'label_size' in kwargs else 12
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
