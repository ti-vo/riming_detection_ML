import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import datetime
from sklearn.metrics import r2_score


def abline(slope, intercept, ax, **kwargs):
    """Add a line to an existing plot, defined by slope and intercept"""
    if not 'color' in kwargs:
        kwargs['color'] = 'k'
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    line,  = ax.plot(x_vals, y_vals, '--', **kwargs)
    return line


def density_plot(x, y, ax, vmin=0, vmax=12, r2=False, return_plot=False):
    """
    Plot a scatter density plot
    :param x: x-values
    :param y: y-values
    :return: matplotlib.pyplot.subplots() fig, ax objects
    """

    # Calculate the point density
    xy = np.vstack([x, y])
    not_nan_index = ~np.isnan(np.mean(xy, axis=0))
    xy = xy[:, not_nan_index]
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[not_nan_index][idx], y[not_nan_index][idx], z[idx]
    p = ax.scatter(x, y, c=z, s=100, edgecolor=None, vmin=vmin, vmax=vmax)
    if r2:
        ax.text(0, 1.2, 'R2: ' + str(r2_score(x, y)))
    if return_plot:
        return p



def plot_learning_curve(train_sizes_abs, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std,
                        **kwargs):
    """
    plot the learning curve
    :param train_sizes_abs: absolute number of training samples
    :param train_scores_mean: mean score for training
    :param train_scores_std: standard deviation for training (added as shaded area around mean)
    :param val_scores_mean: mean score for validation
    :param val_scores_std: standard deviation for validation (added as a shaded area)
    :param kwargs:
    :return: fig, ax
    """
    xlabel = kwargs['x_label'] if 'x_label' in kwargs else 'number of samples for training'
    ylabel = kwargs['y_label'] if 'x_label' in kwargs else 'RMSE'

    fig, ax = plt.subplots(1)
    ax.fill_between(train_sizes_abs, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    alpha=0.1, color='r')
    ax.fill_between(train_sizes_abs, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1,
                    color='blue')

    ax.plot(train_sizes_abs, train_scores_mean, color='r', label='training error')
    ax.plot(train_sizes_abs, val_scores_mean, label='validation error')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax


def meshplot_hyperparameters_ann(rmse_array):
    """
    create a meshplot with pixels colored by rmse
    :param rmse_array: 3D array containing the rmses of k-fold training/ validation runs
    :return: matplotlib.pyplot.subplots fig, ax objects
    """
    fig, ax = plt.subplots(1)
    m = ax.pcolormesh(np.nanmean(rmse_array, axis=0))
    c = fig.colorbar(m)
    ax.set_xlabel('number of neurons')
    ax.set_ylabel('number of hidden layers')
    c.set_label(label="RMSE")
    return fig, ax


def lineplots_hyperparameters_ann(rmse_array, **kwargs):
    """
    create plots of rmse vs. number of neurons for each layer
    :param rmse_array: 3D array containing the rmses of k-fold training/ validation runs
    :return: matplotlib.pyplot.subplots fig, ax objects
    """
    n_layers = rmse_array.shape[1]
    n_neurons = rmse_array.shape[2]
    if 'rmse_val' in kwargs:
        label = 'training'
        assert kwargs['rmse_val'].shape == rmse_array.shape, "RMSE array dimensions do not match"
    else:
        label='RMSE'

    fig, ax = plt.subplots(1, n_layers, figsize=(5*n_layers, 5), sharey=True)
    for i in range(n_layers):
        ax[i].fill_between(np.arange(n_neurons)+1,
                           np.nanmean(rmse_array[:, i, :], axis=0)-np.nanstd(rmse_array[:, i, :], axis=0),
                           np.nanmean(rmse_array[:, i, :], axis=0)+np.nanstd(rmse_array[:, i, :], axis=0), alpha=0.5,
                           facecolor='blue')
        if 'rmse_val' in kwargs:
            rmse_val = kwargs['rmse_val']
            ax[i].fill_between(np.arange(n_neurons)+1,
                           np.nanmean(rmse_val[:, i, :], axis=0)-np.nanstd(rmse_val[:, i, :], axis=0),
                           np.nanmean(rmse_val[:, i, :], axis=0)+np.nanstd(rmse_val[:, i, :], axis=0), alpha=0.5,
                           facecolor='red')
            ax[i].plot(np.arange(n_neurons)+1, np.nanmean(rmse_val[:, i, :], axis=0), color='red', label='validation')

        ax[i].plot(np.arange(n_neurons)+1, np.nanmean(rmse_array[:, i, :], axis=0), color='blue', label=label)
        ax[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
        ax[i].set_xlabel('number of neurons')
        ax[i].set_ylabel('RMSE')
        ax[i].set_title(f'{i+1} hidden layer(s)')
    ax[n_layers-1].legend()
    fig.tight_layout()
    return fig, ax


def plot_3d_scatter(x, y, z, color_by, xlabel, ylabel, zlabel, cbar=True, clabel='FR [unitless]'):
    """
    Plot a 3D scatter plot colored by a fourth variable
    :param x: x values
    :param y: y values
    :param z: z values
    :param color_by: fourth variable by which color code is defined
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param zlabel: z-axis label
    :return: matplotlib.pyplot.subplots fig, ax objects
    """
    fig, ax = initialize_3D_fig(xlabel, ylabel, zlabel)
    fig, sc = init_scatter(fig, ax, x, y, z, color_by)
    if cbar:
        colorbar = fig.colorbar(sc, shrink=0.5)
        colorbar.set_label(clabel)

    ax.view_init(elev=25, azim=25)
    fig.tight_layout()
    return fig, ax


def initialize_3D_fig(xlabel, ylabel, zlabel):
    """
    utility to initialize a 3D figure and add labels to the axes
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param zlabel: z-axis label
    :return: matplotlib.pyplot.subplots fig, ax objects
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return fig, ax


def init_scatter(fig, ax, x, y, z, color_by):
    """
    initialize scatter plot in an existing figure
    :param fig, ax: matplotlib.pyplot.subplots objects
    :param x: x values
    :param y: y values
    :param z: z values
    :param color_by: fourth variable by which color code is defined
    :return: fig object
    """
    sc = ax.scatter(x, y, z, c=color_by, cmap='viridis', edgecolors='face')
    return fig, sc


def animation_plot(x, y, z, color_by, xlabel, ylabel, zlabel, filename):
    """
    create and save a 3D animation of 3D scatter plot
    :param x: x values
    :param y: y values
    :param z: z values
    :param color_by: fourth variable by which color code is defined
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param zlabel: z-axis label
    :param filename: name of the output (video) file, path + animation.mp4 or so
    """

    def animate_vary_azimuth(i):
        ax.view_init(elev=0, azim=i)
        return fig,

    def animate_vary_elev(i):
        ax.view_init(elev=i, azim=0)
        return fig,

    def init():
        ax.scatter(x, y, z, c=color_by)
        return fig,

    fig, ax = initialize_3D_fig(xlabel, ylabel, zlabel)
    # Animate
    anim = animation.FuncAnimation(fig, animate_vary_azimuth, frames=np.arange(0, 362, 2), init_func=init, blit=True)
    anim2 = animation.FuncAnimation(fig, animate_vary_elev, frames=np.arange(0, 362, 2), init_func=init, blit=True)
    anim.save(filename+'_azimuth.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
    anim2.save(filename+'_elevation.mp4', fps=10, extra_args=['-vcodec', 'libx264'])


def plot_histogram(var, label, nbins=50):
    var = var[~np.isnan(var)]
    hist, bins = np.histogram(var, bins=nbins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(1)
    ax.bar(center, hist, align='center', width=width)
    ax.set_xlabel(label)
    ax.set_ylabel('frequency')
    return fig, ax


def shifted_colormap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    From Paul H at
    https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib/20528097

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot_timeseries(timestamp, var):

    dt_list = [datetime.datetime.utcfromtimestamp(time) for time in timestamp]
    jumps = np.where(np.diff(timestamp) > 3600)[0]
    for ind in jumps[::-1].tolist():
        # and modify the dt_list
        dt_list.insert(ind + 1, dt_list[ind] + datetime.timedelta(seconds=5))
        # add the fill array
        var = np.insert(var, ind + 1, -999, axis=0)
    var = np.ma.masked_equal(var, -999)

    fig, ax = plt.subplots(1, figsize=(22, 5.7))
    ax.plot(dt_list, var)
    ax.set_xlabel("Time [UTC]", fontweight='semibold', fontsize=12)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=range(1, 33, 3)))

    return fig, ax