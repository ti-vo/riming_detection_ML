import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def abline(slope, intercept, ax):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--', color='k')


def density_plot(x, y, ax):

    # Calculate the point density
    xy = np.vstack([x, y])
    not_nan_index = ~np.isnan(np.mean(xy, axis=0))
    xy = xy[:, not_nan_index]
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[not_nan_index][idx], y[not_nan_index][idx], z[idx]
    ax.scatter(x, y, c=z, s=100, edgecolor='')


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
    fig, ax = plt.subplots(1)
    m = ax.pcolormesh(np.mean(rmse_array, axis=0))
    c = fig.colorbar(m)
    ax.set_xlabel('number of neurons')
    ax.set_ylabel('number of hidden layers')
    c.set_label(label="RMSE")
    return fig, ax



def plot_3d_scatter(x, y, z, color_by, xlabel, ylabel, zlabel):
    fig, ax = initialize_3D_fig(xlabel, ylabel, zlabel)
    init_scatter(fig, ax, x, y, z, color_by)
    return fig, ax

def initialize_3D_fig(xlabel, ylabel, zlabel):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return fig, ax


def init_scatter(fig, ax, x, y, z, color_by):
    ax.scatter(x, y, z, c=color_by)
    return fig


def animation_plot(x, y, z, color_by, xlabel, ylabel, zlabel, filename):

    def animate_vary_azimuth(i):
        ax.view_init(elev=0, azim=i)
        return fig,

    def init():
        ax.scatter(x, y, z, c=color_by)
        return fig,

    fig, ax = initialize_3D_fig(xlabel, ylabel, zlabel)
    # Animate
    anim = animation.FuncAnimation(fig, animate_vary_azimuth, frames=360, init_func=init)
    # Save
    anim.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
