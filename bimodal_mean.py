"""
Plots density contour plots of poles to plane for each triangle of each fault
in each fault population.  

Modes of each fault population are determined through a modified K-means 
approach (see `fit_fault_set`, `fit_bimodal`, and `fit_bimodal_bidirectional`).
"""
import numpy as np
import scipy.cluster

import matplotlib.pyplot as plt
import mplstereonet.stereonet_math as smath
import mplstereonet

import fault_populations as populations
from swfault_vectors import normals as normal_vecs

def main():
    fig, axes = plt.subplots(ncols=3, figsize=(24, 6), 
                    subplot_kw=dict(projection='stereonet'))

    fit_fault_set(axes[0], populations.main_normal_faults(),
            title='A) Main Normal Fault Population')

    fit_fault_set(axes[1], populations.strike_slip_faults(), split_axis=0,
            title='B) Strike-slip Fault Population')

    fit_fault_set(axes[2], populations.sec_normal_faults(),
            title='C) Secondary Normal Fault Population')

    fig.savefig('FaultPopulationsPlot.pdf')

    plt.show()

def fit_fault_set(ax, faults, split_axis=1, title='Poles to Planes'):
    """Given a sequence of swfaults representing a bi-modal distribution, fit
    a plane to each of the modes."""
    vol = populations.vol
    faults = list(faults)
    numfaults = len(faults)

    normals = [norm for fault in faults for norm in normal_vecs(fault, vol)]
    numvecs = len(normals)
    normals = np.array(normals)

    strikes_dips = fit_bimodal_bidirectional(normals, split_axis=split_axis)

    strike, dip = smath.vector2pole(*normals.T)
    ax.pole(strike, dip, 'ko', markersize=2, alpha=0.5)
    cax = ax.density_contourf(strike, dip, sigma=3)
    ax.density_contour(strike, dip, linewidths=2, sigma=3)
    cbar = ax.figure.colorbar(cax, ax=ax)
    cbar.set_label('Number of standard deviations', rotation=-90)

    conj = []
    for sd, color in zip(strikes_dips, ['r', 'g']):
        ax.pole(*sd, color=color)
        ax.plane(*sd, color=color, lw=2)
        conj.append(u'%0.0f\u00b0/%0.0f\u00b0' % sd)
    ax.set_azimuth_ticks([])
#    ax.grid(True)
    ax.set_title(title + '\n' + ' and '.join(conj), y=1.0, va='bottom')
    ax.annotate('{} planes from {} faults'.format(numvecs, numfaults), 
            xy=(0.5, 0), xytext=(0, -12), xycoords='axes fraction', 
            textcoords='offset points', ha='center', va='top')

def fit_bimodal(normals, weights=None, k=2):
    """Given a set of vectors, find the resultant vector of two modes using
    kmeans to seperate the modes.  The vectors will be weighted by "weights",
    if specified."""
    obs = scipy.cluster.vq.whiten(normals)
    codebook, _ = scipy.cluster.vq.kmeans2(obs, k)
    mode, _ = scipy.cluster.vq.vq(obs, codebook)

    if weights is not None:
        weighted_normals = (normals.T * weights).T
    else:
        weighted_normals = normals
    modes = [weighted_normals[mode==i] for i in range(k)]

    return (smath.vector2pole(*mode.mean(axis=0)) for mode in modes)

def fit_bimodal_bidirectional(normals, weights=None, split_axis=1):
    """Similar to "fit_bimodal", but each vector is assumed to represent a
    bi-directional measurement. i.e. Each vector is assumed to be equivalent to
    its opposite."""
    # Append the opposite vector for each observation...
    normals = np.r_[normals, -normals]
    
    # Now rotate things into PCA space
    normals -= normals.mean(axis=0)
    cov = np.cov(normals.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.abs(vals).argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:,order]
    pca = normals.dot(vecs)

    # And split along the specified axis...
    normals = normals[pca[:,split_axis] > 0]
    if weights is not None:
        weights = np.r_[weights, weights]
        weights = weights[pca[:,split_axis] > 0]

    return fit_bimodal(normals, weights)

if __name__ == '__main__':
    main()


