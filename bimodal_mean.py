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
    fig = fit_fault_set(populations.main_normal_faults(),
            title='Main Normal Fault Population Conjugates')
    fig.savefig('MainNormalFaultConjugates.pdf')

    fig = fit_fault_set(populations.strike_slip_faults(), split_axis=0,
            title='Strike-slip Fault Population Conjugates')
    fig.savefig('StrikeslipFaultConjugates.pdf')

    fig = fit_fault_set(populations.sec_normal_faults(),
            title='Secondary Normal Fault Population Conjugates')
    fig.savefig('SecNormalFaultConjugates.pdf')

    plt.show()

def fit_fault_set(faults, split_axis=1, title='Poles to Planes'):
    """Given a sequence of swfaults representing a bi-modal distribution, fit
    a plane to each of the modes."""
    vol = populations.vol
    normals = [norm for fault in faults for norm in normal_vecs(fault, vol)]
    normals = np.array(normals)

    strikes_dips = fit_bimodal_bidirectional(normals, split_axis=split_axis)

    fig, ax = plt.subplots(subplot_kw=dict(projection='stereonet'))
    strike, dip = smath.vector2pole(*normals.T)
    ax.density_contourf(strike, dip)
#    ax.pole(strike, dip, 'ko', markersize=2)


    conj = []
    for sd, color in zip(strikes_dips, ['r', 'g']):
        ax.pole(*sd, color=color)
        ax.plane(*sd, color=color, lw=2)
        conj.append(u'%0.0f\u00b0/%0.0f\u00b0' % sd)
    ax.set_azimuth_ticks([])
    ax.grid(True)
    ax.set_title(title + '\n' + ' and '.join(conj), y=-0.05, va='top')
    return fig

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


