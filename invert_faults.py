"""
Inverts observed fault plane data for principal stress orientations using
Michael (1984)'s method. Uses each triangle of interpreted fault surfaces as
an independent fault and assumes pure dip or strike slip.
"""
import numpy as np
import matplotlib.pyplot as plt

import mplstereonet
from mplstereonet import vector2pole

import swfault_vectors
import invert
import geometric_functions

import fault_populations as populations

def main():
    """Invert observed fault plane data (from geoprobe faults) assuming pure
    dip-slip."""
    def title(text):
        plt.title(text, y=-0.025, va='top', size=24)

    fig = invert_normal_population(populations.main_normal_faults())
    title('Main normal fault population')
    fig.savefig('MainNormalFaultPop.pdf')

    fig = invert_normal_population(populations.sec_normal_faults())
    title('Secondary normal fault population')
    fig.savefig('SecNormalFaultPop.pdf')

    fig = invert_strikeslip_population()
    title('Strike-slip fault population')
    fig.savefig('StrikeslipFaultPop.pdf')

    fig = invert_all()
    title('All faults except secondary normal faults')
    fig.savefig('AllFaults.pdf')

    plt.show()

def invert_normal_population(faults):
    """Invert only normal faults."""
    slip_func = geometric_functions.normal_slip
    return invert_vectors(*build_vectors(faults, slip_func))

def invert_strikeslip_population():
    """Invert only strike/slip faults."""
    fault_pops = [populations.left_lateral_faults(), 
                  populations.right_lateral_faults()]
    slip_funcs = [geometric_functions.left_slip, 
                  geometric_functions.right_slip]
    return invert_multiple_populations(fault_pops, slip_funcs)

def invert_all():
    """Invert both strike/slip and normal faults."""
    fault_pops = [populations.left_lateral_faults(), 
                  populations.right_lateral_faults(),
                  populations.main_normal_faults()]
    slip_funcs = [geometric_functions.left_slip, 
                  geometric_functions.right_slip,
                  geometric_functions.normal_slip]
    return invert_multiple_populations(fault_pops, slip_funcs)

def invert_multiple_populations(fault_pops, slip_funcs):
    """Given multiple fault populations and slip functions, build a composite
    set of normals, slips, areas, and number of faults."""
    areas, normals, slips = [], [], []
    numfaults, numplanes = 0,0
    for pop, slip_func in zip(fault_pops, slip_funcs):
        norm, slip, area, nfault = build_vectors(pop, slip_func)
        areas.extend(area)
        normals.extend(norm)
        slips.extend(slip)
        numfaults += nfault
    return invert_vectors(normals, slips, areas, numfaults)

def build_vectors(faults, slip_func):
    """Given a set of faults and a slip sense function, build sets of normal
    vectors, slip vectors, areas of each triangle, and the number of faults."""
    areas, normals, slips = [], [], []
    numfaults = 0
    for fault in faults:
        fault_normals = list(swfault_vectors.normals(fault, populations.vol))
        normals.extend(fault_normals)
        areas.extend(swfault_vectors.areas(fault, populations.vol))
        slips.extend(slip_func(*xyz) for xyz in fault_normals)
        numfaults += 1
    return normals, slips, areas, numfaults

def invert_vectors(normals, slips, areas, nfaults):
    """Invert for the principal directions of the stress tensor assuming plane
    stress.  The solution is weighted by the area of each fault triangle 
    squared."""
    normals, slips, areas = np.atleast_1d(normals, slips, areas)
    slips = np.squeeze(slips)
    sigma = invert.invert_plane_stress(normals.T, slips.T, areas**2)
    fig, ax = plot(sigma, normals, nfaults)
    return fig

def plot(sigma, normals, nfaults):
    """Produces a contoured density plot of poles to planes for each triangle
    of each fault with the principal stress axes from the inversion labeled."""
    nplanes = normals.shape[0]
    strikes, dips = vector2pole(*normals.T)
    values, vectors = invert.principal(sigma)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    ax.set_azimuth_ticks([])
    ax.density_contourf(strikes, dips, sigma=2)

#    ax.pole(strikes, dips, 'b.', label='Poles', alpha=0.5)

    colors = {1:'r', 2:'g', 3:'b'}
    for i, (plunge, bearing) in enumerate(vectors, start=1):
        annotate_vector(ax, plunge, bearing, i, label=r'$\sigma_%i$'%i,
                        c=colors[i])

    ax.set_xlabel('\n\nUsing {} triangles from {} faults'.format(nplanes, nfaults),
            va='top')
    ax.legend(scatterpoints=1, numpoints=1, bbox_to_anchor=(1.3,1))
    return fig, ax

def annotate_vector(ax, plunge, bearing, i, **kwargs):
    """Annotate a sigma1, sigma2, or sigma3 plunge/bearing."""
    rotation = 90 - bearing
    lon, lat = mplstereonet.line(plunge, bearing)

    ax.scatter(lon, lat, s=200, marker=(3,0,rotation+33), clip_on=False,
               zorder=10, **kwargs)

    x = 30 * np.cos(np.radians(rotation))
    y = 30 * np.sin(np.radians(rotation))
    x, y = int(x), int(y)

    if plunge < 80:
        ax.annotate('', xy=(lon, lat), xytext=(x, y), xycoords='data', 
                textcoords='offset points', arrowprops=dict(arrowstyle='-'))

        ax.annotate(r'$\sigma_%i : %0.0f^{\circ}$'%(i, bearing), 
                xy=(lon, lat), xytext=(x, y), textcoords='offset points', 
                ha='right', va='center', size=24)
    else:
        ax.annotate(r'$\sigma_%i$'%i, xy=(lon, lat), 
                textcoords='offset points', xytext=(15, 0),
                ha='left', va='center', size=24, color='white')

if __name__ == '__main__':
    main()
