import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

import geometric_functions

def main():
    strikes, dips, normals, slip = generate_normal_ss_data(330, 60, n=500, porp=1)
    #strikes, dips, normals, slip = generate_normal_data(330, 60, n=500, porp=10)
    sigma = invert_plane_stress(normals, slip)
    plot(sigma, strikes, dips)
    plt.show()

def generate_normal_ss_data(avg_strike, avg_dip, n=10, noise_std=5, porp=2):
    """Generate several conjugate normal and s/s faults with noisy s/d"""
    def append(*args):
        return np.hstack(args)
    def filter(strike, dip):
        """Ensure strike & dip are in the correct ranges"""
        strike[dip > 90] -= 180
        dip[dip > 90] = 180 - dip[dip>90]

        strike[dip < 0] -= 180
        dip[dip < 0] *= -1

        strike[strike < 0] += 360
        strike[strike > 360] -= 360

        return strike, dip

    # Generate conjugate normal faults
    norm_strike = avg_strike * np.ones(n)
    norm_strike[n//porp:] = avg_strike + 180
    norm_dip = avg_dip * np.ones(n)

    # Generate strike slip faults @ +- 30 deg of normal strike
    ss_strike = (avg_strike - 30) * np.ones(n)
    ss_strike[n//porp:] = avg_strike + 30
    ss_dip = 90 * np.ones(n)

    # Add noise
    for item in [norm_strike, norm_dip, ss_strike, ss_dip]:
        item += noise_std * np.random.randn(item.size)

    norm_strike, norm_dip = filter(norm_strike, norm_dip)
    ss_strike, ss_dip = filter(ss_strike, ss_dip)

    # Calculate slip
    norm_normal = geometric_functions.plane2normal(norm_strike, norm_dip)
    norm_slip = geometric_functions.normal_slip(*norm_normal)

    ll_ss_normal = geometric_functions.plane2normal(ss_strike[n//2:], ss_dip[n//2:])
    ll_ss_slip = geometric_functions.left_slip(*ll_ss_normal)

    rl_ss_normal = geometric_functions.plane2normal(ss_strike[:n-n//2], ss_dip[:n-n//2])
    rl_ss_slip = geometric_functions.left_slip(*rl_ss_normal)

    strike, dip = append(norm_strike, ss_strike), append(norm_dip, ss_dip)
    normal = append(norm_normal, rl_ss_normal, ll_ss_normal)
    slip = append(norm_slip, rl_ss_slip, ll_ss_slip)
    return strike, dip, normal, slip

def generate_normal_data(avg_strike, avg_dip, n=10, noise_std=5, porp=2):
    """Generate several conjugate normal faults with noisy s/d"""
    opp_strike = avg_strike + 180
    if opp_strike > 360: 
        opp_strike -= 360
    strike = avg_strike * np.ones(n)
    strike[n//porp:] = opp_strike
    dip = avg_dip * np.ones(n)
    
    # Add noise
    strike += noise_std * np.random.randn(n)
    dip += noise_std * np.random.randn(n)

    # Filter out things out of a reasonable range
    strike[dip > 90] -= 180
    dip[dip > 90] = 180 - dip[dip>90]

    strike[dip < 0] -= 180
    dip[dip < 0] *= -1

    strike[strike < 0] += 360
    strike[strike > 360] -= 360
    

    normal = geometric_functions.plane2normal(strike, dip)
    slip = geometric_functions.normal_slip(*normal)

    return strike, dip, normal, slip


def test_slip():
    strike, dip = 90, 45
    x,y,z = geometric_functions.plane2normal(strike, dip)
    print 'Should be ~ [0, 1, -1]'
    print [item * np.sqrt(2) for item in [x,y,z]]
    print 'Should be 090/45 (s/d)', geometric_functions.normal2plane(x,y,z)
    print 'Normal slip should be ~ [0, -1, -1]'
    print [item * np.sqrt(2) for item in geometric_functions.normal_slip(x,y,z)]
    print 'Reverse slip should be ~ [0, 1, 1]'
    print [item * np.sqrt(2) for item in geometric_functions.reverse_slip(x,y,z)]

    print 'Now testing normal2plane'
    print 'Should be 090/45 (s/d)', geometric_functions.normal2plane(0, -1, -1)
    print 'Should be 45/180 (p/b)', geometric_functions.normal2plunge_bearing(0, -1, -1)
    


def invert(normal, slip, weights=None):
    nx, ny, nz = normal
    sx, sy, sz = slip

    Gx = np.zeros((nx.size, 5))
    Gx[:,0] = nx - nx**3 + nx*nz**2 
    Gx[:,1] = ny - 2*ny*nx**2
    Gx[:,2] = nz - 2*nz*nx**2
    Gx[:,3] = -nx*ny**2 + nx*nz**2
    Gx[:,4] = -2*nx*ny*nz
    
    Gy = np.zeros((nx.size, 5))
    Gy[:,0] = -ny*nx**2 + ny*nz**2
    Gy[:,1] = nx - 2*nx*ny**2
    Gy[:,2] = -2*nx*ny*nz
    Gy[:,3] = ny - ny**3 + ny*nz**2
    Gy[:,4] = nz - 2*nz*ny**2
    
    Gz = np.zeros((nx.size, 5))
    Gz[:,0] = -nz*nx**2 - nz + nz**3
    Gz[:,1] = -2*nx*ny*nz
    Gz[:,2] = nx - 2*nx*nz**2
    Gz[:,3] = -ny**2*nz - nz + nz**3
    Gz[:,4] = ny - 2*ny*nz**2

    G = np.vstack((Gx, Gy, Gz)).T
    d = np.hstack([sx,sy,sz])

    if weights is not None:
        weights = np.tile(weights, 3)
        G *= weights
        d *= weights

    m, residual, rank, sing_vals = np.linalg.lstsq(G.T,d.T)

    s11, s12, s13, s22, s23 = m
    s33 = -(s11 + s22)

    sigma = np.array([[s11, s12, s13],
                      [s12, s22, s23],
                      [s13, s23, s33]])
    return sigma
 
def invert_plane_stress(normal, slip, weights=None):
    nx, ny, nz = normal
    sx, sy, sz = slip

    Gx = np.zeros((nx.size, 3))
    Gx[:,0] = nx - nx**3 + nx*nz**2 
    Gx[:,1] = ny - 2*ny*nx**2
    Gx[:,2] = -nx*ny**2 + nx*nz**2
    
    Gy = np.zeros((nx.size, 3))
    Gy[:,0] = -ny*nx**2 + ny*nz**2
    Gy[:,1] = nx - 2*nx*ny**2
    Gy[:,2] = ny - ny**3 + ny*nz**2
    
    Gz = np.zeros((nx.size, 3))
    Gz[:,0] = -nz*nx**2 - nz + nz**3
    Gz[:,1] = -2*nx*ny*nz
    Gz[:,2] = -ny**2*nz - nz + nz**3


    G = np.vstack((Gx, Gy, Gz)).T
    d = np.hstack([sx,sy,sz])

    if weights is not None:
        weights = np.tile(weights, 3)
        G *= weights
        d *= weights

    m, residual, rank, sing_vals = np.linalg.lstsq(G.T,d.T)

    s11, s12, s22 = m
    s33 = -(s11 + s22)

    sigma = np.array([[s11, s12, 0.0],
                      [s12, s22, 0.0],
                      [0.0, 0.0, s33]])
    return sigma
 
def kinematic_inversion(normal, slip):
    normal, slip = np.vstack(normal).T, np.vstack(slip).T
    T, P, B = geometric_functions.kinematic_axes(normal, slip)
    axes = np.vstack((T[:,:2], P[:,:2], B[:,:2]))
    cov = np.cov(axes.T)
    cov_3d = np.zeros((3,3), dtype=np.float)
    cov_3d[:2, :2] = cov
    cov_3d[2,2] = 1
    return cov_3d #, T, P, B
      
def principal(sigma):
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.abs(vals)
    order = vals.argsort()[::-1]
    vecs = vecs[:, order]
    vals = vals[order]

    func = geometric_functions.vector2plunge_bearing
    vec_orientations = [func(*item) for item in vecs.T]

    """
    val1, val2, val3 = vals
    print (val2-val3) / (val1 - val3)
    print 'sigma1', val1 
    print 'sigma2', val2
    print 'sigma3', val3
    """
    return vals, vec_orientations


def plot(sigma, strikes, dips):
    values, vectors = principal(sigma)
    sigma1, sigma2, sigma3 = vectors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='stereonet')
    plt.hold(True)
    ax.density_contourf(strikes, dips)
    #ax.pole(strikes, dips, 'b.')
    ax.line(sigma1[0],sigma1[1], 'r^', label='sigma1', markersize=18)
    ax.line(sigma2[0],sigma2[1], 'g^', label='sigma2', markersize=18)
    ax.line(sigma3[0],sigma3[1], 'b^', label='sigma3', markersize=18)

if __name__ == '__main__':
    main()
