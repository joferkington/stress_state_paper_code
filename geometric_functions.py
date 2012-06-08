import numpy as np

#-- Conversions from strike/dip or plunge/bearing to a vector -----------------
def normalize(*args):
    if len(args) == 3:
        x,y,z = args
        r = np.sqrt(x**2 + y**2 + z**2)
        r[r == 0] = 1e-15
        return x/r, y/r, z/r
    elif len(args) == 1:
        r = np.sqrt(np.sum(args[0]**2, axis=-1))
        r[r == 0] = 1e-15
        arr = args[0].T / r
        return arr.T
    else:
        raise ValueError('Invalid number of input arguments')

def vector2plunge_bearing(x,y,z):
    x, y, z = map(np.asarray, [x,y,z])
    if x.size == 1:
        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

    x,y,z = normalize(x,y,z)
    plunge = np.degrees(np.arcsin(z))
    bearing = np.degrees(np.arctan2(y, x))

    # Rotate bearing so that 0 is north instead of east
    bearing = 90-bearing
    bearing[bearing < 0] += 360

    # If the plunge angle is upwards, get the opposite end of the line
    filter = plunge < 0
    plunge[filter] *= -1
    bearing[filter] -= 180

    bearing[bearing < 0] += 360

    return plunge, bearing

def normal2plane(x,y,z):
    """Converts a normal vector to a plane (given as x,y,z)
    to a strike and dip of the plane using the Right-Hand-Rule.
    Input:
        x: The x-component of the normal vector
        y: The y-component of the normal vector
        z: The z-component of the normal vector
    Output:
        strike: The strike of the plane, in degrees clockwise from north
        dip: The dip of the plane, in degrees downward from horizontal
    """
    # Convert the normal of the plane to a plunge/bearing
    plunge, bearing = vector2plunge_bearing(x, y, z)

    # Now convert the plunge/bearing of the pole to the plane that it represents
    strike = bearing+90
    dip = 90-plunge
    strike[strike > 360] -= 360

    return strike, dip

def plane2normal(strike, dip):
    strike, dip = map(np.asarray, [strike, dip])
    if strike.size == 1:
        strike, dip = strike.reshape(-1), dip.reshape(-1)

    # Following RHR, dip is to the right, so the normal vector be to the right 
    normal_bearing = strike + 90 

    # Convert normal_bearing so that east is at 0 instead of N
    normal_bearing = 90 - normal_bearing
    normal_bearing[normal_bearing < 0] += 360

    # Dip remains mostly unchanged, but we're measuring downward 
    # from vertical now...
    normal_inclination = -dip

    # Convert to radians
    normal_bearing = np.radians(normal_bearing)
    normal_inclination = np.radians(normal_inclination)

    # Convert to cartesian
    x = np.sin(normal_inclination) * np.cos(normal_bearing)
    y = np.sin(normal_inclination) * np.sin(normal_bearing)
    z = -np.cos(normal_inclination) # +Z is up rather than down

    return x, y, z

#-- Slip vector functions -----------------------------------------------------
# All of these are undefined with a horizontal plane!! 
#    Will return <0,0,0>  in that case
# (should work fine for any other condition, though...)

def dip_vector(x,y,z):
    """Takes normal vector to a plane <x,y,z> and returns dip vector"""
    # Strike cross normal --> dip in RHR
    vec1 = np.vstack(strike_vector(x,y,z)).T
    vec2 = np.vstack((x,y,z)).T
    dip = np.cross(vec1, vec2)
    return normalize(dip[:,0], dip[:,1], dip[:,2])

def strike_vector(x,y,z):
    """Takes normal vector to a plane <x,y,z> and returns the strike vector"""
    # Vertical cross normal --> strike in RHR
    vec1 = np.vstack((np.zeros(x.size), np.zeros(x.size), np.ones(x.size))).T
    vec2 = np.vstack((x,y,z)).T
    strike = np.cross(vec1, vec2)
    return normalize(strike[:,0], strike[:,1], strike[:,2])

def normal_slip(x,y,z):
    return dip_vector(x,y,z)

def reverse_slip(x,y,z):
    return [-item for item in dip_vector(x,y,z)]

def left_slip(x,y,z):
    return [-item for item in strike_vector(x,y,z)]

def right_slip(x,y,z):
    return strike_vector(x,y,z)

def kinematic_axes(normal, slip):
    def find_bisector(arr):
        normal, slip = arr[:3], arr[3:]
        # Baxis will be normal x slip
        B = np.cross(normal, slip)

        # Now, solve for P-axis
        G = np.vstack((normal, slip, B))
        d = np.array([0.5, 0.5, 0])
        P = np.linalg.solve(G,d)

        # T-axis will be PxB
        T = np.cross(P, B)

        return np.hstack((T,P,B))

    # First, normalize both...
    normal, slip = normalize(normal), normalize(slip)

    arr = np.hstack((normal, slip))
    arr = np.apply_along_axis(find_bisector, 1, arr)

    T, P, B = arr[:,:3], arr[:,3:6], arr[:,6:]
    return T,P,B
    
