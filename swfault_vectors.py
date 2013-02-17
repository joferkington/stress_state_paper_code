"""Reads triangles of a geoprobe swfault within a non-convex hull of the 
fault's outline."""
import numpy as np
from shapely.geometry import Polygon
from matplotlib.delaunay import Triangulation
import geoprobe

def triangles(fault, vol):
    if isinstance(fault, basestring):
        fault = geoprobe.swfault(fault)
    if isinstance(vol, basestring):
        vol = geoprobe.volume(vol)
    geo_xyz = georef_xyz(fault, vol)

    mask = inside_outline(fault)
    return geo_xyz[fault.tri.triangle_nodes[mask]]

def inside_outline(fault):
    def inside(tri):
        return rotated_outline.contains(Polygon(rotated_xyz[tri]))
    # Iterate through triangles in internal coords and select those inside 
    # outline the non-convex outline of the fault...
    x, y, z = fault._internal_xyz.T
    rotated_tri = Triangulation(x, y)
    rotated_xyz = fault._internal_xyz
    rotated_outline = Polygon(fault._rotated_outline)

    return np.array([inside(tri) for tri in rotated_tri.triangle_nodes])

def georef_xyz(fault, vol):
    x,y = vol.model2world(*fault.xyz[:,:2].T)
    return np.vstack([x, y, -fault.z]).T

def areas(fault, vol):
    for tri in triangles(fault, vol):
        a, b, c = tri
        normal = np.cross(b-a, b-c)
        area = np.linalg.norm(normal) / 2.0
        yield abs(area)

def normals(fault, vol):
    """
    Triangulates an swfault and returns a iterable over the normal vectors
    of each triangle in the fault surface.

    Parameters
    ----------
        fault : A geoprobe.swfault object (or filename)
        vol : A geoprobe.volume object (or filename) (for georeferencing)

    Returns:
    --------
        An iterable of (x,y,z) for each triangle in the fault. x,y,z are the
        x, y, and z lengths of the unit normal to the plane.
    """
    for tri in triangles(fault, vol):
        yield triangle2normal(tri)

def triangle2normal(tri):
    """
    Returns the normal vector of a triangle.

    Parameters
    ----------
        tri : A 3x3 numpy array where each row corresponds to a point at the 
            corner of the triangle in x,y,z space.
    Returns
    -------
        normal : A vector of x,y,z representing the normal vector to the plane
    """
    a, b, c = tri
    normal = np.cross(b-a, b-c)
    normal /= np.linalg.norm(normal)
    return normal
