"""
Provides convenient access to all faults used for inversion.

These are only the recent faults within the piggyback basin.  Faults outside
of the piggyback basin may have been rotated since their formation and are not 
used. Major faults (e.g. the large, right-lateral strike slip faults at the 
western side of the basin) are not used, as their orientation may not reflect
the recent stress state.
"""
import os
from glob import iglob
import geoprobe

DATADIR = os.path.join(os.path.dirname(__file__), 'data')
BASENAME = os.path.join(DATADIR, 'swFaults', 'jdk_piggyback')

vol = geoprobe.volume(os.path.join(DATADIR, 'Volumes', 'example.hdf'))

def all_normal_faults():
    """All normal faults in the piggyback basin."""
    for filename in iglob(BASENAME + '*normal*'):
        yield geoprobe.swfault(filename)

def main_normal_faults():
    """Main normal fault population in the piggyback basin."""
    for filename in iglob(BASENAME + '*normal*'):
        if 'perp' not in filename:
            yield geoprobe.swfault(filename)

def sec_normal_faults():
    """Secondary normal fault population in the piggyback basin."""
    for filename in iglob(BASENAME + '*normal*'):
        if 'perp' in filename:
            yield geoprobe.swfault(filename)

def all_faults():
    """All (recent and minor) faults in the piggyback basin."""
    for filename in iglob(BASENAME + '*'):
        yield geoprobe.swfault(filename)

def strike_slip_faults():
    """All strike-slip faults in the piggyback basin."""
    for filename in iglob(BASENAME + '*strikeslip*'):
        yield geoprobe.swfault(filename)

def left_lateral_faults():
    """All left-lateral strike-slip faults in the piggyback basin."""
    for filename in iglob(BASENAME + '*strikeslip*ll.swf'):
        yield geoprobe.swfault(filename)

def right_lateral_faults():
    """All right-lateral strike-slip faults in the piggyback basin."""
    for filename in iglob(BASENAME + '*strikeslip*rl.swf'):
        yield geoprobe.swfault(filename)
