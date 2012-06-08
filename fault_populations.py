from glob import iglob
import geoprobe

BASEDIR = '/data/nankai/data/swFaults/'
BASENAME = 'jdk_piggyback'

vol = geoprobe.volume('/data/nankai/data/Volumes/kumdep01_flipY.3dv.vol')

def all_normal_faults():
    for filename in iglob(BASEDIR + BASENAME + '*normal*'):
        yield geoprobe.swfault(filename)

def main_normal_faults():
    for filename in iglob(BASEDIR + BASENAME + '*normal*'):
        if 'perp' not in filename:
            yield geoprobe.swfault(filename)

def sec_normal_faults():
    for filename in iglob(BASEDIR + BASENAME + '*normal*'):
        if 'perp' in filename:
            yield geoprobe.swfault(filename)

def all_faults():
    for filename in iglob(BASEDIR + BASENAME + '*'):
        yield geoprobe.swfault(filename)

def strike_slip_faults():
    for filename in iglob(BASEDIR + BASENAME + '*strikeslip*'):
        yield geoprobe.swfault(filename)

def left_lateral_faults():
    for filename in iglob(BASEDIR + BASENAME + '*strikeslip*ll.swf'):
        yield geoprobe.swfault(filename)

def right_lateral_faults():
    for filename in iglob(BASEDIR + BASENAME + '*strikeslip*rl.swf'):
        yield geoprobe.swfault(filename)


