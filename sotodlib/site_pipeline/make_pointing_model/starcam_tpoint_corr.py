import numpy as np

def IA(tpoint_dict, az, el):
    ia = tpoint_dict['IA']
    corr_az = az - ia
    return corr_az, el

def IE(tpoint_dict, az, el):
    ie = tpoint_dict['IE']
    corr_el = el + ie
    return az, corr_el

def HECE(tpoint_dict, az, el):
    hece = tpoint_dict['HECE']
    corr_el = el + hece * np.cos(el)
    return az, corr_el

def CA(tpoint_dict, az, el):
    ca = tpoint_dict['CA']
    corr_az = az - (ca / np.cos(el))
    return corr_az, el

def NPAE(tpoint_dict, az, el):
    npae = tpoint_dict['NPAE']
    corr_az = az - npae * np.tan(el)
    return corr_az, el

def AN(tpoint_dict, az, el):
    an = tpoint_dict['AN']
    corr_az = az - an * np.tan(el) * np.sin(az)
    corr_el = el - an * np.cos(az)
    return corr_az, corr_el

def AW(tpoint_dict, az, el):
    aw = tpoint_dict['AW']
    corr_az = az - aw * np.tan(el) * np.cos(az)
    corr_el = el + aw * np.sin(az)
    return corr_az, corr_el

def correct(tpoint_dict, az, el):
    corr_params = {'IA': IA,
                   'IE': IE,
                   'HECE': HECE,
                   'CA': CA,
                   'NPAE': NPAE,
                   'AN': AN,
                   'AW': AW,
                   }
    corr_az = az
    corr_el = el
    for key in tpoint_dict.keys():
        corr_az, corr_el = corr_params[key](tpoint_dict, corr_az, corr_el)
    return corr_az, corr_el
