import numpy as np

def get_apodize_window(aman, apodize_samps=1600):
    w = np.ones(aman.samps.count)
    cosedge = np.cos(np.linspace(0, np.pi/2, apodize_samps))
    w[-apodize_samps:] = cosedge
    w[:apodize_samps] = np.flip(cosedge)
    return w
    
def apodize_cosine(aman, signal='signal', apodize_samps=1600, in_place=True,
                   apo_axis='apodized'):
    """
    Function to smoothly filter the timestream to 0's on the ends with a
    cosine function.

    Args:
        signal (str): Axis to apodize
        apodize_samps (int): Number of samples on tod ends to apodize.
        in_place (bool): writes over signal with apodized version
        apo_axis (str): 
    """
    w = get_apodize_window(aman, apodize_samps)
    if in_place:
        aman[signal] *= w
    else:
        aman.wrap_new(apo_axis, dtype='float32', shape=('dets', 'samps'))
        aman[apo_axis] = aman[signal]*w

