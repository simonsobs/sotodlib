import numpy as np

def func_sines(t, a0, a1, a2, a3, a4, a5, a6, t0, t1, t2, t3, t4, t5, t6):
    """
    Define fitting function.

    Args:
        t: time or timing_frac of stimulator signal.
        a0-6: Amplitude of sin function for first to seventh frequency signal
        t0-6: Timing offset of sin function for first to seventh frequency signal

    Return:
        y: Value of fitting function.
    """
    y = (a0*np.sin(1*(t-t0)*2*np.pi)
         + a1*np.sin(2*(t-t1)*2*np.pi)
         + a2*np.sin(3*(t-t2)*2*np.pi)
         + a3*np.sin(4*(t-t3)*2*np.pi)
         + a4*np.sin(5*(t-t4)*2*np.pi)
         + a5*np.sin(6*(t-t5)*2*np.pi)
         + a6*np.sin(7*(t-t6)*2*np.pi))

    return y


def func_response_amplitude(f, tau, a):
    """
    Detector response function of amplitude.

    Args:
        f: Chopping frequency
        tau: time constant of a detector in [s]
        a: Amplitude of sin function for '0' frequency signal

    Return:
        y: Stimulator signal amplitude at chopping frequency f.
    """
    y = a /np.sqrt(1+(2*np.pi*f*tau)**2)
    return y


def func_response_phase(f, tau, theta_geo):
    """
    Detector response function of phase.

    Args:
        f: Chopping frequency [Hz]
        tau: time constant of a detector in [s]
        theta_geo: Offset of phase delay [deg]

    Return:
        theta: Phase delay of stimulator signal [deg]
    """
    theta = np.arctan(-2*np.pi*f*tau)*(180/np.pi) + theta_geo
    return theta


def func_response_phase_with_dt(f, tau, theta_geo, dt):
    """
    Detector response function of phase.

    Args:
        f: Chopping frequency [Hz]
        tau: time constant of a detector in [s]
        theta_geo: Offset of phase delay due to hardware effect, geo = geometry [deg]
        theta_dt: Offset of phase delay due to readout issue [deg], theta_dt*(pi/180) =  -delta_t*2pi*f
        dt: Time difference due to wrong time stamps

    Return:
        theta: Phase delay of stimulator signal [deg]
    """
    theta_dt = -dt*2*np.pi*f *(180/np.pi)
    theta = np.arctan(-2*np.pi*f*tau)*(180/np.pi) + theta_geo + theta_dt
    return theta


def get_downsample_factor(aman, ctx):
    """
    Get downsample factor of SMuRF readout for the axis manager data.

    Args:
        aman: Axis manager of detector data
        ctx: context file

    Return:
        downsample_factor: String representing the Downsample factor of SMuRF readout.
    """
    downsample_factor_tag = get_downsample_factor_tags(ctx)

    obs_list = ctx.obsdb.query(
        f"obs.obs_id == '{aman.obs_info.obs_id}'",
        tags=downsample_factor_tag
        )

    obs = obs_list[0]
    for tag in downsample_factor_tag:
        if obs[tag] == 1:
            downsample_factor = tag.split('_')[-1]

    return downsample_factor


def get_downsample_factor_tags(ctx):
    """
    Get downsample factor of SMuRF readout.

    Args:
        ctx: context file

    Return:
        List of downsample factor tags in the database.
    """
    cursor = ctx.obsdb.conn.execute("SELECT DISTINCT tag FROM tags")
    all_tags = np.array([row[0] for row in cursor.fetchall()])
    mask = np.char.find(all_tags,'downsample') != -1

    return np.array(all_tags)[mask].tolist()
