import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d

_num_edges = 1140
_encoder_disk_radius = 346.25
I_fix = 0.8  # Moment of inertia

smoothing = 0
window_width = _num_edges


def update_pattern(solved):
    """
    Update the event timestamps and angles according to
    the definition of the middle point pattern.

    Parameters:
    solved: dict
    """
    solved["fast_time_1"] = (solved["fast_time_1"][1:] +
                             solved["fast_time_1"][:-1]) / 2
    solved["fast_time_2"] = (solved["fast_time_2"][1:] +
                             solved["fast_time_2"][:-1]) / 2

    solved["angle_1"] = solved["angle_1"][:-1]
    solved["angle_2"] = solved["angle_2"][:-1]

    for i in solved["ref_indexes_1"]:
        t1 = solved["fast_time_1"][i-2]
        t2 = solved["fast_time_1"][i+2]
        solved["fast_time_1"][i-2:i+3] = np.linspace(t1, t2, 5)
    for i in solved["ref_indexes_2"]:
        t1 = solved["fast_time_2"][i-2]
        t2 = solved["fast_time_2"][i+2]
        solved["fast_time_2"][i-2:i+3] = np.linspace(t1, t2, 5)


def moving_average(x, w):
    return uniform_filter1d(x, size=w, mode='reflect')


def eval_offset_angle(solved):
    """
    Rough estimation of the offset angle
    """

    spline = UnivariateSpline(
        solved["fast_time_1"], solved["angle_1"], s=smoothing)
    dchi_dt = spline.derivative()(
        solved["fast_time_1"][solved["offcenter_idx1"]])
    dchi_dt_ma = moving_average(dchi_dt, window_width)

    offset_angle = solved["offset_time"] * dchi_dt_ma
    phi_mean = np.mean(offset_angle[_num_edges:-_num_edges])

    return phi_mean


def compute_observables(solved):
    """
    Compute R_{p,m}

    """
    # <(d\chi/dt)^2>_moving_average
    spline = UnivariateSpline(
        solved["fast_time_1"], solved["angle_1"], s=smoothing)
    dchi_dt = spline.derivative()(solved["fast_time_1"])
    dchi_dt2 = dchi_dt ** 2
    dchi_dt2_ma = moving_average(dchi_dt2, window_width)

    # Rpm (m=1,2, although m=0,1 in the documentation)
    deltat_1 = np.diff(solved["fast_time_1"])
    delt_normalized0 = deltat_1 * np.sqrt(dchi_dt2_ma[:-1])
    Rp1 = np.split(delt_normalized0, solved["ref_indexes_1"][:-1])[2:-2]
    Rp1 = np.average(Rp1, axis=0) - 2*np.pi/_num_edges

    spline = UnivariateSpline(
        solved["fast_time_2"], solved["angle_2"], s=smoothing)
    dchi_dt = spline.derivative()(solved["fast_time_2"])
    dchi_dt2 = dchi_dt ** 2
    dchi_dt2_ma = moving_average(dchi_dt2, window_width)

    deltat_2 = np.diff(solved["fast_time_2"])
    delt_normalized1 = deltat_2 * np.sqrt(dchi_dt2_ma[:-1])
    Rp2 = np.split(delt_normalized1, solved["ref_indexes_2"][:-1])[2:-2]
    Rp2 = np.average(Rp2, axis=0) - 2*np.pi/_num_edges

    E_mean = np.mean(dchi_dt2_ma) * I_fix / 2

    return Rp1, Rp2, E_mean


def L_version1(K, phi_mean, E_mean):
    """
    Compute the coefficient matrix L.
    tempalte differential is not taken into account.

    Parameters:
    K: int
        The largest odd number less than or equal to this value
        will be taken as the maximum mode number.
    phi_mean: float
        Mean offset angle.
    E_mean: float
        Mean energy.
    """

    N = _num_edges
    if K % 2 == 0:
        K = K - 1  # K must be odd
    K_ = K // 2 + 1  # number of modes (1, 3, 5, ..., K)

    L = np.zeros((2*N + 1, N + 2*K_))

    for p in range(N):
        for m in [0, 1]:
            i = m * N + p
            # template term
            if p == N-1:
                L[i, p] = -1
                L[i, 0] = 1
            else:
                L[i, p] = -1
                L[i, p + 1] = 1
            # V term
            base = 2 * np.pi / N / E_mean / 2
            for ik in range(K_):
                k = 2 * ik + 1  # only odd k
                angle = 2*np.pi*(2*p+1)/(2*N)
                if m == 0:
                    L[i, N + ik] = base * np.cos(k*(angle - phi_mean))
                    L[i, N + K_ + ik] = base * np.sin(k*(angle - phi_mean))
                elif m == 1:
                    L[i, N + ik] = base * np.cos(k*(angle - np.pi + phi_mean))
                    L[i, N + K_ + ik] = base * \
                        np.sin(k*(angle - np.pi + phi_mean))

    # constraint (\delta\theta_0 = 0)
    L[2*N, 0] = 1

    return L


def L_version2(K, phi_mean, E_mean):
    """
    Compute the coefficient matrix L.

    Parameters:
    K: int
        The largest odd number less than or equal to this value
        will be taken as the maximum mode number.
    phi_mean: float
        Mean offset angle.
    E_mean: float
        Mean energy.
    """

    N = _num_edges
    if K % 2 == 0:
        K = K - 1  # K must be odd
    K_ = K // 2 + 1  # number of modes (1, 3, 5, ..., K)

    L = np.zeros((2*N + 8 + 2*K_, 2*N + 2*K_))

    for p in range(N):
        for m in [0, 1]:
            i = m * N + p
            # template term
            if p == N-1:
                L[i, p] = -1
                L[i, 0] = 1
            else:
                L[i, p] = -1
                L[i, p + 1] = 1

            # V term
            base = 2 * np.pi / N / E_mean / 2
            for ik in range(K_):
                k = 2 * ik + 1  # only odd k
                angle = 2*np.pi*(2*p+1)/(2*N)
                if m == 0:
                    L[i, 2*N + ik] = base * np.cos(k*(angle - phi_mean))
                    L[i, 2*N + K_ + ik] = base * np.sin(k*(angle - phi_mean))
                elif m == 1:
                    L[i, 2*N + ik] = base * \
                        np.cos(k*(angle - np.pi + phi_mean))
                    L[i, 2*N + K_ + ik] = base * \
                        np.sin(k*(angle - np.pi + phi_mean))

        m = 1
        if p == N-1:
            L[m * N + p, N+p] = -1
            L[m * N + p, N] = 1
        else:
            L[m * N + p, N+p] = -1
            L[m * N + p, N+p+1] = 1

    # constraint coefficients for dekta theta and Delta theta
    L[2*N, 0] = 1

    L[2*N+1, 1] = 2
    L[2*N+1, 2] = -1

    L[2*N+2, N-2] = 1
    L[2*N+2, 2] = 1

    L[2*N+3, N-1] = 2
    L[2*N+3, 2] = 1

    L[2*N+4, N+0] = 2
    L[2*N+4, N+2] = -1
    L[2*N+4, N+N-2] = -1

    L[2*N+5, N+1] = 4
    L[2*N+5, N+2] = -3
    L[2*N+5, N+N-2] = -1

    L[2*N+6, N+N-1] = 4
    L[2*N+6, N+2] = -1
    L[2*N+6, N+N-2] = -3

    for i in range(N):
        # DC component of template must be zero
        L[2*N+7, N+i] = 1
        # n<=K mode of template diff diff must be zero
        for ik in range(K_):
            k = 2 * ik + 1
            L[2*N+8+2*ik, N+i] = - \
                np.cos(k * 2 * np.pi * i / N) + \
                np.cos(k * 2 * np.pi * (i-1) / N)
            L[2*N+8+2*ik+1, N+i] = - \
                np.sin(k * 2 * np.pi * i / N) + \
                np.sin(k * 2 * np.pi * (i-1) / N)

    return L


def V_of_chi(A, B, chi):
    ns = np.arange(1, 2*len(A) + 1, 2)  # odd modes only
    return (A[:, None]*np.cos(ns[:, None]*chi) +
            B[:, None]*np.sin(ns[:, None]*chi)).sum(axis=0)


def construct_R(Rp1, Rp2, K, ver=1):
    if K % 2 == 0:
        K = K - 1
    K_ = K // 2 + 1  # number of modes (1, 3, 5, ..., K)
    if ver == 1:
        R = np.concatenate((Rp1, Rp2, [0]))
    elif ver == 2:
        R = np.concatenate((Rp1, Rp2, [0]*(8 + 2*K_)))
    return R


def get_V_estimation(x, K, ver=1):
    N = _num_edges
    if K % 2 == 0:
        K = K - 1
    K_ = K // 2 + 1  # number of modes (1, 3, 5, ..., K)
    if ver == 1:
        A_est = x[N:N+K_]
        B_est = x[N+K_:N+2*K_]
    elif ver == 2:
        A_est = x[2*N:2*N+K_]
        B_est = x[2*N+K_:2*N+2*K_]
    else:
        raise ValueError("ver must be 1 or 2")
    return A_est, B_est


def template_subtraction(solved, template1, template2, A, B, phi_mean):
    template_model1 = np.roll(template1, solved["ref_indexes_1"][0])
    template_model1 = np.tile(
        template_model1, int(np.ceil(len(solved["fast_time_1"])/_num_edges))
    )[:len(solved["fast_time_1"])]

    spline = UnivariateSpline(
        solved["fast_time_1"], solved["angle_1"], s=smoothing)
    chi_smoothed = spline(solved["fast_time_1"])
    dchi_dt = spline.derivative()(solved["fast_time_1"])
    omega_hat = moving_average(dchi_dt, window_width)

    E_slow = 0.5 * I_fix * omega_hat**2
    omega_refined = np.sqrt(
        2.0 / I_fix * (E_slow - V_of_chi(A, B, chi_smoothed-phi_mean)))

    template_model1 = template_model1 / omega_refined

    solved["fast_time_raw_1"] = solved["fast_time_1"].copy()
    solved["fast_time_1"] = solved["fast_time_1"] - template_model1

    template_model2 = np.roll(template2, solved["ref_indexes_2"][0])
    template_model2 = np.tile(
        template_model2, int(np.ceil(len(solved["fast_time_2"])/_num_edges))
    )[:len(solved["fast_time_2"])]

    spline = UnivariateSpline(
        solved["fast_time_2"], solved["angle_2"], s=smoothing)
    chi_smoothed = spline(solved["fast_time_2"])
    dchi_dt = spline.derivative()(solved["fast_time_2"])
    omega_hat = moving_average(dchi_dt, window_width)

    E_slow = 0.5 * I_fix * omega_hat**2
    omega_refined = np.sqrt(
        2.0 / I_fix * (E_slow - V_of_chi(A, B, chi_smoothed-phi_mean)))

    template_model2 = template_model2 / omega_refined

    solved["fast_time_raw_2"] = solved["fast_time_2"].copy()
    solved["fast_time_2"] = solved["fast_time_2"] - template_model2


def correct_offcentering(solved, A, B, phi_mean):

    offcenter_idx1 = solved['offcenter_idx1']
    offcenter_idx2 = solved['offcenter_idx2']
    offset_time = solved['offset_time']

    spline = UnivariateSpline(
        solved["fast_time_1"], solved["angle_1"], s=smoothing)
    chi_smoothed = spline(solved["fast_time_1"][offcenter_idx1])
    dchi_dt = spline.derivative()(solved["fast_time_1"][offcenter_idx1])
    omega_hat = moving_average(dchi_dt, window_width)

    E_slow = 0.5 * I_fix * omega_hat**2
    omega_refined = np.sqrt(
        2.0 / I_fix * (E_slow - V_of_chi(A, B, chi_smoothed-phi_mean)))

    offset_angle = offset_time * omega_refined
    offcentering = np.tan(offset_angle) * _encoder_disk_radius

    solved['offcentering'] = offcentering

    solved['fast_time_raw_1'] = solved['fast_time_raw_1'][offcenter_idx1]
    solved['fast_time_raw_2'] = solved['fast_time_raw_2'][offcenter_idx2]
    solved['fast_time_1'] = solved['fast_time_1'][offcenter_idx1] - offset_time
    solved['fast_time_2'] = solved['fast_time_2'][offcenter_idx2] + offset_time
    solved['angle_1'] = solved['angle_1'][offcenter_idx1]
    solved['angle_2'] = solved['angle_2'][offcenter_idx2]
