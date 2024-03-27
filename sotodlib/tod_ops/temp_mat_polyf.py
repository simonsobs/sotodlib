from scipy.special import eval_legendre
from numpy.polynomial.legendre import Legendre
from sotodlib.tod_ops import sub_polyf as spf
from sotodlib.tod_ops import flags

### Get subscan info 
_ = flags.get_turnaround_flags(aman, merge=False)
subscan_indices_l = spf._get_subscan_range_index(aman.flags["left_scan"].mask())
subscan_indices_r = spf._get_subscan_range_index(aman.flags["right_scan"].mask())
subscan_indices = np.vstack([subscan_indices_l, subscan_indices_r])
subscan_indices= subscan_indices[np.argsort(subscan_indices[:, 0])]

### define the degree of freedom in filter
degree = 12

### Normalization constant of legendre function 
norm_vector = np.arange(degree)
norm_vector = 2./(2.*norm_vector+1)

### Get TOD
tods = aman["dsT"]
time = aman["timestamps"]

### Process each subscan 
for subscan in subscan_indices:
    
    # Get each subscan to be filtered
    tod_mat = tods[:,subscan[0]:subscan[1]+1]
    
    # Scale time range to [-1,1]
    x = np.linspace(-1, 1, tod_mat.shape[1])
    dx = np.mean(np.diff(x))
    sub_time = time[subscan[0]:subscan[1]+1]
    
    # Generate legendre functions of each degree and store them in an array
    arr_legendre = []
    for deg in range(degree) : 
        each_legendre = eval_legendre(deg, x) 
        arr_legendre.append(each_legendre)
    arr_legendre = np.array(arr_legendre)
        
    # Take inner product to obtain the coefficients of legendre expansion
    coeffs = np.dot(arr_legendre, tod_mat.T)

    # plot
    for idet in range(800) : 
    
        model = np.zeros_like(sub_time)
        for deg in range(degree) : 
            model += coeffs[deg,idet]*arr_legendre[deg]/ norm_vector[deg] * dx
        fig = plt.figure()
        plt.plot(sub_time, tod_mat[idet])
        plt.plot(sub_time, model, color="red")
        plt.show()

