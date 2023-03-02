"""
** This is the quaternion code from moby2 **
Quaternions for 3-D coordinate transformation.
Quaternions are stored in numpy arrays with 4 indices in their
right-most dimension.  E.g.
  1 + 2i + 3j + 4k    is expressed    np.array([1,2,3,4])
The Euler quaternion E(i,phi) for rotating by angle phi about axis is,
for example:
 E(1,phi) = (cos(phi/2), sin(phi/2), 0, 0)
The result of applying the rotation to (x,y,z) is
 (0,x',y',z') = E *** (0,x,y,z) *** conj(E)
where *** denotes quaternion multiplication and conj(E) denotes
quaternion conjugation (a+bi+cj+dk -> a-bi-bj-bk).
"""

import numpy as np

def unit(shape=(1,), dtype=float):
    out = np.zeros(shape+(4,), dtype)
    out[...,0] = 1
    return out

def mul(q1, q2, *args):
    qout = q1[...,:1] * q2
    qout[...,1:] += q1[...,1:] * q2[...,:1]
    qout[...,0] -= (q1[...,1:] * q2[...,1:]).sum(axis=-1)
    qout[...,1] += q1[...,2]*q2[...,3] - q1[...,3]*q2[...,2]
    qout[...,2] += q1[...,3]*q2[...,1] - q1[...,1]*q2[...,3]
    qout[...,3] += q1[...,1]*q2[...,2] - q1[...,2]*q2[...,1]
    if len(args) == 0:
        return qout
    return mul(qout, args[0], *args[1:])

def rotate(qr, v):
    """
    Apply the rotation qr to the vector v.
    v may be an array of quaternions (shape = (...,4)) or an array of
    cartesian vectors (shape = (...,3)).
    """
    cartesian = np.asarray(v).shape[-1] == 3
    if cartesian:
        # Do this the most expensive way possible.
        v = np.dot(v, np.diag([0,1,1,1])[1:,:])
    else:
        v = np.asarray(v)
    out = mul(qr, mul(v, conj(qr)))
    if cartesian:
        return out[...,1:].copy()
    return out

def conj(q):
    return q * [1,-1,-1,-1]

def mod(q):
    return (q**2).sum(axis=-1)

def roti(n,phi):
    """
    Euler quaternion, appropriate for rotating by angle phi about axis
    n=(1,2,3).
    """
    phi = np.asarray(phi)/2
    out = np.zeros(np.asarray(phi).shape + (4,))
    out[...,0] = np.cos(phi)
    out[...,n] = np.sin(phi)
    return out

def rots(n_phi_pairs):
    """
    From the provided list of (axis,angle) pairs, construct the
    product of rotations roti(axis0,angle0) *** roti(axis1,angle1) ...
    Because rotation of q by A is achieved through A***q***conj(A),
        rotate( A *** B, q )
    
    is the same as
        rotate( A, rotate(B, q) )
    """
    if len(n_phi_pairs) == 0:
        return unit()
    out = roti(*n_phi_pairs[0])
    for n_phi in n_phi_pairs[1:]:
        out = mul(out, roti(*n_phi))
    return out

def cartesian_R(qr):
    """
    Convert quaternion rotation qr to a cartesian rotation matrix.
    """
    output = np.empty(qr.shape[:-1] + (3,3))
    output[...,:,0] = rotate(qr, [0,1,0,0])[...,1:]
    output[...,:,1] = rotate(qr, [0,0,1,0])[...,1:]
    output[...,:,2] = rotate(qr, [0,0,0,1])[...,1:]
    return output


if __name__ == '__main__':
    for name,axis in [('x', 1), ('y', 2), ('z', 3)]:
        print('Rotating by 90 degrees about %s axis' % name)
        for vi in range(3):
            v = np.array([0,0,0])
            v[vi] = 1
            u = rotate(roti(axis, np.pi/2), v)
            print(v, ' -> ', ' '.join(['%6.3f' % x for x in u]))
        print()
