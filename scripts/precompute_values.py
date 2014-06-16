import h5py
import numpy as np
import scipy.linalg
import scipy.spatial.distance as ssd
import argparse

from rapprentice.tps import tps_kernel_matrix 
from rapprentice.registration import loglinspace
from rapprentice import clouds, plotting_plt

import IPython as ipy
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str)
    parser.add_argument('--bend_coef_init', type=float, default=10)
    parser.add_argument('--bend_coef_final', type=float, default=.1)
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--cloud_name', type=str, default='cloud_xyz')
    return parser.parse_args()

def get_lu_decomp(x_na, bend_coef, rot_coef=np.r_[1e-4, 1e-4, 1e-1]):
    """
    precomputes the LU decomposition and other intermediate results needed
    to fit a TPS to x_na with bend_coef

    all thats needed is to compute the righthand side and do a forward solve

    Current practice is to use the explicit inversion, but leaving this around anyways
    """
    K_nn = tps_kernel_matrix(x_na)
    n,d = x_na.shape
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    # QWQ = Q.T.dot(WQ)
    H = Q.T.dot(Q)
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T

    # f = -WQ.T.dot(y_ng)
    # f[1:d+1,0:d] -= np.diag(rot_coefs)
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]

    p, l, u, = scipy.linalg.lu(N.T.dot(H.dot(N)))

    # z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    # x = N.dot(z)

    res_dict = {'p' : p, 'l' : l, 'u' : u, 'N' : N, 'rot_coefs' : rot_coefs}

    return bend_coef, res_dict

def get_inv(x_na, bend_coef, rot_coef=np.r_[1e-4, 1e-4, 1e-1]):
    """
    precomputes the linear operators to solve this linear system. 
    only dependence on data is through the specified targets

    all thats needed is to compute the righthand side and do a forward solve
    """
    K_nn = tps_kernel_matrix(x_na)
    n,d = x_na.shape
    Q = np.c_[np.ones((n,1)), x_na, K_nn]
    # QWQ = Q.T.dot(WQ)
    H = Q.T.dot(Q)
    H[d+1:,d+1:] += bend_coef * K_nn
    rot_coefs = np.ones(d) * rot_coef if np.isscalar(rot_coef) else rot_coef
    H[1:d+1, 1:d+1] += np.diag(rot_coefs)

    A = np.r_[np.zeros((d+1,d+1)), np.c_[np.ones((n,1)), x_na]].T

    # f = -WQ.T.dot(y_ng)
    # f[1:d+1,0:d] -= np.diag(rot_coefs)
    n_cnts = A.shape[0]
    _u,_s,_vh = np.linalg.svd(A.T)
    N = _u[:,n_cnts:]

    h_inv = scipy.linalg.inv(N.T.dot(H.dot(N)))

    # z = np.linalg.solve(N.T.dot(H.dot(N)), -N.T.dot(f))
    # x = N.dot(z)


    ## x = N * (N' * H * N)^-1 * N' * Q' * y + N * (N' * H * N)^-1 * N' * diag(rot_coeffs)
    ## x = proj_mat * y + offset_mat
    proj_mat = N.dot(h_inv.dot(N.T.dot(Q.T)))
    F = np.zeros(Q.T.dot(x_na).shape)
    F[1:d+1,0:d] -= np.diag(rot_coefs)
    offset_mat = N.dot(h_inv.dot(N.T.dot(F)))

    res_dict = {'proj_mat': proj_mat, 'offset_mat' : offset_mat, 'h_inv': h_inv, 'N' : N, 'rot_coefs' : rot_coefs}

    return bend_coef, res_dict


DS_SIZE = 0.025
def downsample_cloud(cloud_xyz):
    return clouds.downsample(cloud_xyz, DS_SIZE)

def main():
    args = parse_arguments()

    f = h5py.File(args.datafile, 'r+')
    
    bend_coefs = np.around(loglinspace(args.bend_coef_init, args.bend_coef_final, args.n_iter), 6)

    for seg_name, seg_info in f.iteritems():
        if 'inv' in seg_info:
            if args.replace:
                del seg_info['inv'] 
                inv_group = seg_info.create_group('inv')
            else:
                inv_group =  seg_info['inv']
        else:
            inv_group = seg_info.create_group('inv')
        ds_key = 'DS_SIZE_{}'.format(DS_SIZE)
        if ds_key in inv_group:
            x_na = inv_group[ds_key][:]
        else:
            x_na = downsample_cloud(seg_info[args.cloud_name][:, :])
            inv_group[ds_key] = x_na

        for bend_coef in bend_coefs:
            if str(bend_coef) in inv_group:
                continue
            
            bend_coef_g = inv_group.create_group(str(bend_coef))
            _, res = get_inv(x_na, bend_coef)
            for k, v in res.iteritems():
                bend_coef_g[k] = v

        if args.verbose:
            print 'segment {}  bend_coef {}'.format(seg_name, bend_coef)

    f.close()

if __name__=='__main__':
    main()
            
