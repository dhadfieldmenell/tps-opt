#!/usr/bin/env python

from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
import h5py

import pycuda.driver as drv
import pycuda.autoinit
from pycuda import gpuarray
from scikits.cuda import linalg
linalg.init()

from tps import tps_kernel_matrix
from transformations import unit_boxify
from culinalg_exts import dot_batch_nocheck, get_gpu_ptrs
from precompute import downsample_cloud, batch_get_sol_params
from cuda_funcs import init_prob_nm, norm_prob_nm, get_targ_pts, check_cuda_err, fill_mat

import IPython as ipy
from pdb import pm, set_trace
import time

N_ITER_CHEAP = 10
EM_ITER_CHEAP = 1
DEFAULT_LAMBDA = (10, .1)
MAX_CLD_SIZE = 150
DATA_DIM = 3
DS_SIZE = 0.025

class Globals:
    sync = False

def sync():
    if Globals.sync:
        check_cuda_err()

def loglinspace(a,b,n):
    "n numbers between a to b (inclusive) with constant ratio between consecutive numbers"
    return np.exp(np.linspace(np.log(a),np.log(b),n))    

def gpu_pad(x, shape, dtype=np.float32):
    (m, n) = x.shape
    if m > shape[0] or n > shape[1]:
        raise ValueError("Cannot Pad Beyond Normal Dimension")
    x_new = np.zeros(shape, dtype=dtype)    
    x_new[:m, :n] = x
    return gpuarray.to_gpu(x_new)

class GPUContext(object):
    """
    Class to contain GPU arrays
    """
    def __init__(self, bend_coeffs = None):
        if bend_coeffs is None:
            lambda_init, lambda_final = DEFAULT_LAMBDA
            bend_coeffs = np.around(loglinspace(lambda_init, lambda_final, N_ITER_CHEAP), 6)
        self.bend_coeffs = bend_coeffs
        self.ptrs_valid = False
        self.N = 0

        self.tps_params     = []
        self.tps_param_ptrs = None
        self.trans_d        = []
        self.trans_d_ptrs   = None
        self.lin_dd         = []
        self.lin_dd_ptrs    = None
        self.w_nd           = []
        self.w_nd_ptrs      = None
        """
        TPS PARAM FORMAT
        [      np.zeros(DATA_DIM)      ]   [  trans_d  ]   [1 x d]
        [       np.eye(DATA_DIM)       ] = [  lin_dd   ] = [d x d]
        [np.zeros((np.zeros, DATA_DIM))]   [   w_nd    ]   [n x d]
        """
        self.default_tps_params = gpuarray.zeros((DATA_DIM + 1 + MAX_CLD_SIZE, DATA_DIM), np.float32)
        self.default_tps_params[1:DATA_DIM+1, :].set(np.eye(DATA_DIM, dtype=np.float32))

        self.proj_mats       = dict([(b, []) for b in bend_coeffs])
        self.proj_mat_ptrs   = dict([(b, None) for b in bend_coeffs])
        self.offset_mats     = dict([(b, []) for b in bend_coeffs])
        self.offset_mat_ptrs = dict([(b, None) for b in bend_coeffs])

        self.pts             = []
        self.pt_ptrs         = None
        self.kernels         = []
        self.kernel_ptrs     = None
        self.pts_w           = []
        self.pt_w_ptrs       = None
        self.pts_t           = []
        self.pts_t_ptrs      = None
        self.dims            = []
        self.dims_gpu        = None

        self.corr_cm         = []
        self.corr_cm_ptrs    = None
        self.corr_rm         = []
        self.corr_rm_ptrs    = None
        self.seg_names       = []

    def reset_tps_params(self):
        """
        sets the tps params to be identity
        """
        for p in self.tps_params:
            drv.memcpy_dtod_async(p.gpudata, self.default_tps_params.gpudata, p.mem_size)            
    def set_tps_params(self, vals):
        for d, s in zip(self.tps_params, vals):
            drv.memcpy_dtod_async(d.gpudata, s.gpudata, d.mem_size)            

    def check_cld(self, cloud_xyz):
        if cloud_xyz.dtype != np.float32:
            raise TypeError("only single precision operations supported")
        if cloud_xyz.shape[0] > MAX_CLD_SIZE:
            raise ValueError("cloud size exceeds {}".format(MAX_CLD_SIZE))
        if cloud_xyz.shape[1] != DATA_DIM:
            raise ValueError("point cloud must have column dimension {}".format(DATA_DIM))
    # @profile
    def get_sol_params(self, cld):
        self.check_cld(cld)
        K = tps_kernel_matrix(cld)
        proj_mats   = {}
        offset_mats = {}
        (proj_mats_arr, _), (offset_mats_arr, _) = batch_get_sol_params(cld, K, self.bend_coeffs)
        for i, b in enumerate(self.bend_coeffs):
            proj_mats[b]   = proj_mats_arr[i]
            offset_mats[b] = offset_mats_arr[i]
        return proj_mats, offset_mats, K

    def add_cld(self, name, proj_mats, offset_mats, cloud_xyz, kernel, update_ptrs = False):
        """
        adds a new cloud to our context for batch processing
        """
        self.check_cld(cloud_xyz)
        self.ptrs_valid = False
        self.N += 1
        self.seg_names.append(name)
        self.tps_params.append(self.default_tps_params.copy())
        self.trans_d.append(self.tps_params[-1][:1, :1])
        self.lin_dd.append(self.tps_params[-1][1:DATA_DIM+1, :])
        self.w_nd.append(self.tps_params[-1][DATA_DIM + 1:, :])
        n = cloud_xyz.shape[0]
        
        for b in self.bend_coeffs:
            proj_mat   = proj_mats[b]
            offset_mat = offset_mats[b]
            self.proj_mats[b].append(gpu_pad(proj_mat, (MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE)))

            if offset_mat.shape != (n + DATA_DIM + 1, DATA_DIM):
                raise ValueError("Offset Matrix has incorrect dimension")
            self.offset_mats[b].append(gpu_pad(offset_mat, (MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM)))


        if n > MAX_CLD_SIZE or cloud_xyz.shape[1] != DATA_DIM:
            raise ValueError("cloud_xyz has incorrect dimension")
        self.pts.append(gpu_pad(cloud_xyz, (MAX_CLD_SIZE, DATA_DIM)))
        if kernel.shape != (n, n):
            raise ValueError("dimension mismatch b/t kernel and cloud")
        self.kernels.append(gpu_pad(kernel, (MAX_CLD_SIZE, MAX_CLD_SIZE)))
        self.dims.append(n)

        self.pts_w.append(gpuarray.zeros_like(self.pts[-1]))
        self.pts_t.append(gpuarray.zeros_like(self.pts[-1]))
        self.corr_cm.append(gpuarray.zeros((MAX_CLD_SIZE, MAX_CLD_SIZE), np.float32))
        self.corr_rm.append(gpuarray.zeros((MAX_CLD_SIZE, MAX_CLD_SIZE), np.float32))

        if update_ptrs:
            self.update_ptrs()

    def update_ptrs(self):
        self.tps_param_ptrs = get_gpu_ptrs(self.tps_params)
        self.trans_d_ptrs   = get_gpu_ptrs(self.trans_d)
        self.lin_dd_ptrs    = get_gpu_ptrs(self.lin_dd)
        self.w_nd_ptrs      = get_gpu_ptrs(self.w_nd)
        
        for b in self.bend_coeffs:
            self.proj_mat_ptrs[b]   = get_gpu_ptrs(self.proj_mats[b])
            self.offset_mat_ptrs[b] = get_gpu_ptrs(self.offset_mats[b])

        self.pt_ptrs      = get_gpu_ptrs(self.pts)
        self.kernel_ptrs  = get_gpu_ptrs(self.kernels)
        self.pt_w_ptrs    = get_gpu_ptrs(self.pts_w)
        self.pt_t_ptrs    = get_gpu_ptrs(self.pts_t)
        self.corr_cm_ptrs = get_gpu_ptrs(self.corr_cm)
        self.corr_rm_ptrs = get_gpu_ptrs(self.corr_rm)
        
        self.dims_gpu = gpuarray.to_gpu(np.array(self.dims, dtype=np.int32))
        self.ptrs_valid = True

    def read_h5(self, fname):
        f = h5py.File(fname, 'r')
        for seg_name, seg_info in f.iteritems():
            if 'inv' not in seg_info:
                raise KeyError("H5 File does not have precomputed values")
            seg_info = seg_info['inv']

            proj_mats   = {}
            offset_mats = {}
            for b in self.bend_coeffs:
                k = str(b)
                if k not in seg_info:
                    raise KeyError("H5 File {} bend coefficient {}".format(seg_name, k))
                proj_mats[b] = seg_info[k]['proj_mat'][:]
                offset_mats[b] = seg_info[k]['offset_mat'][:]

            ds_key    = 'DS_SIZE_{}'.format(DS_SIZE)
            cloud_xyz = seg_info[ds_key]['scaled_cloud_xyz']
            kernel    = seg_info[ds_key]['scaled_K_nn']
            self.add_cld(seg_name, proj_mats, offset_mats, cloud_xyz, kernel)
        f.close()
        self.update_ptrs()
    # @profile
    def setup_tgt_ctx(self, cloud_xyz):
        """
        returns a GPUContext where all the clouds are cloud_xyz
        and matched in length with this contex

        assumes cloud_xyz is already downsampled and scaled
        """        
        tgt_ctx = TgtContext(self)
        tgt_ctx.set_cld(cloud_xyz)
        return tgt_ctx

    # @profile
    def transform_points(self):
        """
        computes the warp of self.pts under the current tps params
        """
        fill_mat(self.pt_w_ptrs, self.trans_d_ptrs, self.dims_gpu, self.N)
        dot_batch_nocheck(self.pts,         self.lin_dd,      self.pts_w,
                          self.pt_ptrs,     self.lin_dd_ptrs, self.pt_w_ptrs) 
        dot_batch_nocheck(self.kernels,     self.w_nd,        self.pts_w,
                          self.kernel_ptrs, self.w_nd_ptrs,   self.pt_w_ptrs) 
        sync()
    # @profile
    def get_target_points(self, other, outlierprior, outlierfrac, outliercutoff, T, norm_iters = 10):
        """
        computes the target points for self and other
        using the current warped points for both                
        """
        init_prob_nm(self.pt_ptrs, other.pt_ptrs, 
                     self.pt_w_ptrs, other.pt_w_ptrs, 
                     self.dims_gpu, other.dims_gpu,
                     self.N, outlierprior, T, 
                     self.corr_cm_ptrs, self.corr_rm_ptrs)
        sync()
        norm_prob_nm(self.corr_cm_ptrs, self.corr_rm_ptrs, 
                     self.dims_gpu, other.dims_gpu, self.N, outlierfrac, norm_iters)
        sync()
        get_targ_pts(self.pt_ptrs, other.pt_ptrs,
                     self.pt_w_ptrs, other.pt_w_ptrs,
                     self.corr_cm_ptrs, self.corr_rm_ptrs,
                     self.dims_gpu, other.dims_gpu, 
                     outliercutoff, self.N,
                     self.pt_t_ptrs, other.pt_t_ptrs)
        sync()
    # @profile
    def update_transform(self, b):
        """
        computes the TPS associated with the current target pts
        """
        self.set_tps_params(self.offset_mats[b])
        dot_batch_nocheck(self.proj_mats[b],     self.pts_t,     self.tps_params,
                          self.proj_mat_ptrs[b], self.pt_t_ptrs, self.tps_param_ptrs)
        sync()

class TgtContext(GPUContext):
    """
    specialized class to handle the case where we are
    mapping to a single target cloud --> only allocate GPU Memory once
    """
    def __init__(self, src_ctx):
        GPUContext.__init__(self, src_ctx.bend_coeffs)
        self.src_ctx = src_ctx
        ## just setup with 0's
        tgt_cld = np.zeros((MAX_CLD_SIZE, DATA_DIM), np.float32)
        proj_mats = dict([(b, np.zeros((MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE), np.float32)) 
                          for b in self.bend_coeffs])
        offset_mats = dict([(b, np.zeros((MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM), np.float32)) 
                            for b in self.bend_coeffs])
        tgt_K = np.zeros((MAX_CLD_SIZE, MAX_CLD_SIZE), np.float32)
        for n in src_ctx.seg_names:
            name = "{}_tgt".format(n)
            GPUContext.add_cld(self, name, proj_mats, offset_mats, tgt_cld, tgt_K)
        GPUContext.update_ptrs(self)
    def add_cld(self, name, proj_mats, offset_mats, cloud_xyz, kernel, update_ptrs = False):
        raise NotImplementedError("not implemented for TgtConext")
    def update_ptrs(self):
        raise NotImplementedError("not implemented for TgtConext")
    # @profile
    def set_cld(self, cld):
        """
        sets the cloud for this appropriately
        won't allocate any new memory
        """                          
        proj_mats, offset_mats, K = self.get_sol_params(cld)
        
        self.pts         = [gpu_pad(cld, (MAX_CLD_SIZE, DATA_DIM))]
        self.kernels     = [gpu_pad(K, (MAX_CLD_SIZE, MAX_CLD_SIZE))]
        self.proj_mats   = dict([(b, [gpu_pad(p.get(), (MAX_CLD_SIZE + DATA_DIM + 1, MAX_CLD_SIZE))])
                                 for b, p in proj_mats.iteritems()])
        self.offset_mats = dict([(b, [gpu_pad(p.get(), (MAX_CLD_SIZE + DATA_DIM + 1, DATA_DIM))]) 
                                 for b, p in offset_mats.iteritems()])

        self.pt_ptrs.fill(int(self.pts[0].gpudata))
        self.kernel_ptrs.fill(int(self.kernels[0].gpudata))
        for b in self.bend_coeffs:
            self.proj_mat_ptrs[b].fill(int(self.proj_mats[b][0].gpudata))
            self.offset_mat_ptrs[b].fill(int(self.offset_mats[b][0].gpudata))
            
        
    

# @profile
def batch_tps_rpm_bij(src_ctx, tgt_ctx, T_init = 4e-1, T_final = 4e-4, 
                      outlierfrac = 1e-2, outlierprior = 1e-1, outliercutoff = 1e-2, em_iter = EM_ITER_CHEAP):
    """
    computes tps rpm for the clouds in src and tgt in batch
    TODO: Fill out comment cleanly
    """
    ##TODO: add check to ensure that src_ctx and tgt_ctx are formatted properly
    n_iter = len(src_ctx.bend_coeffs)
    T_vals = loglinspace(T_init, T_final, n_iter
)
    src_ctx.reset_tps_params()
    tgt_ctx.reset_tps_params()
    for i, b in enumerate(src_ctx.bend_coeffs):
        T = T_vals[i]
        for _ in range(em_iter):
            src_ctx.transform_points()
            tgt_ctx.transform_points()

            src_ctx.get_target_points(tgt_ctx, outlierprior, outlierfrac, outliercutoff, T)

            src_ctx.update_transform(b)
            tgt_ctx.update_transform(b)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='../data/actions.h5')
    parser.add_argument("--sync", action='store_true')
    return parser.parse_args()
# @profile
def main():
    args = parse_arguments()
    Globals.sync = args.sync
    src_ctx = GPUContext()
    src_ctx.read_h5(args.input_file)
    f = h5py.File(args.input_file, 'r')    
    tgt_cld = downsample_cloud(f['demo1-seg00']['cloud_xyz'][:])
    f.close()
    scaled_tgt_cld, _ = unit_boxify(tgt_cld)
    tgt_ctx = TgtContext(src_ctx)
    times = []
    for i in range(10):
        start = time.time()
        tgt_ctx.set_cld(scaled_tgt_cld)
        batch_tps_rpm_bij(src_ctx, tgt_ctx)
        tgt_ctx.tps_params[0].get()
        time_taken = time.time() - start
        times.append(time_taken)
        print "Batch Computation Complete, Time Taken is {}".format(time_taken)
    print "Mean Compute Time is {}".format(np.mean(times))

if __name__ == "__main__":
    main()
