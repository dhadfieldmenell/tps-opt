import numpy as np
from rapprentice import tps, clouds
from tps import tps_kernel_matrix2
import fastps
import IPython as ipy
import h5py, sys
import scipy.spatial.distance as ssd

DIM = 3
stride = 150
DS_SIZE = 0.025
def downsample_cloud(cloud):
    d = DIM
    cloud_xyz = cloud[:,:d]
    cloud_xyz_downsamp = clouds.downsample(cloud_xyz, DS_SIZE)
    return cloud_xyz_downsamp
# @profile
def test_prob_nm_gen(x_vals, xw_vals, y_vals, yw_vals, x_dims, y_dims, prob_nm_py, outlier_prior, reg):
    print "testing generation"
    max_dim = np.max(np.c_[x_dims, y_dims])
    if max_dim > stride:
        raise Exception, "Matrix Size Exceeds Max Dimensions"
    x_all = np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    y_all = np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    xw_all= np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    yw_all= np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    for i in range(len(x_vals)):
        x_all[i, :x_dims[i], :] = x_vals[i]
        y_all[i, :y_dims[i], :] = y_vals[i]
        xw_all[i, :x_dims[i], :] = xw_vals[i]
        yw_all[i, :y_dims[i], :] = yw_vals[i]

    prob_nm_gpu = fastps.prob_nm(x_all.flatten(), y_all.flatten(), xw_all.flatten(), yw_all.flatten(), 
                                 x_dims, y_dims, len(x_vals), 
                                 max_dim, outlier_prior, reg)
    for i in range(len(x_vals)):
        prob_nm_offset = i * max_dim * max_dim
        size = (x_dims[i] + 1) * (y_dims[i] + 1)
        p_nm_py = prob_nm_py[i]
        p_nm_gpu = prob_nm_gpu[prob_nm_offset:prob_nm_offset + size].reshape(p_nm_py.shape)
        if np.any(np.isnan(p_nm_py - p_nm_gpu)) or np.linalg.norm(p_nm_py - p_nm_gpu) > 1e-4:
            diff = np.abs(p_nm_py - p_nm_gpu)
            nz = np.nonzero(diff > .001)
            nan = np.isnan(p_nm_gpu)
            print 'nonzero differences', nz
            print 'nan values', np.nonzero(nan)            
            print 'difference norm', np.linalg.norm(p_nm_py - p_nm_gpu)
            # print "GPU Results Do Not Match PY Results"
            x = x_vals[i]
            y = y_vals[i]
            d = np.linalg.norm(x[0, :] - y[0, :])
            ipy.embed()
            sys.exit(1)
    print "PROB_NM GPU CALCULATION SUCCEEDED!!!"            
# @profile
def test_prob_nm_norm(x_vals, xw_vals, y_vals, yw_vals, x_dims, y_dims, prob_nm_py, 
                      outlier_prior, reg, outlier_frac, norm_iters):
    print "testing normalization"
    max_dim = np.max(np.c_[x_dims, y_dims])
    if max_dim > stride:
        raise Exception, "Matrix Size Exceeds Max Dimensions"
    x_all = np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    y_all = np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    xw_all= np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    yw_all= np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    for i in range(len(x_vals)):
        x_all[i, :x_dims[i], :] = x_vals[i]
        y_all[i, :y_dims[i], :] = y_vals[i]
        xw_all[i, :x_dims[i], :] = xw_vals[i]
        yw_all[i, :y_dims[i], :] = yw_vals[i]

    prob_nm_gpu = fastps.prob_norm_nm(x_all.flatten(), y_all.flatten(), 
                                      xw_all.flatten(), yw_all.flatten(), 
                                      x_dims, y_dims, len(x_vals), 
                                      max_dim, outlier_prior, reg,
                                      outlier_frac, norm_iters)
    for i in range(len(x_vals)):
        prob_nm_offset = i * max_dim * max_dim
        size = (x_dims[i] + 1) * (y_dims[i] + 1)
        p_nm_py = prob_nm_py[i]
        p_nm_gpu = prob_nm_gpu[prob_nm_offset:prob_nm_offset + size].reshape(p_nm_py.shape)
        ## Max must equal max
        py_max = np.r_[np.argmax(p_nm_py, axis=1), np.argmax(p_nm_py, axis=0)]
        gpu_max = np.r_[np.argmax(p_nm_gpu, axis=1), np.argmax(p_nm_gpu, axis=0)]
        nan = np.isnan(p_nm_gpu)
        success = np.sum(py_max == gpu_max) / float(sum(p_nm_py.shape))
        tgt_rate = .8
        if np.any(nan) or success < tgt_rate:
            diff = np.abs(p_nm_py - p_nm_gpu)
            nz = np.nonzero(diff > 1e-3)


            print 'nonzero differences', nz
            print 'nan values', np.nonzero(nan)            
            print 'difference norm', np.linalg.norm(p_nm_py - p_nm_gpu)
            # print "GPU Results Do Not Match PY Results"
            x = x_vals[i]
            y = y_vals[i]
            d = np.linalg.norm(x[0, :] - y[0, :])
            ipy.embed()
            sys.exit(1)
    print "PROB_NM GPU NORMALIZATION SUCCEEDED MAX AGREEMENT IS {}!!!".format(tgt_rate)
# @profile
def test_get_targ_pts(x_vals, xw_vals, y_vals, yw_vals, x_dims, y_dims, prob_nm_py, 
                      outlier_prior, reg, outlier_frac, norm_iters, outlier_cutoff = 1e-2):
    print "testing get targ pts"
    max_dim = np.max(np.c_[x_dims, y_dims])
    if max_dim > stride:
        raise Exception, "Matrix Size Exceeds Max Dimensions"
    x_all = np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    y_all = np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    xw_all= np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    yw_all= np.zeros((len(x_vals), max_dim, DIM), dtype='float32')
    for i in range(len(x_vals)):
        x_all[i, :x_dims[i], :] = x_vals[i]
        y_all[i, :y_dims[i], :] = y_vals[i]
        xw_all[i, :x_dims[i], :] = xw_vals[i]
        yw_all[i, :y_dims[i], :] = yw_vals[i]

    # prob_nm_gpu = fastps.prob_norm_nm(x_all.flatten(), y_all.flatten(), 
    #                                   xw_all.flatten(), yw_all.flatten(), 
    #                                   x_dims, y_dims, len(x_vals), 
    #                                   max_dim, outlier_prior, reg,
    #                                   outlier_frac, norm_iters)
    xt_gpu, yt_gpu, prob_nm_gpu = fastps.get_targ_pts(x_all.flatten(), y_all.flatten(), 
                                                      xw_all.flatten(), yw_all.flatten(), 
                                                      x_dims, y_dims, len(x_vals), 
                                                      max_dim, outlier_prior, reg,
                                                      outlier_frac, norm_iters, outlier_cutoff)
    for i, x_nd in enumerate(x_vals):
        y_md = y_vals[i]
        prob_nm_offset = i * max_dim * max_dim
        data_offset = i * max_dim * DIM        
        size = (x_dims[i] + 1) * (y_dims[i] + 1)
        x_size = max_dim * DIM
        y_size = max_dim * DIM
        xt_i = np.reshape(xt_gpu[data_offset:data_offset + x_size], (max_dim, DIM), order='F')
        xt_i = xt_i[:x_dims[i], :]
        yt_i = yt_gpu[data_offset:data_offset + y_size].reshape((max_dim, DIM), order='F')
        yt_i = yt_i[:y_dims[i], :]
        p_nm_py = prob_nm_py[i]
        corr_nm = prob_nm_gpu[prob_nm_offset:prob_nm_offset + size].reshape(p_nm_py.shape)[:-1, :-1]

        wt_n = corr_nm.sum(axis=1)

       # set outliers to warp to their most recent warp targets
        inlier = wt_n > outlier_cutoff
        xtarg_nd = np.empty(x_nd.shape)
        xtarg_nd[inlier, :] = (corr_nm/wt_n[:,None]).dot(y_md)[inlier, :]
        xtarg_nd[~inlier, :] = xw_vals[i][~inlier, :] 

        wt_m = corr_nm.sum(axis=0)
        inlier = wt_m > outlier_cutoff
        ytarg_md = np.empty(y_md.shape)
        ytarg_md[inlier, :] = (corr_nm.T/wt_m[:,None]).dot(x_nd)[inlier, :]
        ytarg_md[~inlier, :] = yw_vals[i][~inlier, :] 
        diff = xt_i - xtarg_nd
        for j in range(diff.shape[0]):
            if np.sqrt(np.linalg.norm(diff[j])) > 5 * 1e-3:
                print "X TARGET POINTS ARE MORE THAN 5 mm apart\n"
                ipy.embed()
                sys.exit(1)
        diff = yt_i - ytarg_md
        for j in range(diff.shape[0]):
            if np.sqrt(np.linalg.norm(diff[j])) > 5 * 1e-3:
                print "Y TARGET POINTS ARE MORE THAN 5 mm apart\n"
                ipy.embed()
                sys.exit(1)
    print "GET TARG PTS TEST SUCCEEDED"


    


# @profile
def tps_rpm_em_all(x_vals, y_vals, tps_params = None, outlier_prior = .1, reg = .01, 
                   outlierfrac = .01, norm_iter=10):
    if not tps_params:
        tps_params = []
        for i, x in enumerate(x_vals):
            cur_params = {}
            y = y_vals[i]
            fw_k_mat = tps_kernel_matrix2(x, x)
            inv_k_mat = tps_kernel_matrix2(y, y)
            cur_params['fw_kernel'] = fw_k_mat
            cur_params['fw_lin_ag'] = np.eye(DIM)
            cur_params['fw_trans_g'] = np.zeros(DIM)
            cur_params['fw_w_ng'] = np.zeros((x.shape[0], DIM))
            cur_params['inv_kernel'] = inv_k_mat
            cur_params['inv_lin_ag'] = np.eye(DIM)
            cur_params['inv_trans_g'] = np.zeros(DIM)
            cur_params['inv_w_ng'] = np.zeros((y.shape[0], DIM))
            tps_params.append(cur_params)
    x_warped = []
    y_warped = []
    x_dims = []
    y_dims = []
    prob_nm_py = []

    for i, x in enumerate(x_vals):
        cur_params = tps_params[i]
        y = y_vals[i]
        fw_warp = np.dot(cur_params['fw_kernel'], cur_params['fw_w_ng']) + \
            np.dot(x, cur_params['fw_lin_ag']) + cur_params['fw_trans_g'][None, :]
        inv_warp = np.dot(cur_params['inv_kernel'], cur_params['inv_w_ng']) + \
            np.dot(y, cur_params['inv_lin_ag']) + cur_params['inv_trans_g'][None, :]
        x_warped.append(fw_warp)
        y_warped.append(inv_warp)
        
        fwddist_nm = ssd.cdist(fw_warp, y,'euclidean')
        invdist_nm = ssd.cdist(x, inv_warp,'euclidean')
        prob_nm = np.zeros((x.shape[0]+1, y.shape[0]+1))
        prob_nm[:-1, :-1] = np.exp( -(fwddist_nm + invdist_nm) / (2*reg) )
        prob_nm[-1, :] = outlier_prior
        prob_nm[:, -1] = outlier_prior
        prob_nm[-1, -1] = outlier_prior * np.sqrt(x.shape[0] * y.shape[0])
        prob_nm_py.append(prob_nm)

        x_dims.append(x.shape[0])
        y_dims.append(y.shape[0])
    # test_prob_nm_gen(x_vals, x_warped, y_vals, y_warped, x_dims, y_dims, prob_nm_py, outlier_prior, reg)
    for i, p_nm in enumerate(prob_nm_py):
        (n, m) = p_nm.shape
        a_N = np.ones((n),'f4')
        a_N[n-1] = (m-1)*outlierfrac
        b_M = np.ones((m),'f4')
        b_M[m-1] = (n-1)*outlierfrac
    
        r_N = np.ones(n,'f4')

        for _ in xrange(norm_iter):
            c_M = b_M/r_N.dot(p_nm)
            r_N = a_N/p_nm.dot(c_M)

        p_nm *= r_N[:,None]
        p_nm *= c_M[None,:]
    test_get_targ_pts(x_vals, x_warped, y_vals, y_warped, x_dims, y_dims, prob_nm_py, 
                      outlier_prior, reg, outlierfrac, norm_iter)

    

def tps_rpm_bij(x_nd, y_md, n_iter = 20, reg_init = .1, reg_final = .001, rad_init = .1, rad_final = .005, rot_reg = 1e-3, 
            plotting = False, plot_cb = None, old_xyz=None, new_xyz=None, f_init = None, g_init = None, return_corr = False):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    """
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)
    if not f_init:
        f = ThinPlateSpline(d)
        f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0)
    else:
        f = f_init
    if not f_init:
        g = ThinPlateSpline(d)
        g.trans_g = -f.trans_g
    else:
        g = g_init


    # r_N = None
    
    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, 1e-1, 1e-2)
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)


        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        
        if (plotting and i == n_iter -1) or (plotting and i%plotting==0):
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f, old_xyz, new_xyz, last_one=(i == n_iter -1))
        
        f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
        g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)

    f._cost = tps.tps_cost(f.lin_ag, f.trans_g, f.w_ng, f.x_na, xtarg_nd, regs[i], wt_n=wt_n)/wt_n.mean()
    g._cost = tps.tps_cost(g.lin_ag, g.trans_g, g.w_ng, g.x_na, ytarg_md, regs[i], wt_n=wt_m)/wt_m.mean()
    if return_corr:
        return f, g, corr_nm
    return f,g

REG_CLOUDS_F = '/home/dhm/src/tps-opt/data/registration_clouds.h5'
ACTION_F = '/home/dhm/src/tps-opt/data/actions.h5'

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--not_actionsf", action='store_true')
    parser.add_argument("--small", action='store_true')
    parser.add_argument("--incr_size", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_arguments()
    if not args.not_actionsf:
        infile = h5py.File(ACTION_F, 'r')
        tgt_clds = [downsample_cloud(x['cloud_xyz'][:]) for x in infile.values()]
        src_clds = [downsample_cloud(infile['demo1-seg00']['cloud_xyz'][:]) for _ in infile]
    else:
        infile = h5py.File(REG_CLOUDS_F, 'r')
        tgt_clds = [downsample_cloud(y[:]) for x in infile.values() for y in x['target_clouds'].values() ]
        src_clds = [downsample_cloud(x['source_cloud'][:]) for x in infile.values() for _ in x['target_clouds'].values()]
        if args.incr_size:
            tgt_clds = [x for x in args.incr_size * tgt_clds]
            src_clds = [x for x in args.incr_size * src_clds]
        if args.small:
            tgt_clds = [downsample_cloud(x[:]) for x in infile['0']['target_clouds'].values()]
            src_clds = [downsample_cloud(infile['0']['source_cloud'][:]) for _ in infile['0']['target_clouds'].values()]
        

    
    tps_rpm_em_all(tgt_clds, src_clds)

if __name__=='__main__':
    main()
