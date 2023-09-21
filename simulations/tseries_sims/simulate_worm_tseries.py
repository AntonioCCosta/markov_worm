#data format library
import h5py
#numpy
import sys
import os
sys.path.append('/home/a/antonio-costa/BehaviorModel/utils/')
import new_op_calc as op_calc
import worm_dynamics as worm_dyn
import stats
import clustering_methods as cl
import argparse
import scipy.io
from joblib import Parallel, delayed
import numpy as np
import numpy.ma as ma

def generate_random_tseries(traj_sim_example,traj_matrix,labels,K_star,delay,dim):
    kidx=0
    idx0 = traj_sim_example[kidx]
    sel0 = labels==idx0
    ts0 = traj_matrix[sel0].reshape(sel0.sum(),K_star,dim)[:,::-1,:][0]
    ts_all = ma.zeros((len(traj_matrix),dim))
    ts_all[:K_star] = ts0
    k=len(ts0)
    for kidx in range(len(traj_sim_example)-1):
        idx1 = traj_sim_example[kidx+1]
        sel1 = labels==idx1
        ts_samples1 = traj_matrix[sel1].reshape(sel1.sum(),K_star,dim)[:,::-1,:]
        idx = np.random.randint(0,len(ts_samples1))
        ts0 = ts_samples1[idx]
        ts_all[k+1:k+1+len(ts0)] = ts0
        k= k+1+len(ts0)
        kidx+=1
    ts_all[ts_all==0]=ma.masked
    return ts_all

from joblib import Parallel, delayed

def sim_ts_parallel(sims,traj_matrix,labels,K_star,delay,dim,k_idx):
    traj_sim_example = sims[k_idx]
    return generate_random_tseries(traj_sim_example,traj_matrix,labels,K_star,delay,dim)


def sim_shuffled_ts_parallel(sims,traj_matrix,labels,K_star,delay,dim,k_idx):
    np.random.seed()
    traj_sim_example = sims[k_idx]
    traj_shuffle = traj_sim_example[np.random.randint(0,len(traj_sim_example),len(traj_sim_example))]
    return generate_random_tseries(traj_shuffle,traj_matrix,labels,K_star,delay,dim)

from scipy import interpolate
def interpolate_gaps(ts_sim):
    ts_interp = np.zeros(ts_sim.shape)
    for kd in range(ts_sim.shape[1]):
        x = np.arange(len(ts_sim))
        y = ts_sim[:,kd]
        sel = ~y.mask
        f = interpolate.interp1d(x[sel], y[sel],kind='cubic',fill_value='extrapolate')
        ts_interp[:,kd] = f(x)
    return ts_interp

from scipy.signal import savgol_filter
def smooth_ts(tseries):
    dim = tseries.shape[1]
    ts_smooth = np.vstack([savgol_filter(tseries[:,kd], 11, 3)  for kd in range(dim)]).T
    return ts_smooth


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-worm','--Worm',help='worm_idx',default=0,type=int)
    args=parser.parse_args()

    worm_idx = args.Worm

    print('Loading data',flush=True)
    mat=h5py.File('/bucket/StephensU/antonio/ForagingN2_data/PNAS2011-DataStitched.mat','r')
    refs=list(mat['#refs#'].keys())[1:]
    tseries_data=ma.masked_invalid(np.array(mat['#refs#'][refs[worm_idx]]).T)[:,:5]
    mat.close()
    frameRate=16.
    dt=1/frameRate

    len_w,dim = tseries_data.shape
    delay=12
    K_star=11

    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/phspace_K_10_m_7.h5','r')
    traj_matrix = ma.masked_invalid(ma.array(f['traj_matrix']))
    traj_matrix[traj_matrix==0] = ma.masked
    components = np.array(f['modes'])
    f.close()

    n_clusters=1000
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels = ma.array(f['labels_traj'],dtype=int)
    mask_labels = np.array(f['mask_traj'],dtype=bool)
    labels[mask_labels] = ma.masked
    f.close()
    
    traj_matrix = traj_matrix[worm_idx*len_w:(worm_idx+1)*len_w]
    labels = labels[worm_idx*len_w:(worm_idx+1)*len_w]
    
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/symbol_sequence_simulations.h5','r')
    sims = np.array(f[str(worm_idx)]['sims'],dtype=int)
    f.close()

    print('Making simulations',flush=True)

    n_sims=1000
    ts_sims = Parallel(n_jobs=100)(delayed(sim_ts_parallel)(sims,traj_matrix,labels,K_star,delay,dim,k_idx)
                               for k_idx in range(n_sims))
    ts_sims = ma.masked_invalid(ts_sims)
    ts_sims[ts_sims==0]=ma.masked
    print('Making shuffled simulations',flush=True)

    ts_sims_shuffled = Parallel(n_jobs=100)(delayed(sim_shuffled_ts_parallel)(sims,traj_matrix,labels,K_star,delay,dim,k_idx)
                               for k_idx in range(n_sims))

    ts_sims_shuffled = ma.masked_invalid(ts_sims_shuffled)
    ts_sims_shuffled[ts_sims_shuffled==0]=ma.masked
    print('Interpolating gaps and smoothing',flush=True)
    
   

    ts_interp_sims=[]
    ts_smooth_sims=[]
    ts_shuffled_interp_sims=[]
    ts_shuffled_smooth_sims=[]
    for ks in range(n_sims):
        ts_sim = ts_sims[ks]
        ts_interp = interpolate_gaps(ts_sim)
        ts_smooth = smooth_ts(ts_interp)
        ts_interp_sims.append(ts_interp)
        ts_smooth_sims.append(ts_smooth)
        
        ts_shuffled_sim = ts_sims_shuffled[ks]
        ts_shuffled_interp = interpolate_gaps(ts_shuffled_sim)
        ts_shuffled_smooth = smooth_ts(ts_shuffled_interp)
        ts_shuffled_interp_sims.append(ts_shuffled_interp)
        ts_shuffled_smooth_sims.append(ts_shuffled_smooth)
        
        
    ts_interp_sims = np.array(ts_interp_sims)
    ts_smooth_sims = np.array(ts_smooth_sims)
    ts_shuffled_interp_sims = np.array(ts_shuffled_interp_sims)
    ts_shuffled_smooth_sims = np.array(ts_shuffled_smooth_sims)        
    print(ts_interp_sims.shape,ts_sims.shape,ts_smooth_sims.shape,ts_shuffled_interp_sims.shape,ts_shuffled_smooth_sims.shape,flush=True)
    
    print('Saving results',flush=True)
    output_path = '/flash/StephensU/antonio/Foraging/animations/'
    f = h5py.File(output_path+'tseries_sims_{}.h5'.format(worm_idx),'w')
    ts_ = f.create_dataset('ts_sims',ts_sims.shape)
    ts_[...] = ts_sims
    ts_interp_ = f.create_dataset('ts_interp_sims',ts_interp_sims.shape)
    ts_interp_[...] = ts_interp_sims
    ts_smooth_ = f.create_dataset('ts_smooth_sims',ts_smooth_sims.shape)
    ts_smooth_[...] = ts_smooth_sims
    tss_ = f.create_dataset('ts_shuffled_sims',ts_sims_shuffled.shape)
    tss_[...] = ts_sims_shuffled
    ts_shuffled_interp_ = f.create_dataset('ts_shuffled_interp_sims',ts_shuffled_interp_sims.shape)
    ts_shuffled_interp_[...] = ts_shuffled_interp_sims
    ts_shuffled_smooth_ = f.create_dataset('ts_shuffled_smooth_sims',ts_shuffled_smooth_sims.shape)
    ts_shuffled_smooth_[...] = ts_shuffled_smooth_sims
    sw = f.create_dataset('smoothing_window',(1,))
    sw[...] = 11
    po = f.create_dataset('poly_order',(1,))
    po[...] = 3
    f.create_group('smoothing with savgol filer')
    f.close()

if __name__ == "__main__":
    main(sys.argv)
    
    