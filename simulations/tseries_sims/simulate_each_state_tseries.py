#data format library
import h5py
#numpy
import sys
import os
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import operator_calculations as op_calc
import worm_dynamics as worm_dyn
import stats
import clustering_methods as cl
import argparse
import scipy.io
from joblib import Parallel, delayed
import numpy as np
import numpy.ma as ma


def simulate(P,state0,iters,lcs):
    states = np.zeros(iters,dtype=int)
    states[0]=state0
    state=state0
    for k in range(1,iters):
        new_state = np.random.choice(np.arange(P.shape[1]),p=list(np.hstack(P[state,:].toarray())))
        state=new_state
        states[k]=state
    return lcs[states]

from joblib import Parallel, delayed

def simulate_parallel(P,state0,len_sim,lcs):
    return simulate(P,state0,len_sim,lcs)


def generate_random_tseries(traj_sim_example,traj_matrix,labels,K_star,delay,dim,len_w):
    kidx=0
    idx0 = traj_sim_example[kidx]
    sel0 = labels==idx0
    ts0 = traj_matrix[sel0].reshape(sel0.sum(),K_star,dim)[:,::-1,:][0]
    ts_all = ma.zeros((len_w,dim))
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

def sim_ts_parallel(sims,traj_matrix,labels,K_star,delay,dim,len_w,k_idx):
    traj_sim_example = sims[k_idx]
    return generate_random_tseries(traj_sim_example,traj_matrix,labels,K_star,delay,dim,len_w)


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
    parser.add_argument('-state','--State_idx',help='State_idx',default=0,type=int)
    args=parser.parse_args()

    worm_idx = 0

    print('Loading data',flush=True)
    mat=h5py.File('path_to_data/PNAS2011-DataStitched.mat','r')
    refs=list(mat['#refs#'].keys())[1:]
    tseries_data=ma.masked_invalid(np.array(mat['#refs#'][refs[worm_idx]]).T)[:,:5]
    mat.close()
    frameRate=16.
    dt=1/frameRate

    len_w,dim = tseries_data.shape
    delay=12
    K_star=11

    f = h5py.File('path_to_data/phspace_K_10_m_7.h5','r')
    traj_matrix = ma.masked_invalid(ma.array(f['traj_matrix']))
    traj_matrix[traj_matrix==0] = ma.masked
    components = np.array(f['modes'])
    f.close()

    n_clusters=1000
    f = h5py.File('path_to_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_labels = np.array(f['mask_traj'],dtype=bool)
    labels_traj[mask_labels] = ma.masked
    f.close()


    f = h5py.File('path_to_data/labels_tree.h5','r')
    delay = int(np.array(f['delay'])[0])
    eigfunctions = np.array(f['eigfunctions'])
    final_labels = ma.masked_invalid(np.array(f['final_labels'],dtype=int))
    final_labels_mask = np.array(f['final_labels_mask'])
    sel = final_labels_mask==1
    final_labels[sel] = ma.masked
    mlg = f['measures']
    measures = []
    for k in np.sort(list(mlg.keys())):
        measures.append(np.array(mlg[str(k)]['measures']))
    labels_tree = np.array(f['labels_tree'],dtype=int)
    f.close()

    eigfunctions_traj = ma.array(eigfunctions)[final_labels,:]
    eigfunctions_traj[final_labels.mask] = ma.masked

    phi2 = eigfunctions[:,1]

    kmeans_labels = labels_tree[5,:]
    cluster_traj = ma.copy(final_labels)
    cluster_traj[~final_labels.mask] = ma.array(kmeans_labels)[final_labels[~final_labels.mask]]
    cluster_traj[final_labels.mask] = ma.masked

    state_idx=args.State_idx
    sel = cluster_traj==state_idx
    labels_state = labels_traj.copy()
    labels_state[~sel] = ma.masked



    print('Simulate symbolic sequence',flush=True)

    lcs,P = op_calc.transition_matrix(labels_state,delay,return_connected=True)
    final_labels_state = op_calc.get_connected_labels(labels_state,lcs)
    len_sim = int(len_w/delay)
    n_sims=1000
    states0 = np.ones(n_sims,dtype=int)*final_labels_state.compressed()[0]
    sims = Parallel(n_jobs=100)(delayed(simulate_parallel)(P,state0,len_sim,lcs) for state0 in states0)

    print(sims[0][:10])
    print(len(sims[0]),flush=True)

    print('Making time series  simulations',flush=True)

    ts_sims = Parallel(n_jobs=100)(delayed(sim_ts_parallel)(sims,traj_matrix,labels_traj,K_star,delay,dim,len_w,k_idx) for k_idx in range(n_sims))

    print('Interpolating gaps and smoothing',flush=True)

    ts_interp_sims=[]
    ts_smooth_sims=[]
    for ks in range(n_sims):
        ts_sim = ts_sims[ks]
        print(ts_sim.shape,flush=True)
        ts_interp = interpolate_gaps(ts_sim)
        ts_smooth = smooth_ts(ts_interp)
        ts_interp_sims.append(ts_interp)
        ts_smooth_sims.append(ts_smooth)
        print(ks,flush=True)
    ts_interp_sims = np.array(ts_interp_sims)
    ts_smooth_sims = np.array(ts_smooth_sims)
    ts_sims = np.array(ts_sims)

    print(ts_interp_sims.shape,ts_smooth_sims.shape,flush=True)
    print('Saving results',flush=True)
    output_path = 'path_to_data'
    f = h5py.File(output_path+'tseries_sims_state_{}.h5'.format(state_idx),'w')
    ts_ = f.create_dataset('ts_sims',ts_sims.shape)
    ts_[...] = ts_sims
    ts_interp_ = f.create_dataset('ts_interp_sims',ts_interp_sims.shape)
    ts_interp_[...] = ts_interp_sims
    ts_smooth_ = f.create_dataset('ts_smooth_sims',ts_smooth_sims.shape)
    ts_smooth_[...] = ts_smooth_sims
    f.close()

if __name__ == "__main__":
    main(sys.argv)



#     #example simulation
#     n_sims = 100
#     sim_random = Parallel(n_jobs=102)(delayed(get_random_state)(traj_matrix,labels_traj,sim)
#                                                 for sim in sims[:n_sims])

#     tseries_sim_random = np.array([sim[:,:5] for sim in sim_random])
#     thetas_sim = ma.array([ts.dot(eigenworms_matrix[:,:5].T) for ts in tseries_sim_random])

#     print('Interpolating simulations',flush=True)
#     ts_interp_sim = np.zeros((len(sim_random),tseries_data.shape[0],tseries_data.shape[1]))
#     for k_idx in range(n_sims):
#         ts_all = sim_random[k_idx].reshape(sim_random[0].shape[0],K_star,dim)
#         ts_w_shape = []
#         weights = []
#         kt=0
#         while kt<len(ts_all):
#             ts_w_shape.append(ts_all[kt][::-1])
#             weights.append(gaussian_x(np.arange(K_star),K_star-1,.5))
#             weights.append(0)
#             ts_w_shape.append(np.zeros(5))
#             kt+=1
#         ts_w_shape = ma.vstack(ts_w_shape)
#         ts_w_shape[ts_w_shape==0] = ma.masked
#         weights = np.hstack(weights)
#         ts_interp = smooth_interp_ts(ts_w_shape,weights)
#         ts_interp_sim[k_idx] = ts_interp
#         print(k_idx,flush=True)
