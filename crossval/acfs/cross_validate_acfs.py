import numpy as np
import numpy.ma as ma
import sys
import os
sys.path.append('/home/a/antonio-costa/TransferOperators/bridging_scales_manuscript/utils')
import operator_calculations as op_calc
import worm_dynamics as worm_dyn
import stats
import clustering_methods as cl
import argparse
import h5py

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


    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    parser.add_argument('-train_ratio','--Train_ratio',help='train ratio',default=0.7,type=float)
    args=parser.parse_args()
    kw,idx = np.array(np.loadtxt('indices.txt')[args.Idx],dtype=int)
    
    print(kw,idx,flush=True)
    
    mat=h5py.File('/bucket/StephensU/antonio/ForagingN2_data/PNAS2011-DataStitched.mat','r')

    refs=list(mat['#refs#'].keys())[1:]
    tseries_w=[ma.masked_invalid(np.array(mat['#refs#'][ref]).T)[:,:5] for ref in refs]
    mat.close()
    frameRate=16.
    dt=1/frameRate
    len_w,dim = tseries_w[0].shape
    n_worms = len(tseries_w)
    
    n_clusters=1000
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_traj = np.array(f['mask_traj'],dtype=bool)
    labels_phspace = ma.array(f['labels_phspace'],dtype=int)
    mask_phspace = np.array(f['mask_phspace'],dtype=bool)
    centers_phspace = np.array(f['centers_phspace'])
    centers_traj = np.array(f['centers_traj'])
    f.close()
    labels_traj[mask_traj] = ma.masked
    labels_phspace[mask_phspace] = ma.masked
    
    labels_w = labels_traj.reshape((n_worms,len_w))
    
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/phspace_K_10_m_7.h5','r')
    traj_matrix = ma.masked_invalid(f['traj_matrix'])
    phspace = ma.array(f['phspace'])
    components = np.array(f['modes'])
    traj_matrix[traj_matrix==0]=ma.masked
    f.close()
    
    traj_matrix_w = traj_matrix.reshape((n_worms,len_w,traj_matrix.shape[1]))
    
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/labels_tree.h5','r')
    delay = int(np.array(f['delay'])[0])
    eigfunctions = np.array(f['eigfunctions'])
    final_labels = ma.masked_invalid(np.array(f['final_labels'],dtype=int))
    final_labels_mask = np.array(f['final_labels_mask'])
    sel = final_labels_mask==1
    final_labels[sel] = ma.masked
    labels_tree = np.array(f['labels_tree'])
    f.close()
    
    kmeans_labels = labels_tree[0,:]
    
   
    wsize = 3360
    t0s_range =  np.arange(0,len_w,wsize)
    labels_seg = ma.zeros((len(t0s_range),wsize),dtype=int)
    tseries_seg = ma.zeros((len(t0s_range),wsize,dim))
    traj_matrix_seg = ma.zeros((len(t0s_range),wsize,traj_matrix.shape[1]))
    for ks,t0 in enumerate(t0s_range):
        tf = t0+wsize
        labels_here = labels_w[kw][t0:tf]
        labels_here[-1] = ma.masked
        tseries_here = tseries_w[kw][t0:tf]
        tseries_here[-1] = ma.masked
        tm_here = traj_matrix_w[kw][t0:tf]
        tm_here[-1] = ma.masked
        labels_seg[ks] = labels_here
        tseries_seg[ks] = tseries_here
        traj_matrix_seg[ks] = tm_here

    train_ratio=args.Train_ratio
    print('train ratio ',train_ratio,flush=True)

    ntrain = int(len(t0s_range)*train_ratio)
    sel = np.zeros(len(labels_seg),dtype='bool')
    sel[np.random.choice(np.arange(0,len(labels_seg)),ntrain,replace=False)]=True
    print(sel.sum(),(~sel).sum())
    labels_train = ma.hstack(labels_seg[sel])
    labels_test = ma.hstack(labels_seg[~sel])
    tseries_train = ma.vstack(tseries_seg[sel])
    tseries_test = ma.vstack(tseries_seg[~sel])
    traj_matrix_train = ma.vstack(traj_matrix_seg[sel])
    traj_matrix_test = ma.vstack(traj_matrix_seg[~sel])

    lcs,P = op_calc.transition_matrix(labels_train,delay,return_connected=True)



    len_sim = int(len(tseries_test)/delay)

    n_sims=100

    states0_sample = np.random.choice(P.shape[0],5*n_sims,replace=False)
    states0 = states0_sample[[lcs[state0] in labels_test for state0 in states0_sample]][:n_sims]

    sims = Parallel(n_jobs=50)(delayed(simulate_parallel)(P,state0,len_sim,lcs)
                               for state0 in states0)

    K_star=11
    ts_sims = Parallel(n_jobs=50)(delayed(sim_ts_parallel)(sims,traj_matrix_train,labels_train,K_star,delay,dim,k_idx) for k_idx in range(n_sims))
    ts_sims = ma.masked_invalid(ts_sims)
    ts_sims[ts_sims==0]=ma.masked

    
    lags = np.arange(0,int(10*frameRate))
    acfs_data = np.zeros((len(lags),dim))
    acfs_sims = np.zeros((n_sims,len(lags),dim))

    print('Computing acfs',flush=True)
    
    for km in range(dim):
        acfs_data[:,km] = stats.acf(tseries_test[:,km],lags)
        for ks in range(n_sims):
            acfs_sims[ks,:,km] = stats.acf(ts_sims[ks][:,km],lags)
     
    print('Saving acfs',flush=True)
    
    f = h5py.File('/flash/StephensU/antonio/Foraging/cross_validate_acfs/acfs_sims_train_ratio_{:.1f}_{}.h5'.format(train_ratio,args.Idx),'w')
    acf_data_ = f.create_dataset('acfs_data',acfs_data.shape)
    acf_data_[...] = acfs_data
    acf_sims_ = f.create_dataset('acfs_sims',acfs_sims.shape)
    acf_sims_[...] = acfs_sims
    l_ = f.create_dataset('lags',lags.shape)
    l_[...] = lags
    ws_ = f.create_dataset('wsize',(1,))
    ws_[...] = wsize
    f.close()
    

if __name__ == "__main__":
    main(sys.argv)