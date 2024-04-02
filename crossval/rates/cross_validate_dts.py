import numpy as np
import numpy.ma as ma
import sys
import os
sys.path.append('../../utils/bridging_scales_manuscript/utils')
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

def bootstrap_rates(l,n_times,confidence_interval=95):
    per=(100-confidence_interval)/2
    new_means=[]
    for i in range(n_times):
        indices=np.random.choice(range(0,len(l)),len(l))
        new_list=[l[idx] for idx in indices]
        new_means.append(ma.mean(new_list,axis=0))
    new_means=ma.vstack(new_means)
    cil=np.zeros(new_means.shape[1])
    ciu=np.zeros(new_means.shape[1])
    for i in range(new_means.shape[1]):
        cil[i]=np.nanpercentile(1/new_means[:,i].filled(np.nan),per)
        ciu[i]=np.nanpercentile(1/new_means[:,i].filled(np.nan),100-per)
    cil = ma.masked_array(cil, np.isnan(cil))
    ciu = ma.masked_array(ciu, np.isnan(ciu))
    return 1/ma.mean(l,axis=0),cil,ciu


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    parser.add_argument('-train_ratio','--Train_ratio',help='train ratio',default=0.8,type=float)

    args=parser.parse_args()
    kw,idx = np.array(np.loadtxt('indices.txt')[args.Idx],dtype=int)

    print(kw,idx,flush=True)

    mat=h5py.File('../data/ForagingN2/PNAS2011-DataStitched.mat','r')

    refs=list(mat['#refs#'].keys())[1:]
    tseries_w=[ma.masked_invalid(np.array(mat['#refs#'][ref]).T)[:,:5] for ref in refs]
    mat.close()
    frameRate=16.
    dt=1/frameRate
    len_w,dim = tseries_w[0].shape
    n_worms = len(tseries_w)

    n_clusters=1000
    f = h5py.File('../data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
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

    f = h5py.File('../data/ForagingN2/phspace_K_10_m_7.h5','r')
    traj_matrix = ma.masked_invalid(f['traj_matrix'])
    phspace = ma.array(f['phspace'])
    components = np.array(f['modes'])
    traj_matrix[traj_matrix==0]=ma.masked
    f.close()

    traj_matrix_w = traj_matrix.reshape((n_worms,len_w,traj_matrix.shape[1]))


    f = h5py.File('../data/labels_tree/labels_tree.h5','r')
    delay = int(np.array(f['delay'])[0])
    eigfunctions = np.array(f['eigfunctions'])
    final_labels = ma.masked_invalid(np.array(f['final_labels'],dtype=int))
    final_labels_mask = np.array(f['final_labels_mask'])
    sel = final_labels_mask==1
    final_labels[sel] = ma.masked
    labels_tree = np.array(f['labels_tree'])
    f.close()

    kmeans_labels = labels_tree[0,:]

    def get_cluster_traj(final_labels,kmeans_labels):
        cluster_traj = ma.copy(final_labels)
        cluster_traj[~final_labels.mask] = ma.masked_invalid(kmeans_labels)[final_labels[~final_labels.mask]]
        cluster_traj[final_labels.mask] = ma.masked
        return cluster_traj

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

    print(len(sims),len(sims[0]),flush=True)

    dts_sims = [stats.state_lifetime(ma.masked_invalid(get_cluster_traj(ma.masked_invalid(sims[ks]),kmeans_labels)),dt*delay) for ks in range(n_sims)]
    dts_data = stats.state_lifetime(ma.masked_invalid(get_cluster_traj(labels_test[::delay],kmeans_labels)),dt*delay)

    rates_sims = np.zeros((2,3))
    for state_idx in range(2):
        all_sim_rate_means = [1/np.mean(dts_sims[ks][state_idx]) for ks in range(n_sims)]
        mean_sim_rate = 1/np.hstack([dts_sims[ks][state_idx] for ks in range(n_sims)]).mean()
        cil_sim_rate = np.percentile(all_sim_rate_means,2.5)
        ciu_sim_rate = np.percentile(all_sim_rate_means,97.5)
        rates_sims[state_idx] = mean_sim_rate,cil_sim_rate,ciu_sim_rate

    rates_data = np.vstack([np.hstack(bootstrap_rates(dts_data[0],n_times=100)),np.hstack(bootstrap_rates(dts_data[1],n_times=100))])

    print('Saving sims',flush=True)

    f = h5py.File('../data/cross_validate_dts/dts_train_ratio_{:.1f}_{}.h5'.format(train_ratio,args.Idx),'w')
    rates_sims_ = f.create_dataset('rates_sims',rates_sims.shape)
    rates_sims_[...] = rates_sims
    rates_data_ = f.create_dataset('rates_data',rates_data.shape)
    rates_data_[...] = rates_data
    ws_ = f.create_dataset('wsize',(1,))
    ws_[...] = wsize
    f.close()


if __name__ == "__main__":
    main(sys.argv)
