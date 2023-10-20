import os
import h5py
import numpy as np
import numpy.ma as ma
import argparse
import sys
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import operator_calculations as op_calc
import time
from scipy.sparse import csr_matrix,lil_matrix
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def shuffle_masked(labels):
    segs = op_calc.segment_maskedArray(labels)
    labels_shuffle = ma.zeros(labels.shape,dtype=int)
    labels_shuffle[labels.mask] = ma.masked
    for seg in segs:
        t0,tf=seg
        indices = np.random.randint(t0,tf,tf-t0)
        labels_shuffle[t0:tf] = labels[indices]
    return labels_shuffle

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm_idx',help="worm",default=12,type=int)
    args=parser.parse_args()
    kw = args.worm_idx

    delay_range = np.unique(np.array(np.logspace(0,3.5,100),dtype=int))
    print(delay_range.shape)
    n_clusters = 1000
    f = h5py.File('path_to_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_traj = np.array(f['mask_traj'],dtype=bool)
    f.close()
    labels_traj[mask_traj] = ma.masked
    labels_w = labels_traj.reshape((12,33600))
    labels = labels_w[kw]

    print(labels[:10],labels.shape,labels.shape,flush=True)
    dt = 1/16
    n_shuffle=100
    eigs = np.zeros(delay_range.shape[0])
    eigvals_traj_shuffle = np.zeros((len(delay_range),n_shuffle))
    for kd,delay in enumerate(delay_range):
        try:
            lcs,P=op_calc.transition_matrix(labels,delay,return_connected=True)
            inv_measure = op_calc.stationary_distribution(P)
            final_labels = op_calc.get_connected_labels(labels,lcs)
            R = op_calc.get_reversible_transition_matrix(P)
            eigvals,eigvecs = op_calc.sorted_spectrum(R,k=2)
            eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)
            phi2 = eigfunctions[:,1]
            kmeans_labels = op_calc.optimal_partition(phi2,inv_measure,P,return_rho=False)
            cluster_traj = ma.copy(final_labels)
            cluster_traj[~final_labels.mask] = ma.array(kmeans_labels)[final_labels[~final_labels.mask]]
            cluster_traj[final_labels.mask] = ma.masked
            P_coarse = op_calc.transition_matrix(cluster_traj,delay)
            eigvals = np.linalg.eigvals(P_coarse.todense())
            eigs[kd] = np.min(np.abs(eigvals))
            for ks in range(n_shuffle):
                try:
                    labels_shuffle = shuffle_masked(cluster_traj)
                    P_coarse = op_calc.transition_matrix(labels_shuffle,delay)
                    eigvals,eigvecs = op_calc.sorted_spectrum(P_coarse.todense(),k=2)
                    eigvals_traj_shuffle[kd,ks] = np.min(np.abs(eigvals))
                except:
                    eigvals_traj_shuffle[kd,ks] = np.nan
        except:
            print('Failed to compute for kd={}'.format(kd),flush=True)
            continue
        print(kd,flush=True)
    print(eigvals_traj_shuffle.mean(),flush=True)

    f = h5py.File('path_to_data/kinetic_properties/coarse_eigvals_{}.h5'.format(kw),'w')
    eigvals_ = f.create_dataset('eigs',eigs.shape)
    eigvals_[...]=eigs
    eigvals_s = f.create_dataset('eigvals_traj_shuffle',eigvals_traj_shuffle.shape)
    eigvals_s[...]=eigvals_traj_shuffle
    kds_ = f.create_dataset('delay_range',delay_range.shape)
    kds_[...] = delay_range
    f.close()


if __name__ == "__main__":
    main(sys.argv)
