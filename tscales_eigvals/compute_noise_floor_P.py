import numpy as np
import numpy.ma as ma
import argparse
import sys
import time
import h5py
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import operator_calculations as op_calc


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
    parser.add_argument('-kw','--worm',help='worm',default=0,type=int)
    args=parser.parse_args()

    kw = int(args.worm)


    n_clusters=1000
    f = h5py.File('path_to_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_traj = np.array(f['mask_traj'],dtype=bool)
    f.close()

    labels_traj[mask_traj] = ma.masked

    labels_w = labels_traj.reshape((12,33600))

    labels = labels_w[kw]

    n_modes = 30
    dt = 1/16.
    nstates = n_clusters
    delay_range = np.unique(np.array(np.logspace(0,4,200),dtype=int))
    print(delay_range.shape)

    n_shuffle = 100

    ts_traj_shuffle = np.zeros((len(delay_range),n_shuffle))
    eigvals_traj_shuffle = np.zeros((len(delay_range),n_shuffle))
    ts_traj_w = np.zeros((len(delay_range),n_modes))
    eigvals_traj_w = np.zeros((len(delay_range),n_modes))
    for kd,delay in enumerate(delay_range):
        P = op_calc.transition_matrix(labels,delay)
        eigvals,eigvecs = op_calc.sorted_spectrum(P.T,k=2*n_modes+2)
        unique_real_eigvals = np.unique(eigvals[1:].real)[::-1]
        timp = -(delay*dt)/np.log(unique_real_eigvals)
        ts_traj_w[kd,:] = timp[:n_modes]
        eigvals_traj_w[kd,:] = unique_real_eigvals[:n_modes]
        for ks in range(n_shuffle):
            try:
                labels_shuffle = shuffle_masked(labels)
                P = op_calc.transition_matrix(labels_shuffle,delay)
                eigvals,eigvecs = op_calc.sorted_spectrum(P.T,k=2)
                ts_traj_shuffle[kd,ks] = -(delay*dt)/np.log(eigvals[1].real)
                eigvals_traj_shuffle[kd,ks] = eigvals[1].real
            except:
                eigvals_traj_shuffle[kd,ks] = np.nan
        print(kd,delay,flush=True)

    print('Saving results',flush=True)
    f = h5py.File('path_to_data/tscales_noise_floor_P/results_{}.h5'.format(kw),'w')
    ts_s = f.create_dataset('ts_traj_shuffle',ts_traj_shuffle.shape)
    ts_s[...] = ts_traj_shuffle
    eigs_s = f.create_dataset('eigvals_traj_shuffle',eigvals_traj_shuffle.shape)
    eigs_s[...] = eigvals_traj_shuffle
    ts_w = f.create_dataset('ts_traj_w',ts_traj_w.shape)
    ts_w[...] = ts_traj_w
    eigs_w = f.create_dataset('eigvals_traj_w',eigvals_traj_w.shape)
    eigs_w[...] = eigvals_traj_w
    ds = f.create_dataset('delay_range',delay_range.shape)
    ds[...] = delay_range
    n = f.create_dataset('n_clusters',(1,))
    n[...] = n_clusters
    f.close()

if __name__ == "__main__":
    main(sys.argv)
