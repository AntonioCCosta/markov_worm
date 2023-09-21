#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
# %matplotlib notebook
import sys
import os
sys.path.append('/home/a/antonio-costa/bridging_scales_manuscript/utils')
import operator_calculations as op_calc
import argparse

def subdivide_state_optimal(state_to_split,phi2,kmeans_labels,inv_measure,P,indices):
    kmeans_labels[indices] = op_calc.optimal_partition(phi2,inv_measure,P,return_rho=False)+np.max(kmeans_labels)+10
    #check if there were no labels left behind because they were not connected components
    mask = np.ones(kmeans_labels.shape[0],dtype=bool)
    mask[indices] = False
    if np.any(kmeans_labels[mask]==state_to_split):
        idx = np.arange(len(kmeans_labels))[mask][np.where(kmeans_labels[mask]==state_to_split)[0]]
        kmeans_labels[idx] = np.nan

    final_kmeans_labels = np.zeros(kmeans_labels.shape)
    sel = ~np.isnan(kmeans_labels)
    final_kmeans_labels[np.isnan(kmeans_labels)] = np.nan
    for new_idx,label in enumerate(np.sort(np.unique(kmeans_labels[sel]))):
        final_kmeans_labels[kmeans_labels==label]=new_idx
    return final_kmeans_labels

def recursive_partitioning_optimal(final_labels,delay,phi2,inv_measure,P,n_final_states,save=False):
    c_range,rho_sets,c_opt,kmeans_labels =  op_calc.optimal_partition(phi2,inv_measure,P,return_rho=True)
    
    labels_tree=np.zeros((n_final_states,len(kmeans_labels)))
    labels_tree[0,:] = kmeans_labels
    k=1
    measures_iter = []
    for k in range(1,n_final_states):
        print(k)
        eigfunctions_states=[]
        indices_states=[]
        im_states=[]
        P_states=[]
        for state in np.unique(kmeans_labels):
            cluster_traj = ma.zeros(final_labels.shape,dtype=int)
            cluster_traj[~final_labels.mask] = np.array(kmeans_labels)[final_labels[~final_labels.mask]]
            cluster_traj[final_labels.mask] = ma.masked
            labels_here = ma.zeros(final_labels.shape,dtype=int)
            sel = cluster_traj==state
            labels_here[sel] = final_labels[sel]
            labels_here[~sel] = ma.masked

            lcs,P = op_calc.transition_matrix(labels_here,delay,return_connected=True)
            R = op_calc.get_reversible_transition_matrix(P)
            im = op_calc.stationary_distribution(P)
            eigvals,eigvecs = op_calc.sorted_spectrum(R,k=2)
            indices = np.zeros(len(np.unique(final_labels.compressed())),dtype=bool)
            indices[lcs] = True
            print(lcs.shape,np.unique(labels_here.compressed()).shape)

            eigfunctions_states.append((eigvecs.real/np.linalg.norm(eigvecs.real,axis=0))[:,1])
            indices_states.append(indices)
            P_states.append(P)
            im_states.append(im)

        measures = [(inv_measure[kmeans_labels==state]).sum() for state in np.unique(kmeans_labels)]
        measures_iter.append(measures)

        state_to_split = np.argmax(measures)
        print(state_to_split,measures)
        kmeans_labels = subdivide_state_optimal(state_to_split,eigfunctions_states[state_to_split],
                                               kmeans_labels,im_states[state_to_split],P_states[state_to_split],
                                               indices_states[state_to_split])
        labels_tree[k,:] = np.copy(kmeans_labels)
        k+=1
    sel = ~np.isnan(kmeans_labels)
    measures = [(inv_measure[sel][kmeans_labels[sel]==state]).sum() for state in np.unique(kmeans_labels[sel])]
    measures_iter.append(measures)
    return labels_tree,measures_iter 


def main():
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
    
    frameRate=16.
    dt=1/frameRate
    delay = int(.75*frameRate)
    print(delay)
    # delay=13
    lcs,P = op_calc.transition_matrix(labels_traj,delay,return_connected=True)
    final_labels = op_calc.get_connected_labels(labels_traj,lcs)
    n_modes=10
    inv_measure = op_calc.stationary_distribution(P)
    R = op_calc.get_reversible_transition_matrix(P)
    eigvals,eigvecs = op_calc.sorted_spectrum(R,k=n_modes)
    sorted_indices = np.argsort(eigvals.real)[::-1]
    eigvals = eigvals[sorted_indices][1:].real
    eigvals[np.abs(eigvals-1)<1e-12] = np.nan
    eigvals[eigvals<1e-12] = np.nan
    eigfunctions = eigvecs.real/np.linalg.norm(eigvecs.real,axis=0)
    eigfunctions_traj = ma.array(eigfunctions)[final_labels,:]
    eigfunctions_traj[final_labels.mask] = ma.masked

    phi2 = eigfunctions[:,1]
    
    n_final_states=6
    labels_tree,measures = recursive_partitioning_optimal(final_labels,delay,phi2,inv_measure,P,n_final_states)

    f = h5py.File('/flash/StephensU/antonio/Foraging/subdivide_states/labels_tree.h5','w')
    d_ = f.create_dataset('delay',(1,))
    d_[...] = delay
    ef_ = f.create_dataset('eigfunctions',eigfunctions.shape)
    ef_[...] = eigfunctions
    fl_ = f.create_dataset('final_labels',final_labels.shape)
    fl_[...] = final_labels
    final_labels_mask = np.zeros(final_labels.shape)
    final_labels_mask[final_labels.mask] = 1
    flm_ = f.create_dataset('final_labels_mask',final_labels_mask.shape)
    flm_[...] = final_labels_mask
    lt_ = f.create_dataset('labels_tree',labels_tree.shape)
    lt_[...] = labels_tree
    m_ = f.create_group('measures')
    for ks in range(len(measures)):
        ml_ = m_.create_dataset(str(ks),np.array(measures[ks]).shape)
        ml_[...] = np.array(measures[ks])
    f.close()
        
    
    
    
if __name__ == "__main__":
    main()