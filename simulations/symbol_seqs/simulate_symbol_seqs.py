import h5py
import numpy as np
import numpy.ma as ma
import os
import sys
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import clustering_methods as cl
import operator_calculations as op_calc
import delay_embedding as embed
import worm_dynamics as worm_dyn
import stats


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

n_clusters=1000

f = h5py.File('path_to_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters))
labels_traj = ma.array(f['labels_traj'],dtype=int)
mask_traj = np.array(f['mask_traj'],dtype=bool)
labels_phspace = ma.array(f['labels_phspace'],dtype=int)
mask_phspace = np.array(f['mask_phspace'],dtype=bool)
centers_phspace = np.array(f['centers_phspace'])
centers_traj = np.array(f['centers_traj'])
f.close()

labels_traj[mask_traj] = ma.masked
labels_phspace[mask_phspace] = ma.masked


mat=h5py.File('path_to_data/PNAS2011-DataStitched.mat','r')
refs=list(mat['#refs#'].keys())[1:]
tseries_w=[ma.masked_invalid(np.array(mat['#refs#'][ref]).T)[:,:5] for ref in refs]
mat.close()
frameRate=16.
dt=1/frameRate

worms = np.arange(len(tseries_w))
len_w = len(tseries_w[0])
ensemble_labels_w=[]
for worm in worms:
    ensemble_labels_w.append(labels_traj[len_w*worm:len_w*(worm+1)])


delay = int(.75*frameRate)
n_sims = 1000


f = h5py.File('path_to_data/sims/symbol_sequence_simulations_1000_clusters.h5','w')

metaData = f.create_group('MetaData')
dl = metaData.create_dataset('delay',(1,))
dl[...] = delay
nc = metaData.create_dataset('n_clusters',(1,))
nc[...] = n_clusters

for worm in worms:
    wg = f.create_group(str(worm))

    labels = ensemble_labels_w[worm]

    lcs,P = op_calc.transition_matrix(labels,delay,return_connected=True)

    final_labels = op_calc.get_connected_labels(labels,lcs)

    len_sim = int(len(labels)/delay)

    states0 = np.ones(n_sims,dtype=int)*final_labels.compressed()[0]

    sims = Parallel(n_jobs=50)(delayed(simulate_parallel)(P,state0,len_sim,lcs)
                               for state0 in states0)
    sims = np.array(sims)
    s_ = wg.create_dataset('sims',sims.shape)
    s_[...] = sims
    print(worm)
