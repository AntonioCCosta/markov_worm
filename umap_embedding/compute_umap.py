#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
from sklearn.cluster import KMeans
import sys
import os
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import operator_calculations as op_calc
import worm_dynamics as worm_dyn
import stats
import clustering_methods as cl
import argparse


import umap

def draw_umap(data,n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    return u


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='idx',default=0,type=int)
    args=parser.parse_args()

    n_neighbors,min_dist = np.array(np.loadtxt('iteration_indices.txt')[args.Idx])


    print(n_neighbors,min_dist,flush=True)

    f = h5py.File('path_to_data/phspace_K_10_m_7.h5','r')
    traj_matrix = ma.masked_invalid(f['traj_matrix'])
    traj_matrix[traj_matrix==0]=ma.masked
    f.close()

    n_clusters=1000
    f = h5py.File('path_to_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_traj = np.array(f['mask_traj'],dtype=bool)
    labels_phspace = ma.array(f['labels_phspace'],dtype=int)
    mask_phspace = np.array(f['mask_phspace'],dtype=bool)
    centers_phspace = np.array(f['centers_phspace'])
    centers_traj = ma.masked_invalid(np.array(f['centers_traj']))
    f.close()

    traj = ma.vstack([traj_matrix,centers_traj])

    sel = ~np.any(traj.mask,axis=1)
    u = np.zeros((traj.shape[0],2))
    data = traj[sel]
    u[sel] = draw_umap(data,n_neighbors=n_neighbors,min_dist=min_dist,metric='chebyshev')

    print('Saving results',flush=True)
    f = h5py.File('path_to_data/umap_embeddings/umap_n_{}_d_{:.2f}.h5'.format(n_neighbors,min_dist),'w')
    um_ = f.create_dataset('umap',u.shape)
    um_[...] = u
    f.close()

if __name__ == "__main__":
    main(sys.argv)
