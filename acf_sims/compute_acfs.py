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


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-worm','--Worm',help='worm_idx',default=0,type=int)
    args=parser.parse_args()

    worm_idx = args.Worm

    print('Loading data',flush=True)
    mat=h5py.File('path_to_data/PNAS2011-DataStitched.mat','r')
    refs=list(mat['#refs#'].keys())[1:]
    tseries_data=ma.masked_invalid(np.array(mat['#refs#'][refs[worm_idx]]).T)[:,:5]
    mat.close()
    frameRate=16.
    dt=1/frameRate
    len_w,dim = tseries_data.shape


    output_path = 'path_to_data/animations/'
    f = h5py.File(output_path+'tseries_sims_{}.h5'.format(worm_idx),'r')
    sims_ts_shuffled = ma.masked_invalid(np.array(f['ts_shuffled_sims']))
    sims_ts = ma.masked_invalid(np.array(f['ts_sims']))
    f.close()
    sims_ts[sims_ts==0]=ma.masked
    sims_ts_shuffled[sims_ts_shuffled==0]=ma.masked

#     print(sims_ts[0][:20,0],sims_ts_shuffled[0][:20,0],flush=True)

    lags_acf = np.array(np.arange(0,10*frameRate,1),dtype=int)
    acfs_data = np.vstack([stats.acf(tseries_data[:,kd],lags_acf) for kd in range(dim)]).T
    acfs_sims = np.zeros((sims_ts.shape[0],len(lags_acf),sims_ts.shape[2]))
    acfs_shuffled_sims = np.zeros((sims_ts_shuffled.shape[0],len(lags_acf),sims_ts_shuffled.shape[2]))
    n_sims=sims_ts.shape[0]
    for sim_idx in range(n_sims):
        for kd in range(sims_ts.shape[2]):
            acfs_sims[sim_idx,:,kd] = stats.acf(sims_ts[sim_idx,:,kd],lags_acf)
            acfs_shuffled_sims[sim_idx,:,kd] = stats.acf(sims_ts_shuffled[sim_idx,:,kd],lags_acf)
        if sim_idx%50==0:
            print(sim_idx,flush=True)

    print(acfs_data.shape,acfs_sims.shape,acfs_shuffled_sims.shape,flush=True)

    print('Saving results',flush=True)
    output_path = 'path_to_data/animations/'
    f = h5py.File(output_path+'acfs_{}.h5'.format(worm_idx),'w')
    acfs_sim_ = f.create_dataset('acfs_sims',acfs_sims.shape)
    acfs_sim_[...] = acfs_sims
    acfs_shuffled_sim_ = f.create_dataset('acfs_shuffled_sims',acfs_shuffled_sims.shape)
    acfs_shuffled_sim_[...] = acfs_shuffled_sims
    acfs_data_ = f.create_dataset('acfs_data',acfs_data.shape)
    acfs_data_[...] = acfs_data
    lags_ = f.create_dataset('lags',lags_acf.shape)
    lags_[...] = lags_acf
    f.close()

if __name__ == "__main__":
    main(sys.argv)
