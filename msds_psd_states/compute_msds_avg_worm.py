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
import scipy.io
from scipy.signal import savgol_filter, general_gaussian


def MSD_unc(x, lags=500, exclude=None):
    if exclude is None:
        exclude = np.zeros(x.shape[0])
    exclude = np.cumsum(exclude.astype(int))

    if type(lags) is int:
        lags = np.arange(lags)
    mu = ma.zeros((len(lags),))
    Unc = ma.zeros((len(lags),))
    Unc.mask=True
    for i, lag in enumerate(lags):
        if lag==0:
            mu[i] = 0
            Unc[0]=len(x[:,0].compressed())
        elif lag >= x.shape[0]:
            mu[i] = ma.masked
        else:
            x0 = x[lag:,:].copy()
            x1 = x[:-lag,:].copy()
            reject = (exclude[lag:]-exclude[:-lag])>0
            x0[reject,:] = ma.masked
            x1[reject,:] = ma.masked
            displacements = ma.sum((x0 - x1)**2,axis=1)
            mu[i] = displacements.mean()
            Unc[i]=len(displacements.compressed())
    return mu,Unc 



def main():    
    frameRate=16.
    dt=1/frameRate
    
    print('Loading sims')
    f = h5py.File('/flash/StephensU/antonio/Foraging/centroid_simulations/X_scaled_avg_worm.h5','r')
    X_sims = np.array(f['X_scaled'])
    f.close()

    print(X_sims.shape,flush=True)

    print('Computing MSDs',flush=True)
            
    n_sims = X_sims.shape[0]    
    lags_msd = np.arange(0,int(15*60*frameRate),int(.5*np.ceil(frameRate)))
    mu_sims = np.zeros((n_sims,len(lags_msd)))
    for ks in range(n_sims):
        mu_,Unc_ = MSD_unc(ma.array(X_sims[ks]),lags=lags_msd)
        mu_sims[ks] = mu_
        if ks%5==0:
            print(ks,flush=True)
            
    print(mu_sims.shape,flush=True)
    
    print('Saving msd results',flush=True)
    output_path = '/flash/StephensU/antonio/Foraging/msds/'
    f = h5py.File(output_path+'msds_avg_worm.h5','w')
    mu_sim_ = f.create_dataset('mu_sims',mu_sims.shape)
    mu_sim_[...] = mu_sims
    lags_ = f.create_dataset('lags',lags_msd.shape)
    lags_[...] = lags_msd
    f.close()

if __name__ == "__main__":
    main()
