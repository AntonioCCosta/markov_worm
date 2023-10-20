#data format library
import h5py
#numpy
import sys
import os
import argparse
import scipy.io
import numpy as np
import numpy.ma as ma
import scipy.io
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import rft_reconstruct_traj as rt


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

from scipy.signal import welch

def compute_ps(X,frameRate,nfreqs=1000):
    stride=1
    Xprime = ma.zeros(X.shape)
    Xprime[int(np.floor(stride/2)):-int(np.ceil(stride/2))] = X[stride:]-X[:-stride]
    Xprime[Xprime==0]=ma.masked
    phi = np.arctan2(Xprime[:,1],Xprime[:,0])
    f, psd = welch(np.cos(phi),
                   fs=frameRate,  # sample rate
                   window='hanning',   # apply a Hanning window before taking the DFT
                   nperseg=10*60*frameRate,        # compute periodograms of 256-long segments of x
                   detrend='constant')
    sel = f>1e-3
    freq = f[sel]
    phi_psd = psd[sel]
    log_indices = np.unique(np.array(np.logspace(0,np.log10(len(freq)),nfreqs),dtype=int))
    return f[log_indices],psd[log_indices]

def rec_traj_from_sim(angleArray,L,dt,alpha=35.):
    theta = -angleArray
    ds = L/(theta.shape[0])
    skel = rt.get_skels(theta,L)
    X = skel[:,:,0]
    Y = skel[:,:,1]

    XCM, YCM, UX, UY, UXCM, UYCM, TX, TY, NX, NY, I, OMEG = rt.get_RBM(skel,L,ds,dt)
    DX, DY, ODX, ODY, VX, VY, Xtil, Ytil, THETA = rt.subtractRBM(X, Y, XCM, YCM, UX, UY, UXCM, UYCM, OMEG, dt)
    TX,TY = rt.lab2body(TX, TY, THETA)
    VX,VY = rt.lab2body(VX, VY, THETA)

    RBM = rt.posture2RBM(TX,TY,Xtil,Ytil,VX,VY,L,I,ds,alpha)
    XCM_recon,YCM_recon,THETA_recon = rt.integrateRBM(RBM,dt,THETA)
    Xrecon = np.vstack([XCM_recon,YCM_recon]).T
    return Xrecon



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-state','--State',help='state_idx',default=0,type=int)
    args=parser.parse_args()

    state_idx = args.State

    eigenworms_matrix = np.loadtxt('path_to_data/EigenWorms.csv', delimiter=',').astype(np.float32)

    frameRate=16.
    dt=1/frameRate

    n_sims=1000

    L=1
    alpha=30.

    lags_msd = np.unique(np.array(np.logspace(np.log10(frameRate),np.log10(int(500*frameRate)),250),dtype=int))
    mu_sims = np.zeros((n_sims,len(lags_msd)))
    psd_sims = []
    for ks in range(n_sims):
        f = h5py.File('path_to_data/animations/tseries_sims_state_{}.h5'.format(state_idx),'r')
        ts = np.array(f['ts_smooth_sims'])[ks]
        thetas = ts.dot(eigenworms_matrix[:,:5].T)
        f.close()
        angleArray = thetas.T
        Xrecon =  rec_traj_from_sim(angleArray,L,dt,alpha=alpha)
        mu_,Unc_ = MSD_unc(ma.array(Xrecon),lags=lags_msd)
        mu_sims[ks] = mu_
        f,psd = compute_ps(Xrecon,frameRate)
        psd_sims.append(psd)
        if ks%5==0:
            print(ks,flush=True)

    psd_sims = np.vstack(psd_sims)
    freq = f

    print(mu_sims.shape,flush=True)

    print('Saving msd results',flush=True)
    output_path = '/flash/StephensU/antonio/Foraging/msds_psds_states/'
    f = h5py.File(output_path+'state_{}.h5'.format(state_idx),'w')
    mu_sim_ = f.create_dataset('mu_sims',mu_sims.shape)
    mu_sim_[...] = mu_sims
    lags_ = f.create_dataset('lags',lags_msd.shape)
    lags_[...] = lags_msd
    psd_sim_ = f.create_dataset('psd_sims',psd_sims.shape)
    psd_sim_[...] = psd_sims
    f_ = f.create_dataset('freqs',freq.shape)
    f_[...] = freq
    f.close()

if __name__ == "__main__":
    main(sys.argv)
