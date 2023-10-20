#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import argparse
import sys
#replace 'path_to_utils' and 'path_to_data'
sys.path.append('path_to_utils')
import rft_reconstruct_traj as rt
from scipy.io import loadmat

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


def MSD(x, lags=500):
    if type(lags) is int:
        lags = np.arange(lags)
    mu = ma.zeros((len(lags),))
    for i, lag in enumerate(lags):
        if lag==0:
            mu[i] = 0
        elif lag >= x.shape[0]:
            mu[i] = ma.masked
        else:
            x0 = x[lag:,:].copy()
            x1 = x[:-lag,:].copy()
            displacements = ma.sum((x0 - x1)**2,axis=1)
            mu[i] = displacements.mean()
    return mu

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm',help='worm_idx',default=0,type=int)
    args=parser.parse_args()
    kw = int(args.worm)

    frameRate = 16.
    dt = 1/frameRate
    f = h5py.File('path_to_data/animations/tseries_sims_{}.h5'.format(kw),'r')
    ts_smooth_sims = np.array(f['ts_smooth_sims'])
    f.close()

    alpha=30.

    eigenworms_matrix = np.loadtxt('path_to_data/EigenWorms.csv', delimiter=',').astype(np.float32)
    thetas_sim = ma.array([ts.dot(eigenworms_matrix[:,:5].T) for ts in ts_smooth_sims])

    mat = loadmat('path_to_data/shapes.mat')
    theta_ensemble = np.array(mat['theta_ensemble'],dtype=float)
    stepper_to_mm = 788
    wormCM = ma.array(mat['wormCm'][kw,::2,:],dtype=float)/stepper_to_mm
    wormCM[wormCM==0] = ma.masked
    pix_to_mm = 405
    wormLength = ma.array(mat['wormLength'][kw],dtype=float)/pix_to_mm
    wormLength[wormLength==0]=ma.masked
    L = np.median(wormLength.compressed())

    print(alpha,L,wormCM.shape,flush=True)
    
    X_data_c = ma.masked_invalid(wormCM)
    X_data_c[X_data_c==0] = ma.masked

    n_sims = len(ts_smooth_sims)

    print(n_sims,flush=True)

    lags = np.unique(np.array(np.logspace(np.log10(frameRate),np.log10(int(150*frameRate)),300),dtype=int))
    mu_data = MSD(X_data_c,lags)
    mu_sims = np.zeros((n_sims,len(lags)))
    for ksim in range(n_sims):
        angleArray = thetas_sim[ksim].T
        Xrecon =  rec_traj_from_sim(angleArray,L,dt,alpha=alpha)
        mu_sim = MSD(Xrecon,lags)
        mu_sims[ksim] = mu_sim
        print(ksim,flush=True)

    print('Saving results',flush=True)

    f = h5py.File('/flash/StephensU/antonio/Foraging/centroid_sims_median_alpha/msds_sims_w_{}.h5'.format(kw),'w')
    a_ = f.create_dataset('alpha',(1,))
    a_[...] = alpha
    msd_sims_ = f.create_dataset('mu_sims',mu_sims.shape)
    msd_sims_[...] = mu_sims
    msd_data_ = f.create_dataset('mu_data',mu_data.shape)
    msd_data_[...] = mu_data
    lags_ = f.create_dataset('lags',lags.shape)
    lags_[...] = lags
    f.close()

if __name__ == "__main__":
    main(sys.argv)
