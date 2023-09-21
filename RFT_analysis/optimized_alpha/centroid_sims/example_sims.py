#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import argparse
import sys
sys.path.append('/home/a/antonio-costa/modes2centroid_python/')
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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm',help='worm_idx',default=0,type=int)
    args=parser.parse_args()
    kw = int(args.worm)
    
    frameRate = 16.
    dt = 1/frameRate
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/animations/tseries_sims_{}.h5'.format(kw),'r')
    ts_smooth_sims = np.array(f['ts_smooth_sims'])
    f.close()
    
    alpha=30.
    
    eigenworms_matrix = np.loadtxt('/bucket/StephensU/antonio/ForagingN2_data/EigenWorms.csv', delimiter=',').astype(np.float32)
    thetas_sim = ma.array([ts.dot(eigenworms_matrix[:,:5].T) for ts in ts_smooth_sims])
    
    mat = loadmat('/bucket/StephensU/antonio/ForagingN2_data/shapes.mat')
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
    
    n_sims = 10

    print(n_sims,flush=True)
    
    X_sims = np.zeros((n_sims,len(thetas_sim[0]),2))
    for ks,ksim in enumerate(np.random.randint(0,len(thetas_sim),n_sims)):
        angleArray = thetas_sim[ksim].T
        Xrecon =  rec_traj_from_sim(angleArray,L,dt,alpha=alpha)
        X_sims[ks] = Xrecon
        print(kw,ksim,flush=True)
    
    print('Saving results',flush=True)
    
    f = h5py.File('/flash/StephensU/antonio/Foraging/centroid_sims_median_alpha/example_sims_w_{}.h5'.format(kw),'w')
    a_ = f.create_dataset('alpha',(1,))
    a_[...] = alpha
    X_sims_ = f.create_dataset('X_sims',X_sims.shape)
    X_sims_[...] = X_sims
    f.close()
    
if __name__ == "__main__":
    main(sys.argv)  
