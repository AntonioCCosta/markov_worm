#data format library
import h5py
#numpy
import numpy as np
import numpy.ma as ma
import argparse
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

    Xskel,Yskel = rt.addRBMRotMat(Xtil, Ytil, XCM_recon, YCM_recon, THETA_recon, XCM, YCM, THETA)

    return Xrecon,Xskel,Yskel

def get_bins(epsilon,r_max):
    xrange = np.arange(-r_max,r_max+epsilon,epsilon)
    yrange = np.arange(-r_max,r_max+epsilon,epsilon)
    centers_x = (xrange[1:]+xrange[:-1])/2
    centers_y = (yrange[1:]+yrange[:-1])/2
    n_bins = len(centers_x)
    rads = np.zeros((n_bins,n_bins))
    for kx,x in enumerate(centers_x):
        for ky,y in enumerate(centers_y):
            rads[kx,ky] = np.sqrt(x**2+y**2)
    return xrange,yrange,rads

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-ks','--kstate',help='ks',default=0,type=int)
    args=parser.parse_args()
    kstate = int(args.kstate)


    eigenworms_matrix = np.loadtxt('path_to_data/EigenWorms.csv', delimiter=',').astype(np.float32)
    f = h5py.File('path_to_data/animations/tseries_sims_state_{}.h5'.format(kstate),'r')
    ts_smooth_sims = np.array(f['ts_smooth_sims'])
    thetas_sim = ma.array([ts.dot(eigenworms_matrix[:,:5].T) for ts in ts_smooth_sims])
    f.close()

    frameRate = 16.
    dt = 1/frameRate
    alpha=30.

    maxR = 25
    dr=.05
    rmin=.5
    xrange,yrange,rads = get_bins(dr,maxR)

    wsize = int(100*frameRate)
    L=1
    stride_t=8
    stride_ds = 5
    n_samples = 1000

    dist_range = np.logspace(np.log10(rmin),np.log10(maxR),30)
    ac_samples= np.zeros((n_samples,len(dist_range)))
    kinE_samples=np.zeros(n_samples)
    for k in range(n_samples):
        try:
            t0 = np.random.randint(0,len(thetas_sim[0])-wsize)
            ksim=np.random.randint(0,1000)
            angleArray = thetas_sim[ksim][t0:t0+wsize].T
            Xrecon,Xskel,Yskel = rec_traj_from_sim(angleArray,L,dt,alpha)
            skel = np.array([Xskel,Yskel])
            vskel = np.diff(skel.T,axis=1)*dt
            W = (vskel**2).sum()
            kinE_samples[k]=W
            xy_all = np.concatenate(skel[:,::stride_t,::stride_ds].T,axis=0)
            freqs,_,_= np.histogram2d(xy_all[:,0],xy_all[:,1],bins=[xrange,yrange])
            for kd,dist in enumerate(dist_range):
                sel = rads<dist
                ac_samples[k,kd] = (freqs[sel]>0).sum()/sel.sum()
        except:
            print('Caution',flush=True)
            continue
        if k%10==0:
            print(kstate,k,flush=True)



    f = h5py.File('path_to_data/blind_search_opt_alpha/ac_samples_ks_{}.h5'.format(kstate),'w')
    ac_ = f.create_dataset('ac_samples',ac_samples.shape)
    ac_[...] = ac_samples
    kE_ = f.create_dataset('kinE_samples',kinE_samples.shape)
    kE_[...] = kinE_samples
    d_ = f.create_dataset('dist_range',dist_range.shape)
    d_[...] = dist_range
    a_ = f.create_dataset('alpha',(1,))
    a_[...] = alpha
    mR_ = f.create_dataset('maxR',(1,))
    mR_[...] = maxR
    dr_ = f.create_dataset('dr',(1,))
    dr_[...] = dr
    rm_ = f.create_dataset('rmin',(1,))
    rm_[...] = rmin
    ws_ = f.create_dataset('wsize',(1,))
    ws_[...] = wsize
    l_ = f.create_dataset('L',(1,))
    l_[...] = L
    st_ = f.create_dataset('stride_t',(1,))
    st_[...] = stride_t
    ss_ = f.create_dataset('stride_ds',(1,))
    ss_[...] = stride_ds
    f.close()

    print('Saved results',flush=True)

if __name__ == "__main__":
    main(sys.argv)
