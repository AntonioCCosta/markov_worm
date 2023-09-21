import h5py
import numpy as np
import argparse
import numpy.ma as ma
import sys
import os
sys.path.append('/home/a/antonio-costa/modes2centroid_python/')
import rft_reconstruct_traj as rt
from scipy.io import loadmat

from scipy.optimize import minimize


def dist_alpha(alpha,params):
    
    def posture2RBM(TX, TY, Xtil, Ytil, VX, VY, L, I, ds, alpha):
        RBM = np.zeros((TX.shape[0]-1,3))

        #get tangential component of velocity at each skeleton point without
        #centre of mass (the wiggles)
        TdotV = TX[1:,:]*VX + TY[1:,:]*VY

        #get cross product of relative skeleton position with tangent
        DelXcrossT = Xtil[1:,:]*TY[1:,:] - Ytil[1:,:]*TX[1:,:]

        for i in range(TX.shape[0]-1):
            #assemble right hand side
            b1 = (alpha - 1)*ds*np.trapz(TX[i+1,:]*TdotV[i,:])
            b2 = (alpha - 1)*ds*np.trapz(TY[i+1,:]*TdotV[i,:])
            b3 = (alpha - 1)*ds*np.trapz(DelXcrossT[i,:]*TdotV[i,:])

            #the matrix relating rigid body motion to the wiggles
            A11 = alpha*L + (1 - alpha)*ds*np.trapz(TX[i+1,:]**2)
            A12 = (1 - alpha)*ds*np.trapz(TX[i+1,:]*TY[i+1,:])
            A13 = (1 - alpha)*ds*np.trapz(TX[i+1,:]*DelXcrossT[i,:])

            A22 = alpha*L + (1 - alpha)*ds*np.trapz(TY[i+1,:]**2)
            A21 = A12
            A23 = (1 - alpha)*ds*np.trapz(TY[i+1,:]*DelXcrossT[i,:])

            A31 = A13
            A32 = A23
            A33 = alpha*I[i+1] + (1 - alpha)*ds*np.trapz(DelXcrossT[i,:]**2)

            #solve the linear system
            bvec = np.array([b1, b2, b3]).T
            Amat = np.array([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
            if len(Amat.shape)>2:
                Amat = Amat[:,:,0]
                bvec = bvec[0]
            RBM[i, :] = np.linalg.lstsq(Amat.T, bvec,rcond=-1)[0]
        return RBM
    
    
    def body2lab(X,Y,THETA):
        Xp = np.zeros(X.shape)
        Yp = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xp[i,:]= np.cos(THETA[i])*X[i,:] - np.sin(THETA[i])*Y[i,:]
            Yp[i,:] = np.cos(THETA[i])*Y[i,:] + np.sin(THETA[i])*X[i,:]
        return Xp,Yp
    
    def integrateRBM(RBM, dt, THETAr):
        RBM = RBM.copy()
        Nt = RBM.shape[0]+1
        XCM = np.zeros(Nt)
        YCM = np.zeros(Nt)
        THETA = np.zeros(Nt)
        for i in range(1,Nt):
            THETA[i] = THETA[i-1] + RBM[i-1,2]*dt
        THETA = THETA - THETA[0] + THETAr[0]
        #ROTATE VELOCITIES INTO LAB FRAME
        Xt,Yt = body2lab(RBM[:,0].reshape(-1,1), RBM[:,1].reshape(-1,1), THETA)
        RBM[:,0] = Xt[:,0]
        RBM[:,1] = Yt[:,0]
        for i in range(1,Nt):  
            XCM[i] = XCM[i-1] + RBM[i-1,0]*dt
            YCM[i] = YCM[i-1] + RBM[i-1,1]*dt  
        return XCM,YCM,THETA
    
    R = lambda theta: np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    
    TX,TY,Xtil,Ytil,VX,VY,L,I,ds,dt,THETA,X_data_c = params
    RBM = posture2RBM(TX,TY,Xtil,Ytil,VX,VY,L,I,ds,alpha)
    XCM_recon,YCM_recon,THETA_recon = integrateRBM(RBM,dt,THETA)
    Xrecon = np.vstack([XCM_recon,YCM_recon]).T
    rot_angle_range = np.linspace(0,2*np.pi,1000)
    dists = np.array([np.linalg.norm(X_data_c-Xrecon.dot(R(alpha)),axis=1).max() for alpha in rot_angle_range])
    Xrot = Xrecon.dot(R(rot_angle_range[np.argmin(dists)]))
    dist = np.linalg.norm(Xrot-X_data_c,axis=1).max()
    return dist

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    args=parser.parse_args()
    idx = int(args.Idx)
    
    print('Load data',flush=True)
    
    mat=h5py.File('/bucket/StephensU/antonio/ForagingN2_data/PNAS2011-DataStitched.mat','r')
    refs=list(mat['#refs#'].keys())[1:]
    tseries_w=[ma.masked_invalid(np.array(mat['#refs#'][ref]).T)[:,:5] for ref in refs]
    mat.close()
    frameRate=16.
    dt=1/frameRate
    
    n_worms = len(tseries_w)

    eigenworms_matrix = np.loadtxt('/bucket/StephensU/antonio/ForagingN2_data/EigenWorms.csv', delimiter=',').astype(np.float32)
    thetas_w = ma.array([ts.dot(eigenworms_matrix[:,:5].T) for ts in tseries_w])

    mat = loadmat('/bucket/StephensU/antonio/ForagingN2_data/shapes.mat')
    theta_ensemble = np.array(mat['theta_ensemble'],dtype=float)
    stepper_to_mm = 788
    wormCM = ma.array(mat['wormCm'][:,::2,:],dtype=float)/stepper_to_mm
    wormCM[wormCM==0] = ma.masked
    headTail_theta_w = np.array(mat['wormHeadTailTheta'],dtype=float)
    pix_to_mm = 405
    wormLength = ma.array(mat['wormLength'],dtype=float)/pix_to_mm
    wormLength[wormLength==0]=ma.masked    
    
    L_w = np.array([np.median(wormLength[kw].compressed())for kw in range(n_worms)])
    
    #flip back worms that were previously flipped
    theta_ensemble = ma.masked_invalid(theta_ensemble)
    theta_ensemble[theta_ensemble==0]= ma.masked
    mean_diff = np.array([ma.abs(thetas_w[kw]-theta_ensemble[kw][::2]).mean() for kw in range(n_worms)])
    mean_neg_diff = np.array([ma.abs(thetas_w[kw]+theta_ensemble[kw][::2]).mean() for kw in range(n_worms)])
    flipped_worms_neg = np.arange(n_worms)[np.array(mean_neg_diff)<.5]
    flipped_worms = np.arange(n_worms)[np.array(mean_diff)>1]
    for kw in flipped_worms:
        thetas_w[kw] = -thetas_w[kw]
    
    print('Grab random segs',flush=True)

    
    #grab a random segment
    wsize = int(100*frameRate)
    theta_all = ma.vstack(thetas_w)[:,0]
    n_frames = len(theta_all)
    len_w = len(thetas_w[0])
    indices = np.arange(n_frames-2*wsize)[~np.any(np.vstack([theta_all[t:t+wsize].mask for t in range(n_frames-2*wsize)]),axis=1)]
    
    print('Optimize alpha',flush=True)
    
    n_sims = 10
    random_indices = np.random.randint(0,len(indices),n_sims)
    opt_alphas = np.zeros(n_sims)
    kw_sims = np.zeros(n_sims)
    t0_sims = np.zeros(n_sims)
    for i in range(n_sims):
        random_k = np.random.randint(0,len(indices))
        kw = np.array(np.floor(indices/len_w),dtype=int)[random_k]
        random_idx = indices[random_k]-kw*len_w
        kw_sims[i] = kw
        t0_sims[i] = random_idx
        t0 = random_idx
        tf = t0+wsize
        angleArray = thetas_w[kw][t0:tf].T
        XCM_data = wormCM[kw][t0:tf,0]
        YCM_data = wormCM[kw][t0:tf,1]
        X_data_c = np.vstack([XCM_data-XCM_data[0],YCM_data-YCM_data[0]]).T

        L = L_w[kw]
        theta = -angleArray
        ds = L/(theta.shape[0])
        skel = rt.get_skels(theta,L)
        X = skel[:,:,0]
        Y = skel[:,:,1]

        XCM, YCM, UX, UY, UXCM, UYCM, TX, TY, NX, NY, I, OMEG = rt.get_RBM(skel,L,ds,dt)
        DX, DY, ODX, ODY, VX, VY, Xtil, Ytil, THETA = rt.subtractRBM(X, Y, XCM, YCM, UX, UY, UXCM, UYCM, OMEG, dt)
        TX,TY = rt.lab2body(TX, TY, THETA)
        VX,VY = rt.lab2body(VX, VY, THETA)

        params = [TX,TY,Xtil,Ytil,VX,VY,L,I,ds,dt,THETA,X_data_c]
        try:
            alpha0 = 80
            res = minimize(dist_alpha, alpha0,method='Nelder-Mead',bounds=(5,1000),options={'initial_simplex':np.array([[60,200]]).T}, args=(params))
            opt_alpha = res.x[0]
            opt_alphas[i] = opt_alpha
            print(i,kw,random_idx,opt_alpha,flush=True)
        except:
            print('optimization failed for kw={}, t0={}'.format(kw,random_idx))
        
    print('Save results',flush=True)

    f = h5py.File('/flash/StephensU/antonio/Foraging/optimize_alpha/opt_alphas_{}.h5'.format(idx),'w')
    ar = f.create_dataset('opt_alphas',opt_alphas.shape)
    ar[...] = opt_alphas
    w_ = f.create_dataset('kw_sims',kw_sims.shape)
    w_[...] = kw_sims
    rt_ = f.create_dataset('t0_sims',t0_sims.shape)
    rt_[...] = t0_sims
    f.close()
    
    
if __name__ == "__main__":
    main(sys.argv)
