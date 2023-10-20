import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt

#this script is based on a matlab implementation from Keavey et al. (2017) Physical Biology
#https://github.com/aexbrown/Crawl_Model

def get_skels(theta,L):
    numAngles = theta.shape[0]
    numFrames = theta.shape[1]
    skel = np.zeros((numFrames,numAngles+1,2))
    for kt in range(numFrames):
        skelX = np.hstack([0,np.cumsum(np.cos(theta[:,kt]))])*L/numAngles
        skelY = np.hstack([0,np.cumsum(np.sin(theta[:,kt]))])*L/numAngles
        skel[kt] = np.vstack([skelX,skelY]).T
    return skel

def TvecFromSkel(X,Y,ds):
    Nf,Ns = X.shape
    TX = np.zeros((Ns, Nf))
    TY = np.zeros((Ns, Nf))
    for n in range(Ns):
        if n == 0:
            TX[n,:] = X[:,n+1] - X[:,n]
            TY[n,:] = Y[:,n+1] - Y[:,n]
        elif n == Ns-1:
            TX[n,:] = X[:,n] - X[:,n-1]
            TY[n,:] = Y[:,n] - Y[:,n-1]
        else:
            TX[n,:] = 0.5*(X[:,n+1] - X[:,n-1])
            TY[n,:] = 0.5*(Y[:,n+1] - Y[:,n-1])
    TX /= ds
    TY /= ds
    Tmag = np.sqrt(TX**2 + TY**2)
    TX /= Tmag
    TY /= Tmag
    return -TX.T,-TY.T


def get_RBM(skel, L, ds, dt):
    X = skel[:,:,0]
    Y = skel[:,:,1]
    skel_CM = np.trapz(skel,axis=1)*ds/L
    XCM = skel_CM[:,0]
    YCM = skel_CM[:,1]
    Uskel = (np.diff(skel,axis=0)/dt)
    UX = Uskel[:,:,0]
    UY = Uskel[:,:,1]

    Uskel_CM = ds*np.trapz(Uskel,axis=1)/L
    UXCM = Uskel_CM[:,0]
    UYCM = Uskel_CM[:,1]

    TX, TY = TvecFromSkel(X,Y,ds)
    NX = -TY
    NY = TX

    Iint = np.zeros(X.shape) #moment of inertia
    Omegint = np.zeros(X.shape) #angular velocity
    for i in range(X.shape[0]):
        Iint[i,:] = (X[i,:]-XCM[i])**2+(Y[i,:]-YCM[i])**2
        if i>0:
            it1 = i - 1
            Omegint[it1,:] = (X[i,:] - XCM[i])*(UY[it1,:]-UYCM[it1]) - (Y[i,:] - YCM[i])*(UX[it1,:]-UXCM[it1])
    I = ds*np.trapz(Iint,axis=1)
    OMEG = ds*np.trapz(Omegint,axis=1)/I
    OMEG = OMEG[:-1]

    return XCM, YCM, UX, UY, UXCM, UYCM, TX, TY, NX, NY, I, OMEG


def subtractRBM(X, Y, XCM, YCM, UX, UY, UXCM, UYCM, OMEG, dt):
    DX = np.zeros(X.shape)
    DY = np.zeros(X.shape)
    ODX = np.zeros(((X.shape[0]-1),X.shape[1]))
    ODY = np.zeros(((X.shape[0]-1),X.shape[1]))

    Xtil = np.zeros(X.shape)
    Ytil = np.zeros(X.shape)
    THETA = np.zeros((X.shape[0],1))
    THETA[0] = 0

    for i in range(X.shape[0]):
        DX[i,:] = X[i,:] - XCM[i]
        DY[i,:] = Y[i,:] - YCM[i]
        Xtil[i,:] = DX[i,:]
        Ytil[i,:] = DY[i,:]
        if i>0:
            # cross product of dX with U (for angular velocity)
            it1 = i-1
            ODX[it1,:] = OMEG[it1]*DX[i,:]
            ODY[it1,:] = OMEG[it1]*DY[i,:]
            THETA[i] = THETA[i-1] + OMEG[it1]*dt
            Xtil[i,:]= np.cos(THETA[i])*DX[i,:] + np.sin(THETA[i])*DY[i,:]
            Ytil[i,:] = np.cos(THETA[i])*DY[i,:] - np.sin(THETA[i])*DX[i,:]

    VX = UX - np.tile(UXCM.reshape(-1,1),(1,X.shape[1])) + ODY
    VY = UY - np.tile(UYCM.reshape(-1,1),(1,X.shape[1])) - ODX

    return DX, DY, ODX, ODY, VX, VY, Xtil, Ytil, THETA

def lab2body(X,Y,THETA):
    Xp = np.zeros(X.shape)
    Yp = np.zeros(Y.shape)
    for i in range(X.shape[0]):
        Xp[i,:] = np.cos(THETA[i])*X[i,:] + np.sin(THETA[i])*Y[i,:]
        Yp[i,:] = np.cos(THETA[i])*Y[i,:] - np.sin(THETA[i])*X[i,:]
    return Xp,Yp


def posture2RBM(TX, TY, Xtil, Ytil, VX, VY, L, I, ds, alpha):
    RBM = np.zeros((TX.shape[0]-1,3))

    #get tangential component of velocity at each skeleton point without
    #centre of mass
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

def addRBMRotMat(Xtil, Ytil, XCM, YCM, THETA, XCMi, YCMi, THETAi):
    X = np.zeros(Xtil.shape)
    Y = np.zeros(Ytil.shape)
    for ii in range(Xtil.shape[0]):
        xt = XCM[ii] - XCM[0] + XCMi[0]
        yt = YCM[ii] - YCM[0] + YCMi[0]
        tht = THETA[ii] - THETA[0] + THETAi[0]
        XNR = Xtil[ii,:]
        YNR = Ytil[ii,:]
        Xtilt = np.cos(tht)*XNR - np.sin(tht)*YNR
        Ytilt = np.cos(tht)*YNR + np.sin(tht)*XNR
        X[ii, :] = Xtilt + xt
        Y[ii, :] = Ytilt + yt
    return X,Y
