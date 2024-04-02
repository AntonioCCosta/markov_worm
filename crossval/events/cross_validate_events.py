import numpy as np
import numpy.ma as ma
import sys
import os
sys.path.append('/home/a/antonio-costa/TransferOperators/bridging_scales_manuscript/utils')
import operator_calculations as op_calc
import worm_dynamics as worm_dyn
import stats
import clustering_methods as cl
import argparse
import h5py

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

def generate_random_tseries(traj_sim_example,traj_matrix,labels,K_star,delay,dim):
    kidx=0
    idx0 = traj_sim_example[kidx]
    sel0 = labels==idx0
    ts0 = traj_matrix[sel0].reshape(sel0.sum(),K_star,dim)[:,::-1,:][0]
    ts_all = ma.zeros((int(len(traj_sim_example)*delay),dim))
    ts_all[:K_star] = ts0
    k=len(ts0)
    for kidx in range(len(traj_sim_example)-1):
        idx1 = traj_sim_example[kidx+1]
        sel1 = labels==idx1
        ts_samples1 = traj_matrix[sel1].reshape(sel1.sum(),K_star,dim)[:,::-1,:]
        idx = np.random.randint(0,len(ts_samples1))
        ts0 = ts_samples1[idx]
        ts_all[k+1:k+1+len(ts0)] = ts0
        k= k+1+len(ts0)
        kidx+=1
    ts_all[ts_all==0]=ma.masked
    return ts_all

from joblib import Parallel, delayed

def sim_ts_parallel(sims,traj_matrix,labels,K_star,delay,dim,k_idx):
    traj_sim_example = sims[k_idx]
    return generate_random_tseries(traj_sim_example,traj_matrix,labels,K_star,delay,dim)

from scipy import interpolate
def interpolate_gaps(ts_sim):
    ts_interp = np.zeros(ts_sim.shape)
    for kd in range(ts_sim.shape[1]):
        x = np.arange(len(ts_sim))
        y = ts_sim[:,kd]
        sel = ~y.mask
        f = interpolate.interp1d(x[sel], y[sel],kind='cubic',fill_value='extrapolate')
        ts_interp[:,kd] = f(x)
    return ts_interp

from scipy.signal import savgol_filter
def smooth_ts(tseries):
    dim = tseries.shape[1]
    ts_smooth = np.vstack([savgol_filter(tseries[:,kd], 11, 3)  for kd in range(dim)]).T
    return ts_smooth
    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx','--Idx',help='index',default=0,type=int)
    parser.add_argument('-train_ratio','--Train_ratio',help='train ratio',default=0.8,type=float)

    args=parser.parse_args()
    kw,idx = np.array(np.loadtxt('indices.txt')[args.Idx],dtype=int)
    
    print(kw,idx,flush=True)
    
    mat=h5py.File('/bucket/StephensU/antonio/ForagingN2_data/PNAS2011-DataStitched.mat','r')

    refs=list(mat['#refs#'].keys())[1:]
    tseries_w=[ma.masked_invalid(np.array(mat['#refs#'][ref]).T)[:,:5] for ref in refs]
    mat.close()
    frameRate=16.
    dt=1/frameRate
    len_w,dim = tseries_w[0].shape
    n_worms = len(tseries_w)
    
    n_clusters=1000
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/symbol_sequences/labels_{}_clusters.h5'.format(n_clusters),'r')
    labels_traj = ma.array(f['labels_traj'],dtype=int)
    mask_traj = np.array(f['mask_traj'],dtype=bool)
    labels_phspace = ma.array(f['labels_phspace'],dtype=int)
    mask_phspace = np.array(f['mask_phspace'],dtype=bool)
    centers_phspace = np.array(f['centers_phspace'])
    centers_traj = np.array(f['centers_traj'])
    f.close()
    labels_traj[mask_traj] = ma.masked
    labels_phspace[mask_phspace] = ma.masked
    
    labels_w = labels_traj.reshape((n_worms,len_w))
    
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/phspace_K_10_m_7.h5','r')
    traj_matrix = ma.masked_invalid(f['traj_matrix'])
    phspace = ma.array(f['phspace'])
    components = np.array(f['modes'])
    traj_matrix[traj_matrix==0]=ma.masked
    f.close()
    
    traj_matrix_w = traj_matrix.reshape((n_worms,len_w,traj_matrix.shape[1]))
    
    f = h5py.File('/bucket/StephensU/antonio/ForagingN2_data/labels_tree.h5','r')
    delay = int(np.array(f['delay'])[0])
    eigfunctions = np.array(f['eigfunctions'])
    final_labels = ma.masked_invalid(np.array(f['final_labels'],dtype=int))
    final_labels_mask = np.array(f['final_labels_mask'])
    sel = final_labels_mask==1
    final_labels[sel] = ma.masked
    labels_tree = np.array(f['labels_tree'])
    f.close()
    
    eigenworms_matrix = np.loadtxt('/bucket/StephensU/antonio/ForagingN2_data/EigenWorms.csv', delimiter=',').astype(np.float32)

    
    kmeans_labels = labels_tree[0,:]
    
   
    wsize = 3360
    t0s_range =  np.arange(0,len_w,wsize)
    labels_seg = ma.zeros((len(t0s_range),wsize),dtype=int)
    tseries_seg = ma.zeros((len(t0s_range),wsize,dim))
    traj_matrix_seg = ma.zeros((len(t0s_range),wsize,traj_matrix.shape[1]))
    for ks,t0 in enumerate(t0s_range):
        tf = t0+wsize
        labels_here = labels_w[kw][t0:tf]
        labels_here[-1] = ma.masked
        tseries_here = tseries_w[kw][t0:tf]
        tseries_here[-1] = ma.masked
        tm_here = traj_matrix_w[kw][t0:tf]
        tm_here[-1] = ma.masked
        labels_seg[ks] = labels_here
        tseries_seg[ks] = tseries_here
        traj_matrix_seg[ks] = tm_here

    train_ratio=args.Train_ratio
    
    print('train ratio ',train_ratio,flush=True)
    
    ntrain = int(len(t0s_range)*train_ratio)
    sel = np.zeros(len(labels_seg),dtype='bool')
    sel[np.random.choice(np.arange(0,len(labels_seg)),ntrain,replace=False)]=True
    print(sel.sum(),(~sel).sum())
    labels_train = ma.hstack(labels_seg[sel])
    labels_test = ma.hstack(labels_seg[~sel])
    tseries_train = ma.vstack(tseries_seg[sel])
    tseries_test = ma.vstack(tseries_seg[~sel])
    traj_matrix_train = ma.vstack(traj_matrix_seg[sel])
    traj_matrix_test = ma.vstack(traj_matrix_seg[~sel])

    lcs,P = op_calc.transition_matrix(labels_train,delay,return_connected=True)


    len_sim = int(len(tseries_test)/delay)

    n_sims=100

    states0_sample = np.random.choice(P.shape[0],5*n_sims,replace=False)
    states0 = states0_sample[[lcs[state0] in labels_test for state0 in states0_sample]][:n_sims]
    
    sims = Parallel(n_jobs=50)(delayed(simulate_parallel)(P,state0,len_sim,lcs)
                               for state0 in states0)

    K_star=11
    ts_sims = Parallel(n_jobs=50)(delayed(sim_ts_parallel)(sims,traj_matrix_train,labels_train,K_star,delay,dim,k_idx) for k_idx in range(n_sims))
    ts_sims = ma.masked_invalid(ts_sims)
    ts_sims[ts_sims==0]=ma.masked

    
    ts_smooth_sims=[]
    for ks in range(n_sims):
        ts_sim = ts_sims[ks]
        ts_interp = interpolate_gaps(ts_sim)
        ts_smooth = smooth_ts(ts_interp)
        ts_smooth_sims.append(ts_smooth)
        
    def rev_rate(omegas,thetas,omega_max = -.2,theta_min = -3.e-4,theta_max = 3.e-4,minw = 8,bootstrap=False):
        sel1 = omegas<omega_max
        sel2 = np.logical_and(theta_min<thetas, thetas<theta_max)
        mask = np.logical_and(sel1,sel2)
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        if len(segments)>0:
            sel = np.hstack(np.diff(segments))>minw
            if bootstrap:
                n_boot=1000
                sum_samples = np.array([sel[np.random.randint(0,len(sel),len(sel))].sum() for k in range(n_boot)])
                cil = np.percentile(sum_samples,2.5)/(len(omegas)*dt/60)
                ciu = np.percentile(sum_samples,97.5)/(len(omegas)*dt/60)
                return sel.sum()/(len(omegas)*dt/60),cil,ciu
            return sel.sum()/(len(omegas)*dt/60) #in per minute
        else:
            if bootstrap:
                return 0,0,0
            return 0

    def ventral_turn_rate(thetas,theta_max = -3.5e-4,minw = 8,bootstrap=False):
        mask = thetas<theta_max
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        if len(segments)>0:
            sel = np.hstack(np.diff(segments))>minw
            if bootstrap:
                n_boot=1000
                sum_samples = np.array([sel[np.random.randint(0,len(sel),len(sel))].sum() for k in range(n_boot)])
                cil = np.percentile(sum_samples,2.5)/(len(thetas)*dt/60)
                ciu = np.percentile(sum_samples,97.5)/(len(thetas)*dt/60)
                return sel.sum()/(len(thetas)*dt/60),cil,ciu
            return sel.sum()/(len(thetas)*dt/60) #in per minute
        else:
            if bootstrap:
                return 0,0,0
            return 0


    def dorsal_turn_rate(thetas,theta_min = 3.5e-4,minw = 8 ,bootstrap=False):
        mask = thetas>theta_min
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        if len(segments)>0:
            sel = np.hstack(np.diff(segments))>minw
            if bootstrap:
                n_boot=1000
                sum_samples = np.array([sel[np.random.randint(0,len(sel),len(sel))].sum() for k in range(n_boot)])
                cil = np.percentile(sum_samples,2.5)/(len(thetas)*dt/60)
                ciu = np.percentile(sum_samples,97.5)/(len(thetas)*dt/60)
                return sel.sum()/(len(thetas)*dt/60),cil,ciu
            return sel.sum()/(len(thetas)*dt/60) #in per minute
        else:
            if bootstrap:
                return 0,0,0
            return 0
        
        
    from scipy.interpolate import CubicSpline
    def unwrapma(x):
        idx= ma.array(np.arange(0,x.shape[0]),mask=x.mask)
        idxc=idx.compressed()
        xc=x.compressed()
        dd=np.diff(xc)
        ddmod=np.mod(dd+np.pi,2*np.pi)-np.pi
        ddmod[(ddmod==-np.pi)&(dd>0)]=np.pi
        phc_correct = ddmod-dd
        phc_correct[np.abs(dd)<np.pi] = 0
        ph_correct = np.zeros(x.shape)
        ph_correct[idxc[1:]] = phc_correct
        up = x + ph_correct.cumsum()
        return up
    
    def compute_phi_omega_a3(tseries,t0,tf,frameRate=16.):
        time=np.arange(t0,tf)
        X=tseries[time]
        phi=-np.arctan2(X[:,1],X[:,0])
        cs = CubicSpline(time, phi)
        phiFilt=cs(time)
        phi_unwrap=unwrapma(phi)
        sel=~phi_unwrap.mask
        cs = CubicSpline(time[sel], phi_unwrap[sel])
        #normalize by frame rate
        phiFilt_unwrap=cs(time[sel])
        omegaFilt=cs(time[sel],1)*frameRate/(2*np.pi)
        return phiFilt,omegaFilt,X[:,2]
    
    print(ts_smooth_sims[0].shape)
    rev_rates=[]
    vt_rates=[]
    dt_rates=[]
    for sim_idx in range(n_sims):
        sample_sim_ts = ma.masked_invalid(ts_smooth_sims[sim_idx])
        phi_sim,omegas_sim,a3_sim=compute_phi_omega_a3(sample_sim_ts,0,len(sample_sim_ts))
        thetas_sim = sample_sim_ts.dot(eigenworms_matrix[:,:5].T)
        thetas_sim_sum = thetas_sim.sum(axis=1)

        rev_rates.append(rev_rate(omegas_sim,thetas_sim_sum))
        vt_rates.append(ventral_turn_rate(thetas_sim_sum))
        dt_rates.append(dorsal_turn_rate(thetas_sim_sum))
    print(omegas_sim.shape,thetas_sim.shape)

    rev_rates_sim_ci = np.array([np.mean(rev_rates),np.percentile(rev_rates,2.5),np.percentile(rev_rates,97.5)])
    vt_rates_sim_ci = np.array([np.mean(vt_rates),np.percentile(vt_rates,2.5),np.percentile(vt_rates,97.5)])
    dt_rates_sim_ci = np.array([np.mean(dt_rates),np.percentile(dt_rates,2.5),np.percentile(dt_rates,97.5)])

    
    
    ts_data = tseries_test
    print(ts_data.shape)
    segments=op_calc.segment_maskedArray(ts_data,5)
    omegas_data = ma.zeros(ts_data.shape[0])
    for t0,tf in segments:
        phi,omega,a3=worm_dyn.compute_phi_omega_a3(ts_data,t0,tf)
        omegas_data[t0:tf] = omega
    omegas_data[omegas_data==0]=ma.masked
    thetas_data= ts_data.dot(eigenworms_matrix[:,:5].T)
    thetas_data_sum = thetas_data.sum(axis=1)
    
    print(omegas_data.shape,thetas_data.shape)
    
    rev_rates_data_ci = np.array([rev_rate(omegas_data,thetas_data_sum,bootstrap=True)])
    vt_rates_data_ci = np.array([ventral_turn_rate(thetas_data_sum,bootstrap=True)])
    dt_rates_data_ci = np.array([dorsal_turn_rate(thetas_data_sum,bootstrap=True)])
    
    print('Saving results',flush=True)
    
    f = h5py.File('/flash/StephensU/antonio/Foraging/cross_validate_events/events_train_ratio_{:.1f}_{}.h5'.format(train_ratio,args.Idx),'w')
    rev_data_ = f.create_dataset('rev_rates_data',rev_rates_data_ci.shape)
    rev_data_[...] = rev_rates_data_ci
    rev_sim_ = f.create_dataset('rev_rates_sim',rev_rates_sim_ci.shape)
    rev_sim_[...] = rev_rates_sim_ci
    vt_data_ = f.create_dataset('vt_rates_data',vt_rates_data_ci.shape)
    vt_data_[...] = vt_rates_data_ci
    vt_sim_ = f.create_dataset('vt_rates_sim',vt_rates_sim_ci.shape)
    vt_sim_[...] = vt_rates_sim_ci
    dt_data_ = f.create_dataset('dt_rates_data',dt_rates_data_ci.shape)
    dt_data_[...] = dt_rates_data_ci
    dt_sim_ = f.create_dataset('dt_rates_sim',dt_rates_sim_ci.shape)
    dt_sim_[...] = dt_rates_sim_ci
    ws_ = f.create_dataset('wsize',(1,))
    ws_[...] = wsize
    f.close()
    

if __name__ == "__main__":
    main(sys.argv)