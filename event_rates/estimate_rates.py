import numpy as np
import numpy.ma as ma
import sys
import os
sys.path.append('/home/a/antonio-costa/operator_worm_manuscript/')
import operator_calculations as op_calc
import worm_dynamics as worm_dyn
import stats
import clustering_methods as cl
import argparse
import h5py


    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-kw','--worm_idx',help='index',default=0,type=int)
    args=parser.parse_args()
    kw = int(args.worm_idx)
    
    def rev_rate(omegas,thetas,omega_max = -.2,theta_min = -3.e-4,theta_max = 3.e-4,minw = 8,bootstrap=False):
        sel1 = omegas<omega_max
        sel2 = np.logical_and(theta_min<thetas, thetas<theta_max)
        mask = np.logical_and(sel1,sel2)
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        sel = np.hstack(np.diff(segments))>minw
        if bootstrap:
            n_boot=1000
            sum_samples = np.array([sel[np.random.randint(0,len(sel),len(sel))].sum() for k in range(n_boot)])
            cil = np.percentile(sum_samples,2.5)/(len_w*dt/60)
            ciu = np.percentile(sum_samples,97.5)/(len_w*dt/60)
            return sel.sum()/(len_w*dt/60),cil,ciu
        return sel.sum()/(len_w*dt/60) #in per minute
    
    
    def ventral_turn_rate(thetas,theta_max = -3.5e-4,minw = 8,bootstrap=False):
        mask = thetas<theta_max
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        sel = np.hstack(np.diff(segments))>minw
        if bootstrap:
            n_boot=1000
            sum_samples = np.array([sel[np.random.randint(0,len(sel),len(sel))].sum() for k in range(n_boot)])
            cil = np.percentile(sum_samples,2.5)/(len_w*dt/60)
            ciu = np.percentile(sum_samples,97.5)/(len_w*dt/60)
            return sel.sum()/(len_w*dt/60),cil,ciu
        return sel.sum()/(len_w*dt/60) #in per minute


    def dorsal_turn_rate(thetas,theta_min = 3.5e-4,minw = 8 ,bootstrap=False):
        mask = thetas>theta_min
        segments = np.where(np.abs(np.diff(np.concatenate([[False], mask, [False]]))))[0].reshape(-1, 2)
        if len(segments>0):
            sel = np.hstack(np.diff(segments))>minw
            if bootstrap:
                n_boot=1000
                sum_samples = np.array([sel[np.random.randint(0,len(sel),len(sel))].sum() for k in range(n_boot)])
                cil = np.percentile(sum_samples,2.5)/(len_w*dt/60)
                ciu = np.percentile(sum_samples,97.5)/(len_w*dt/60)
                return sel.sum()/(len_w*dt/60),cil,ciu
            return sel.sum()/(len_w*dt/60) #in per minute
        else:
            if bootstrap:
                return 0,0,0
            return 0
    
    mat=h5py.File('/bucket/StephensU/antonio/ForagingN2_data/PNAS2011-DataStitched.mat','r')

    refs=list(mat['#refs#'].keys())[1:]
    tseries_w=[ma.masked_invalid(np.array(mat['#refs#'][ref]).T)[:,:5] for ref in refs]
    mat.close()
    frameRate=16.
    dt=1/frameRate
    ts_data = tseries_w[kw]
    len_w,dim = ts_data.shape
    
    f = h5py.File("/bucket/StephensU/antonio/ForagingN2_data/animations/tseries_sims_{}.h5".format(kw),'r')
    ts_sims = ma.array(f['ts_smooth_sims'])
    f.close()
        
    eigenworms_matrix = np.loadtxt('/bucket/StephensU/antonio/ForagingN2_data/EigenWorms.csv', delimiter=',').astype(np.float32)
    segments=op_calc.segment_maskedArray(ts_data,5)
    omegas = ma.zeros(ts_data.shape[0])
    for t0,tf in segments:
        phi,omega,a3=worm_dyn.compute_phi_omega_a3(ts_data,t0,tf)
        omegas[t0:tf] = omega
    omegas[omegas==0]=ma.masked

    thetas= ts_data.dot(eigenworms_matrix[:,:5].T)
    thetas_sum = thetas.sum(axis=1)
    
    n_sims=1000

    dorsal_turn_rate_sims = np.zeros((n_sims,3))
    ventral_turn_rate_sims = np.zeros((n_sims,3))
    rev_rate_sims = np.zeros((n_sims,3))
    
    rev_rate_data = np.array(rev_rate(omegas,thetas_sum,bootstrap=True))
    ventral_turn_rate_data = np.array(ventral_turn_rate(thetas_sum,bootstrap=True))
    dorsal_turn_rate_data = np.array(dorsal_turn_rate(thetas_sum,bootstrap=True))
    for ks in range(n_sims):
        sample_sim_ts = ts_sims[ks]
        phi_sim,omegas_sim,a3_sim=worm_dyn.compute_phi_omega_a3(sample_sim_ts,0,len_w)
        thetas_sim = sample_sim_ts.dot(eigenworms_matrix[:,:5].T)
        thetas_sim_sum = thetas_sim.sum(axis=1)
        rev_rate_sims[ks] = rev_rate(omegas_sim,thetas_sim_sum,bootstrap=True)
        ventral_turn_rate_sims[ks] = ventral_turn_rate(thetas_sim_sum,bootstrap=True)
        dorsal_turn_rate_sims[ks] = dorsal_turn_rate(thetas_sim_sum,bootstrap=True)
        if ks%10==0:
            print(ks,flush=True)
            
    f = h5py.File('/flash/StephensU/antonio/Foraging/sim_event_rates/rates_{}.h5'.format(kw),'w')
    rrd_ = f.create_dataset('rev_rate_data',rev_rate_data.shape)
    rrd_[...] = rev_rate_data
    vtd_ = f.create_dataset('ventral_turn_rate_data',ventral_turn_rate_data.shape)
    vtd_[...] = ventral_turn_rate_data
    dtd_ = f.create_dataset('dorsal_turn_rate_data',dorsal_turn_rate_data.shape)
    dtd_[...] = dorsal_turn_rate_data
    rrs_ = f.create_dataset('rev_rate_sims',rev_rate_sims.shape)
    rrs_[...] = rev_rate_sims
    vts_ = f.create_dataset('ventral_turn_rate_sims',ventral_turn_rate_sims.shape)
    vts_[...] = ventral_turn_rate_sims
    dts_ = f.create_dataset('dorsal_turn_rate_sims',dorsal_turn_rate_sims.shape)
    dts_[...] = dorsal_turn_rate_sims
    f.close()
    

if __name__ == "__main__":
    main(sys.argv)