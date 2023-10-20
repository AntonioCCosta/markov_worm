import numpy as np
import h5py
import os

def main():
    results_path = 'path_to_data/embedding_results/'
    K_range=np.arange(1,61)
    #sample file
    f = h5py.File(results_path+'entropic_properties_K_1_0.h5','r')
    seed_range = np.array(f['seed_range'])
    f.close()

    n_segs = 12
    seg_range = np.arange(n_segs)

    probs_K_s = np.zeros((len(K_range),len(seed_range),n_segs))
    H_K_s = np.zeros((len(K_range),len(seed_range),n_segs))
    Ipred_K_s = np.zeros((len(K_range),len(seed_range),n_segs))
    h_K_s = np.zeros((len(K_range),len(seed_range),n_segs))
    eps_K_s = np.zeros((len(K_range),len(seed_range),n_segs))
    for k,K in enumerate(K_range):
        for seg_idx in seg_range:
            try:
                f = h5py.File(results_path+'entropic_properties_K_{}_{}.h5'.format(K,seg_idx),'r')
                probs_K_s[k,:,seg_idx] = np.array(f['probs'])
                H_K_s[k,:,seg_idx] = np.array(f['entropies'])
                h_K_s[k,:,seg_idx] = np.array(f['entropy_rates'])
                Ipred_K_s[k,:,seg_idx] = np.array(f['Ipreds'])
                eps_K_s[k,:,seg_idx] = np.array(f['eps_scale'])
                f.close()
            except:
                print('Could not compute for K = {} and idx={}'.format(K,seg_idx))
    f = h5py.File('path_to_data/partition_combined_results.h5','w')
    probs_ = f.create_dataset('probs',probs_K_s.shape)
    probs_[...] = probs_K_s
    H_ = f.create_dataset('entropies',H_K_s.shape)
    H_[...] = H_K_s
    h_ = f.create_dataset('entropy_rates',h_K_s.shape)
    h_[...] = h_K_s
    I_ = f.create_dataset('Ipreds',Ipred_K_s.shape)
    I_[...] = Ipred_K_s
    eps_ = f.create_dataset('eps_scale',eps_K_s.shape)
    eps_[...] = eps_K_s
    K_ = f.create_dataset('K_range',K_range.shape)
    K_[...]= K_range
    seed_ = f.create_dataset('seed_range',seed_range.shape)
    seed_[...] = seed_range
    f.close()



if __name__ == "__main__":
    main()
