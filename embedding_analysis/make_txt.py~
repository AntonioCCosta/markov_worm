import numpy as np

worm_range = np.arange(12)
K_range = np.arange(1,61)

all_indices = []
for K in K_range:
    for worm in worm_range:
        all_indices.append([K,worm])

all_indices = np.array(np.vstack(all_indices),dtype=int)
np.savetxt('iteration_indices_K_1_60.txt',all_indices,fmt='%i')
print(len(all_indices))
