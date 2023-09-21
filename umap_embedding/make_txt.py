import numpy as np

n_neighbors_range = np.arange(10,101,10)
min_dist_range = np.arange(0.01,.11,.01)
all_indices=[]
for n in n_neighbors_range:
    for d in min_dist_range:
        all_indices.append([n,d])
all_indices = np.array(np.vstack(all_indices),dtype=float)
np.savetxt('iteration_indices.txt',all_indices)
print(len(all_indices))
