import numpy as np

worm_range = np.arange(12)
n_seeds = 50
l=[]
for kw in worm_range:
    for idx in range(n_seeds):
        l.append([kw,idx])
    print(kw)
l=np.array(l)
print(l.shape)
np.savetxt('indices.txt',l)
