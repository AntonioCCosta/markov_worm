# markov_worm

This repository contains the scripts for reproducing the results presented in

Costa AC, Ahamed T, Jordan D, Stephens GJ (2024) "A Markovian dynamics for *C. elegans* behavior across scales" [PNAS](https://www.pnas.org/doi/10.1073/pnas.2318805121)

This model is able to predict worm foraging behavior across scales, from sub-second posture movements to minutes long search strategies, bridging from posture to path through a combination of delay embedding, Markov modelling and resistive force theory.

See example [reconstruction of path from posture](https://antonioccosta.github.io/download/combined_traj.mp4), and a [comparison between simulations and data](https://antonioccosta.github.io/download/postures_sim_vs_data.mp4).

To run the scripts and jupyter notebooks to reproduce the figures, you'll need to [download the dataset at https://doi.org/10.34740/kaggle/ds/3882219](https://doi.org/10.34740/kaggle/ds/3882219), and update the paths to the ./utils folder and the data folders.

The following files correspond to the dataset: 

(1) data/Foraging_N2/EigenWorms.csv \
(2) data/Foraging_N2/shapes.mat \
(3) data/Foraging_N2/PNAS2011-DataStitched.mat 

(1) was analyzed in Stephens et al. (2008) PLoS Comput. Biol. to produce (2), and (3) results from the work on Broekmans et al. (2016) eLife.

In the dataset, we also include some files used in the notebooks that result from calculations obtained with the python scripts within each folder. For example 

(1) "data/tscales_P/results_{}.h5" results from running "./tscales_eigvals/compute_noise_floor_P.py"; \
(2) "data/Foraging_N2/phspace_K_10_m_7.h5" results from running "./symbol_sequences/get_phspace_labels.py"; \
(3) "data/labels_tree/labels_tree.h5" results from running "./subdivide_states/subdivide_states.py". 


Our calculations were performed using Python 3.7.3 and the following packages:

- numpy 1.18.3
- scipy 1.7.3
- scikit-learn 0.22
- msmtools 1.2.4
- umap 0.5.3 
