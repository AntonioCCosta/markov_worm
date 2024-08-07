# markov_worm

This repository contains the scripts for reproducing the results presented in

Costa AC, Ahamed T, Jordan D, Stephens GJ (2024) "A Markovian dynamics for *C. elegans* behavior across scales" [PNAS](https://www.pnas.org/doi/10.1073/pnas.2318805121)

This model is able to predict worm foraging behavior across scales, from sub-second posture movements to minutes long search strategies, bridging from posture to path through a combination of delay embedding, Markov modelling and resistive force theory. To run the scripts and jupyter notebooks to reproduce the figures, you'll need to [download the dataset at https://doi.org/10.34740/kaggle/ds/3882219](https://doi.org/10.34740/kaggle/ds/3882219), and update the paths to the ./utils folder and the data folders.

See example [reconstruction of path from posture](https://antonioccosta.github.io/download/combined_traj.mp4), and a [comparison between simulations and data](https://antonioccosta.github.io/download/postures_sim_vs_data.mp4).

Our calculations were performed using Python 3.7.3 and the following packages:

- numpy 1.18.3
- scipy 1.7.3
- scikit-learn 0.22
- msmtools 1.2.4
- umap 0.5.3 
