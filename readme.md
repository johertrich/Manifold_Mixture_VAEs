# Manifold Learning by Mixture Models of VAEs for Inverse Problems

This repository contains the code for the paper "Manifold Learning by Mixture Models of VAEs for Inverse Problems" available at  
https://jmlr.org/papers/v25/23-0396.html  
Please cite the paper, if you use the code.

This repository contains code for approximating the charts of a manifold by a mixture model of VAEs. More precisely, the following examples are implemented.

- The script `manifold_vae_toy.py`, `manifold_vae_bar.py` and `manifold_vae_balls.py` contain the source code for training the mixtures of VAEs in Section 5, 6.1 and 6.2.

- The script `move_on_toy.py` reproduces the trajectories from Figure 6.

- The scripts `move_on_bar_fig.py` and `move_on_bar_trajectories.py` reproduce the deblurring experiments from Figure 7 and 8.

- The script `move_on_balls_calderon.py` reproduces the EIT experiment from Section 6.2.

The code is written with PyTorch 1.12.0. The EIT experiment uses version 2019.1.0 of the Fenics library.

For questions, bugs or any other comments, please contact Johannes Hertrich (an up-to date email address can be found [here](https://johertrich.github.io)).

## Citation

```
@article{ManifoldVAEs,
  author  = {Giovanni S. Alberti and Johannes Hertrich and Matteo Santacesaria and Silvia Sciutto},
  title   = {Manifold Learning by Mixture Models of VAEs for Inverse Problems},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {202},
  pages   = {1--35},
  url     = {http://jmlr.org/papers/v25/23-0396.html}
}
```
