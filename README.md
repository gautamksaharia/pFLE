# pFLE
# Dynamics of ion Cyclotron wave under Perturbation

Physics-Informed Neural Network(PyTorch) to simulate bright soliton of the perturbed Fokas Lenells Equation.

## Features

- Custom PINN model with SIREN
- Adam + L-BFGS optimization


## Datasets

Data generation scripts are available in the [`data`](data) folder in respective experiment:

- [Linear Damping in Collisional Plasma](src/LinearDamping-collisional)
- [Linear Damping in Collisionless Plasma](src/LinearDamping-collisionless)
- [Stochastic Perturbation](src/stochastic-perturbation)

## Code
All code is provided in the [`src`](src) folder


## Cite this work

If you use this data or code for academic research, please cite the following paper:

```bibtex
@article{saharia2026dynamics,
  title   = {Dynamics of Ion Cyclotron Wave under Perturbed Environment using Physics Informed Neural Networks},
  author  = {Gautam Kumar Saharia, Sagardeep Talukdar, Riki Dutta, Sudipta Nandy},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {2026},
  doi     = {https://doi.org/10.21203/rs.3.rs-8651617/v1}
}


  
