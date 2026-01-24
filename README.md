# pFLE
# Dynamics of ion Cyclotron wave under Perturbation

Physics-Informed Neural Network(PyTorch) to simulate bright soliton of the perturbed Fokas Lenells Equation.

## Features

- Custom PINN model with SIREN
- Adam + L-BFGS optimization


## Datasets

Data generation scripts are available in the [`data`](data) folder in respective experiment:

- [Linear Damping in Collisional Plasma](src/LinearDamping-collisional)
- [Burgers equation](data/burgers)
- [Diffusion–reaction equation](data/diffusion_reaction)
- [Lid-driven cavity flow](data/lid_driven_cavity)

Helper functions for data generation in a federated learning setting (1D and 2D) are provided in  
[`data_assignment.py`](data_assignment.py).

Details are described in **Section II** of the paper.
