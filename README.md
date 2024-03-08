# Large-Scale Gaussian Processes via Alternating Projection

This repository reproduces experiments in the paper [Large-Scale Gaussian Processes via Alternating Projection][link].
We are working on a more efficient implementation and plan to integrate it into GPyTorch.
Stay tuned!

[link]: https://arxiv.org/abs/2310.17137


## Dependency 

Install the following packages
- python==3.8
- pykeops==2.1.2
- pandas

You also need `git lfs` to download the data file in `./data`.

Install the GPyTorch branch that implements alternating projection
```
pip install git+https://github.com/cornellius-gp/gpytorch.git@altproj
```

The implementation is based on GPyTorch 1.6.
At the time of the development, GPyTorch >= 1.7 has a bug in the KeOps kernel which prevents training on large datasets.

In addition, we have provided a docker file that installs all dependencies.

## Run the Code
The following scripts train Gaussian processes from scratch
```
train_altproj.py # train Gaussian processes with alternating projection
train_cg.py      # train Gaussian processes with conjugate gradient
train_svgp.py    # train stochastic variational Gaussian processes
```

The following scripts evaluate pretrained Gaussian processes
```
test_altproj_inference.py # evaluate pretrained Gaussian processes with alternating projection
test_cg_inference.py      # evaluate pretrained Gaussian processes with conjugate gradient
```

The following script computes the predictive mean by solving a kernel linear system with either alternating projection or conjugate gradient on hyperparameters loaded from a checkpoint.
```
linear_solve_predictive_mean.py
```

For more details on their arguments, refer to the batch scripts in `./scripts/`.

## Checkpoints
Since training Gaussian processes on large-scale datasets is time consuming, we release the pretrained checkpoints under the folder `./checkpoints/`.
