#!/bin/bash

python train_altproj.py --dataset 3droad --seed 0 --kernel matern --noise_constraint 1e-4 --eta 0.1 --maxiter 50 --batch 6000 --tol 1. --save_loc ./checkpoints/tmp

python train_cg.py --dataset 3droad --seed 0 --kernel matern --noise_constraint 1e-4 --eta 0.1 --maxiter 50 --precond_size 500 --max_cg_iterations 1000 --tol 1. --save_loc ./checkpoints/tmp

# train a SVGP for 50 epochs
python train_svgp.py --dataset 3droad --seed 0 --kernel matern --num_inducing_points 1024 --batch 4096 --eta 1e-2 --maxiter 50 --save_loc ./checkpoints/tmp

# train the SVGP for another 50 epochs from a previous checkpoint
python train_svgp.py --dataset 3droad --seed 0 --kernel matern --num_inducing_points 1024 --batch 4096 --eta 1e-2 --maxiter 50 --save_loc ./checkpoints/tmp --last_ckpt_idx 49
