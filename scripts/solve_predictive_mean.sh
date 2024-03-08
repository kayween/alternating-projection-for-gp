#!/bin/bash

python linear_solve_predictive_mean.py \
       --dataset 3droad --seed 0 --kernel matern \
       --method altproj --max_iterations 4000 --tol 1e-2 --batch 6000 \
       --checkpoint ./checkpoints/altproj/3droad-matern-0/epoch_49.tar \
       --output ./outputs/tmp.tar

python linear_solve_predictive_mean.py \
       --dataset 3droad --seed 0 --kernel matern \
       --method cg --max_iterations 4000 --tol 1e-2 --precond_size 500 \
       --checkpoint ./checkpoints/altproj/3droad-matern-0/epoch_49.tar \
       --output ./outputs/tmp.tar



