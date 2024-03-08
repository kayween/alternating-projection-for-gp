#!/bin/bash

python test_altproj_inference.py --dataset 3droad --seed 0 --kernel matern \
    --batch 6000 --max_cg_iterations 4000 --max_lanczos_iterations 1000 \
    --checkpoint ./checkpoints/altproj/3droad-matern-0/epoch_49.tar \
    --output ./outputs/tmp.tar

python test_cg_inference.py --dataset 3droad --seed 0 --kernel matern \
    --precond_size 500 --max_cg_iterations 4000 --max_lanczos_iterations 1000 \
    --checkpoint ./checkpoints/cg/3droad-matern-0/epoch_49.tar \
    --output ./outputs/tmp.tar