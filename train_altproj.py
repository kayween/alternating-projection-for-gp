import torch

import gpytorch
import gpytorch.settings as settings

from data import load_uci_data

import argparse

from model import ExactGPModel

from utils import train

import time

if __name__ == "__main__":
    print(gpytorch.__file__)
    print(gpytorch.__version__)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = "cuda:0"

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=str, choices=['matern15', 'matern'])
    parser.add_argument("--noise_constraint", type=float, default=1e-4)

    parser.add_argument("--eta", type=float)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--save_loc", type=str, default="./checkpoints")

    args = parser.parse_args()
    print(args)

    train_x, train_y, test_x, test_y = load_uci_data(
        "./uci/", args.dataset, seed=args.seed, device=device)[:4]

    print(train_x.size(), train_x.device)
    print(test_x.size(), test_x.device)

    # training
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(args.noise_constraint)
    ).to(device)

    model = ExactGPModel(
        train_x, train_y, likelihood, args.kernel,
    ).to(device)

    start = time.time()
    with settings.max_cholesky_size(0), \
         settings.max_preconditioner_size(0), \
         settings.skip_logdet_forward(), \
         settings.use_alternating_projection(), \
         settings.altproj_batch_size(args.batch), \
         settings.cg_tolerance(args.tol), \
         settings.max_cg_iterations(2000):
        model = train(
            model, likelihood, train_x, train_y,
            eta=args.eta, maxiter=args.maxiter, save_loc=args.save_loc,
        )
    end = time.time()

    print("total training time {:f}s".format(end - start))
