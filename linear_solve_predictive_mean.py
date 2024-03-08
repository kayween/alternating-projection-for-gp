import torch
from torch.distributions import Normal

import gpytorch
import gpytorch.settings as settings

from data import load_uci_data

from model import ExactGPModel

import time


if __name__ == "__main__":
    print(gpytorch.__file__)
    print(gpytorch.__version__)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = "cuda:0"

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--kernel", type=str)

    parser.add_argument("--method", type=str, choices=["cg", "altproj"])
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--tol", type=float)

    parser.add_argument("--precond_size", type=int)
    parser.add_argument("--batch", type=int)

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    print(args)

    train_x, train_y, test_x, test_y = load_uci_data(
        "./uci/", args.dataset, seed=args.seed, device=device)[:4]

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel=args.kernel)

    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model_state_dict'])

    model.to(device).eval()
    likelihood.to(device).eval()

    print(
        "outputscale {:.4f},".format(model.covar_module.outputscale.item()),
        "avg lengthscale {:.4f},".format(model.covar_module.base_kernel.lengthscale.mean().item()),
        "noise {:.4f}".format(model.likelihood.noise.item()),
    )

    # inference
    start = time.time()

    if args.method == "cg":
        with torch.no_grad(), \
            settings.max_cholesky_size(0), \
            settings.max_preconditioner_size(args.precond_size), \
            settings.cg_tolerance(args.tol), \
            settings.eval_cg_tolerance(args.tol), \
            settings.max_cg_iterations(args.max_iterations), \
            settings.skip_posterior_variances(), \
            settings.verbose():

            print("use CG inference")
            settings.record_residual.lst_residual_norm = []

            pred_dist = likelihood(model(test_x))

            mean = pred_dist.mean
            rmse = mean.sub(test_y).pow(2).mean().sqrt()

            lst_residual_norm = settings.record_residual.lst_residual_norm

    elif args.method == "altproj":
        with torch.no_grad(), \
            settings.max_cholesky_size(0), \
            settings.max_preconditioner_size(0), \
            settings.use_alternating_projection(), \
            settings.altproj_batch_size(args.batch), \
            settings.cg_tolerance(args.tol), \
            settings.eval_cg_tolerance(args.tol), \
            settings.max_cg_iterations(args.max_iterations), \
            settings.skip_posterior_variances(), \
            settings.verbose():

            print("use alternating projection inference")
            settings.record_residual.lst_residual_norm = []

            pred_dist = likelihood(model(test_x))

            mean = pred_dist.mean
            rmse = mean.sub(test_y).pow(2).mean().sqrt()

            lst_residual_norm = settings.record_residual.lst_residual_norm

    end = time.time()

    print(
        "use {} in inference\n".format(args.method),
        "rmse {:.3f},".format(rmse.item()),
        "prediction time {:.0f}".format(end - start),
    )

    torch.save(
        {
            'args': args,
            'rmse': rmse.item(),
            'lst_residual_norm': lst_residual_norm,
            'prediction time': end - start,
        }, args.output
    )
