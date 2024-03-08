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
    parser.add_argument("--precond_size", type=int)
    parser.add_argument("--max_cg_iterations", type=int)
    parser.add_argument("--max_lanczos_iterations", type=int)
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

    # CG inference
    start = time.time()
    with torch.no_grad(), \
         settings.max_preconditioner_size(args.precond_size), \
         settings.cg_tolerance(0.01), \
         settings.eval_cg_tolerance(0.01), \
         settings.max_cg_iterations(args.max_cg_iterations), \
         settings.fast_pred_var(), \
         settings.max_root_decomposition_size(args.max_lanczos_iterations), \
         settings.verbose():
        
        print("start CG in inference...")
        start_mean = time.time()
        with settings.skip_posterior_variances():
            pred_dist = likelihood(model(test_x))
            mean = pred_dist.mean
        end_mean = time.time()
        print("finish predicitve mean")

        print("start Lanczos iterations...")
        pred_dist = likelihood(model(test_x))
        stddev = pred_dist.stddev

        rmse = mean.sub(test_y).pow(2).mean().sqrt()
        nll = -1. * Normal(mean, stddev).log_prob(test_y).mean()
    end = time.time()

    print(
        "rmse {:.3f},".format(rmse.item()),
        "nll {:.3f},".format(nll.item()),
        "predictive mean time {:.0f},".format(end_mean - start_mean),
        "prediction time {:.0f}".format(end - start),
    )

    torch.save(
        {
            'args': args,
            'rmse': rmse.item(),
            'nll': nll.item(),
            'lst_residual_norm': settings.record_residual.lst_residual_norm,
            "predictive mean time": end_mean - start_mean,
            'prediction time': end - start,
        }, args.output
    )
