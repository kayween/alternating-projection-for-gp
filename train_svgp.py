import os

import torch
from torch.distributions import Normal

import gpytorch
import gpytorch.settings as settings

from data import load_uci_data

import argparse

from torch.utils.data import TensorDataset, DataLoader

from model import SVGP

import time


@torch.no_grad()
def test(model, likelihood, test_loader):
    model.eval()
    likelihood.eval()

    means = torch.tensor([0.], device=test_x.device)
    log_prob = torch.tensor([0.], device=test_x.device)

    for x_batch, y_batch in test_loader:
        pred = likelihood(model(x_batch))
        means = torch.cat([means, pred.mean])
        log_prob = torch.cat([
            log_prob,
            Normal(pred.mean, pred.stddev).log_prob(y_batch)]
        )

    means = means[1:]
    log_prob = log_prob[1:]

    rmse = test_y.sub(means).square().mean().sqrt()
    nll = -1. * log_prob.mean()

    model.train()
    likelihood.train()

    return rmse, nll


def train(
    model, likelihood, train_x, train_y, test_x, test_y,
    batch=1024, eta=0.1, maxiter=20, save_loc=None, last_ckpt_idx=-1,
):
    if not os.path.isdir(save_loc):
        print("creating folder \'{}\'".format(save_loc))
        os.mkdir(save_loc)

    if last_ckpt_idx >= 0:
        print("loading checkpoint...")
        ckpt = torch.load("{}/epoch_{}.tar".format(args.save_loc, last_ckpt_idx))
        model.load_state_dict(ckpt['model_state_dict'])
        likelihood.load_state_dict(ckpt['likelihood_state_dict'])

        print(
            "checkpoint parameters: ",
            "outputscale {:.4f},".format(model.covar_module.outputscale.item()),
            "avg lengthscale {:.4f},".format(model.covar_module.base_kernel.lengthscale.mean().item()),
            "noise {:.4f},".format(likelihood.noise.item()),
        )

    model.train()
    likelihood.train()

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=eta,
    )

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    for i in range(last_ckpt_idx + 1, last_ckpt_idx + 1 + maxiter):
        # Within each iteration, we will go over each minibatch of data
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()

            optimizer.step()

        with torch.no_grad():
            print(
                "iter {:3d}/{:3d},".format(i + 1, last_ckpt_idx + 1 + maxiter),
                "loss {:.6f},".format(loss.item()),
                "outputscale {:.4f},".format(model.covar_module.outputscale.item()),
                "avg lengthscale {:.4f},".format(model.covar_module.base_kernel.lengthscale.mean().item()),
                "noise {:.4f},".format(likelihood.noise.item()),
            )

            if (i + 1) % 10 == 0:
                print("evaluating test rmse and nll...")
                rmse, nll = test(model, likelihood, test_loader)
                print("rmse {:.6f}, nll {:.6f}".format(rmse.item(), nll.item()))

                torch.save(
                    {
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'likelihood_state_dict': likelihood.state_dict(),
                        'rmse': rmse.item(),
                        'nll': nll.item(),
                    }, "{}/epoch_{}.tar".format(save_loc, i)
                )

    model.eval()
    likelihood.eval()


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

    parser.add_argument("--num_inducing_points", type=int)
    parser.add_argument("--batch", type=int)

    parser.add_argument("--eta", type=float)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--save_loc", type=str, default="./checkpoints")
    parser.add_argument("--last_ckpt_idx", type=int, default=-1)

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

    model = SVGP(
        inducing_points=train_x[:args.num_inducing_points], kernel=args.kernel,
    ).to(device)

    start = time.time()
    with settings.max_cholesky_size(20000):
        model = train(
            model, likelihood, train_x, train_y, test_x, test_y,
            batch=args.batch, eta=args.eta, maxiter=args.maxiter,
            save_loc=args.save_loc, last_ckpt_idx=args.last_ckpt_idx,
        )
    end = time.time()

    print("total training time {:f}s".format(end - start))
