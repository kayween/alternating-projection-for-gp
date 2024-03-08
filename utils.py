import os

import torch

import gpytorch
import gpytorch.settings as settings


def train(
    model, likelihood, train_x, train_y,
    eta=0.1, maxiter=20, save_loc=None,
):
    if not os.path.isdir(save_loc):
        print("creating folder \'{}\'".format(save_loc))
        os.mkdir(save_loc)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(maxiter):
        optimizer.zero_grad()
        settings.record_residual.lst_residual_norm = []

        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            print(
                "iter {:3d}/{:3d},".format(i + 1, maxiter),
                "loss {:.6f},".format(loss.item()),
                "outputscale {:.4f},".format(model.covar_module.outputscale.item()),
                "avg lengthscale {:.4f},".format(model.covar_module.base_kernel.lengthscale.mean().item()),
                "noise {:.4f},".format(model.likelihood.noise.item()),
                "CG/Altproj iterations {:4d}".format(len(settings.record_residual.lst_residual_norm)),
            )

            torch.save(
                {
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'lst_residual_norm': settings.record_residual.lst_residual_norm,
                }, "{}/epoch_{}.tar".format(save_loc, i)
            )

    model.eval()
    likelihood.eval()

    return model
