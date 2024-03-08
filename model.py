import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.size(-1))
            )
        elif kernel == "matern15":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1))
            )
        elif kernel == "matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1))
            )
        else:
            assert 0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGP(ApproximateGP):
    def __init__(self, inducing_points, kernel):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1))
            )
        elif kernel == "matern15":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=1.5, ard_num_dims=inducing_points.size(-1))
            )
        elif kernel == "matern":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=inducing_points.size(-1))
            )
        else:
            assert 0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
