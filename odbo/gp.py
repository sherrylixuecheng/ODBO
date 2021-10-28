from __future__ import annotations
from typing import Any, Dict, Optional
import warnings
import torch
from torch import Tensor
from botorch import settings
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.utils.containers import TrainingData
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import StudentTLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.module import Module
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class GP(SingleTaskGP):
    def __init__(self,
                 train_X: Tensor,
                 train_Y: Tensor,
                 likelihood: Optional[Likelihood] = None,
                 covar_module: Optional[Module] = None,
                 outcome_transform: Optional[OutcomeTransform] = None,
                 input_transform: Optional[InputTransform] = None,
                 min_inferred_noise_level: Optional[Float] = 1e-4) -> None:
        if likelihood == None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (
                noise_prior.concentration - 1) / noise_prior.rate
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    self._min_inferred_noise_level,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        super(GP, self).__init__(X, Y, likelihood=likelihood)
        self._min_inferred_noise_level = min_inferred_noise_level


class StudentTGP(BatchedMultiOutputGPyTorchModel, ApproximateGP):
    def __init__(self,
                 train_X: Tensor,
                 train_Y: Tensor,
                 likelihood: Optional[Likelihood] = None,
                 covar_module: Optional[Module] = None,
                 outcome_transform: Optional[OutcomeTransform] = None,
                 input_transform: Optional[InputTransform] = None,
                 min_inferred_noise_level: Optional[Float] = 1e-4) -> None:
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        self._min_inferred_noise_level = min_inferred_noise_level
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            noise_prior = GammaPrior(1.1, 0.05)
            noise_prior_mode = (
                noise_prior.concentration - 1) / noise_prior.rate
            likelihood = StudentTLikelihood(
                noise_prior=noise_prior,
                batch_shape=self._aug_batch_shape,
                noise_constraint=GreaterThan(
                    self._min_inferred_noise_level,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        else:
            self._is_custom_likelihood = True
        variational_distribution = CholeskyVariationalDistribution(
            train_X.size(0))
        variational_strategy = VariationalStrategy(
            self,
            train_X,
            variational_distribution,
            learn_inducing_locations=False)
        ApproximateGP.__init__(self, variational_strategy)

        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=transformed_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        else:
            self.covar_module = covar_module
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData,
                         **kwargs: Any) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {"train_X": training_data.X, "train_Y": training_data.Y}
