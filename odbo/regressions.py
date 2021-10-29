import numpuy as np
import torch
import gpytorch
from botorch.fit import fit_gpytorch_model
from .gp import StudentTGP, GP

def GPRegression(X,
                 Y,
                 noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-2),
                 min_inferred_noise_level=1e-4,
                 **kwargs):
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.likelihoods import GaussianLikelihood
    from botorch.fit import fit_gpytorch_model
    likelihood = GaussianLikelihood(noise_constraint)
    model = GP(
        X,
        Y,
        likelihood=likelihood,
        min_inferred_noise_level=min_inferred_noise_level,
        **kwargs)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def RobustRegression(X,
                     Y,
                     noise_constraint=gpytorch.constraints.Interval(
                         1e-4, 1e-2),
                     min_inferred_noise_level=1e-4,
                     optimizer=None,
                     maxiter=100,
                     thresh=0.001,
                     std_factor=1.5,
                     **kwargs):
    from gpytorch.mlls import VariationalELBO
    from gpytorch.likelihoods import StudentTLikelihood
    likelihood = StudentTLikelihood(noise_constraint=noise_constraint)
    model = StudentTGP(
        X, Y, min_inferred_noise_level=min_inferred_noise_level, **kwargs)
    model.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, model, Y.ravel().numel())
    lossvalues = []
    if optimizer == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for i in range(maxiter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y.ravel())
        loss.backward()
        lossvalues.append(loss.item())
        if i >= 50 and abs(lossvalues[-1] -
                           np.mean(lossvalues[-10:])) <= thresh:
            break
        optimizer.step()
    model.eval()
    with torch.no_grad():
        observed_pred = model(X)
        pred_labels = np.mean(likelihood(observed_pred).mean.numpy(), axis=0)
        std = np.sqrt(observed_pred.variance.numpy())
        mean_std = np.mean(std)
        inlier_ids, outlier_ids = [], []
        for m in range(len(std)):
            if std[m] < std_factor * mean_std:
                inlier_ids.append(m)
            else:
                outlier_ids.append(m)
    return model, inliers, outlier_ids

