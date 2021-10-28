import botorch
import torch
import botorch.acquisition as acqf


def generate_batch(model,
                   X,
                   Y,
                   batch_size,
                   X_pending=None,
                   n_candidates=None,
                   num_restarts=10,
                   raw_samples=512,
                   acqfn='ei',
                   **kwargs):
    dtype = X.dtype
    device = X.get_device()
    if acqfn == 'ts':
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
        from contextlib import ExitStack
        from botorch.generation import MaxPosteriorSampling
        if X_pending == None:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(X.shape[-1], scramble=True)
            X_cand = sobol.draw(n_candidates)
        else:
            if X_pending.shape[0] < n_candidates:
                n_candidates = int(0.8 * X_pending.shape[0])
            id_choice = np.random.choice(
                range(X_pending.shape[0]), n_candidates, replace=False)
            X_cand = X_pending[id_choice, :].to(dtype=dtype, device=device)

        with ExitStack() as es:
            es.enter_context(gpts.max_cholesky_size(float("inf")))
            thompson_sampling = MaxPosteriorSampling(
                model=model, replacement=False)
            X_next = thompson_sampling(X_cand, num_samples=batch_size)
            acq_value = None
    else:
        from botorch.optim import optimize_acqf, optimize_acqf_discrete
        if acqfn == "ei":
            acq = acqf.monte_carlo.qExpectedImprovement(model, Y.max())
        elif acqfn == "pi":
            acq = acqf.monte_carlo.qProbabilityOfImprovement(model, Y.max())
        if acqfn == "ucb":
            acq = acqf.monte_carlo.qUpperConfidenceBound(model, 0.1)

        if X_pending == None:
            X_next, acq_value = optimize_acqf(
                acq,
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                **kwargs)
        else:
            X_next, acq_value = optimize_acqf_discrete(
                acq,
                choices=X_pending,
                q=batch_size,
                max_batch_size=raw_samples,
                **kwargs)

    return X_next, acq_value
