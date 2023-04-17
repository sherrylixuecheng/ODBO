import numpy as np
from .run_exp import bo_design, turbo_design
import random
import botorch
import torch
from gpytorch.utils.errors import NanError, NotPSDError
from .regressions import GPRegression, RobustRegression, HeteroskedasticGPRegression
from .utils import normalize_data
import turbo
import warnings


def circuit_optimize(init_X,
                     eval_objective,
                     total_cycles=1000,
                     method='turbo',
                     acqfn='ucb',
                     tr_length=[1.6],
                     batch_size=1,
                     failure_tolerance=10,
                     switch_counter=4,
                     verbose=True,
                     device=torch.device("cpu")):
    ### note ODBO is a maximization tool, but here we perform minimization.
    # 2nd point is randomly picked, assume all the parameters are in [-pi, pi]
    # The init_X should also be within  [-pi, pi]
    # Suggest to not optimize over 1000 cycles.
    # Example eval_objective could be defined as following
    '''
		def exp_val_wrapper(param):
		    global rkey
		    rkey, skey = K.random_split(rkey)
		    return exp_val(param, skey)

		def eval_objective(x, example_graph):
		    """This is a helper function we use to unnormalize and evalaute a point"""
		    a = tc.array_to_tensor(x, dtype=tc.rdtypestr).reshape(nlayers, 2)
		    m = exp_val_wrapper(a)
		    return -np.array(m)
	'''

    para_num = len(init_X)
    X_new = np.random.uniform(low=0, high=1, size=[1, para_num])
    paras = [np.array(init_X)]
    switch = 'small'

    X_turbo = torch.tensor(
        np.vstack(
            [np.array(init_X + np.pi).reshape(1, para_num) / 2 / np.pi,
             X_new]))
    Y_turbo = torch.tensor(
        [eval_objective(x * 2 * np.pi - np.pi, g) for x in X_turbo],
        dtype=dtype,
        device=device).unsqueeze(-1)
    paras.append(2 * np.pi * X_new.numpy() - np.pi)

    if method.lower() in ['turbo', 'switchturbo']:
        state = turbo.TurboState(dim=X_turbo.shape[1],
                                 batch_size=batch_size,
                                 length=tr_length,
                                 n_trust_regions=len(tr_length),
                                 failure_tolerance=failure_tolerance)
        state.best_value = Y_turbo.max()
    if method.lower() not in ['bo', 'turbo', 'switchturbo']:
        print(
            'The input method is not implemented. Please select from "bo", "turbo", "switchturbo"'
        )
        break

    if verbose == True:
        print('Initialization values: ', -Y_turbo.numpy())

    while len(Y_turbo) < total_cycles:
        if method.lower() == 'switchturbo':
            if switch_counter >= 4:
                if switch == 'small':
                    switch = 'big'
                    X_turbo = X_turbo / 2
                else:
                    switch = 'small'
                    X_turbo = X_turbo * 2
                switch_counter = 0

        if len(Y_turbo) > 1000:
            indice = torch.topk(input=torch.ravel(Y_turbo), k=1000)[1]
        else:
            indice = range(len(Y_turbo))

        if method.lower() == 'bo':
            X_next, acq_value, next_exp_id = bo_design(X=X_turbo[indice, :],
                                                       Y=Y_turbo[indice],
                                                       batch_size=batch_size,
                                                       acqfn=acqfn)
            Y_next = torch.tensor(
                [eval_objective(x * 2 * np.pi - pi, g) for x in X_next],
                dtype=dtype,
                device=device)
            Y_next_true = torch.tensor([
                eval_objective_true(x * 2 * np.pi - np.pi, g) for x in X_next
            ],
                                       dtype=dtype,
                                       device=device)
            paras.append([np.array(X_next) * 2 * np.pi - np.pi])
        elif method.lower() == 'turbo':
            a = 0.2
            X_next, acq_value, ids = turbo_design(
                state=state,
                X=X_turbo,
                Y=Y_turbo,
                n_trust_regions=len(tr_length),
                batch_size=batch_size,
                a=a,
                acqfn=acqfn,
                normalize=False,
                verbose=False)
            X_next = torch.reshape(X_next,
                                   [len(tr_length) * batch_size, para_num])
            # Update state
            state = turbo.update_state(state=state,
                                       Y_next=torch.reshape(
                                           Y_next,
                                           [len(tr_length), batch_size, 1]))
            paras.append([np.array(X_next) * 2 * np.pi - np.pi])

        elif method.lower() == 'switchturbo':
            a = 0.2
            X_next, acq_value, ids = turbo_design(
                state=state,
                X=X_turbo,
                Y=Y_turbo,
                n_trust_regions=len(tr_length),
                batch_size=batch_size,
                a=a,
                acqfn=acqfn,
                normalize=False,
                verbose=False)
            X_next = torch.reshape(X_next,
                                   [len(tr_length) * batch_size, 2 * nlayers])
            if switch == 'small':
                Y_next = torch.tensor(
                    [eval_objective(x * np.pi - np.pi / 2, g) for x in X_next],
                    dtype=dtype,
                    device=device)
                Y_next_true = torch.tensor([
                    eval_objective_true(x * np.pi - np.pi / 2, g)
                    for x in X_next
                ],
                                           dtype=dtype,
                                           device=device)
            else:
                Y_next = torch.tensor(
                    [eval_objective(x * 2 * np.pi - np.pi, g) for x in X_next],
                    dtype=dtype,
                    device=device)
                Y_next_true = torch.tensor([
                    eval_objective_true(x * 2 * np.pi - np.pi, g)
                    for x in X_next
                ],
                                           dtype=dtype,
                                           device=device)

            # Update state
            state = odbo.turbo.update_state(
                state=state,
                Y_next=torch.reshape(Y_next, [len(tr_length), batch_size, 1]))

            if np.max(Y_next.numpy()) < np.max(np.array(Y_turbo)):
                switch_counter = switch_counter + 1
                if verbose == True:
                    print('Current range: ', switch, ' counter: ',
                          switch_counter)
            else:
                switch_counter = 0
            if switch == 'small':
                paras.append([np.array(X_next) * np.pi - np.pi / 2])
            else:
                paras.append([np.array(X_next) * 2 * np.pi - np.pi])
            # Update state
            state = odbo.turbo.update_state(
                state=state,
                Y_next=torch.reshape(Y_next, [len(tr_length), batch_size, 1]))

    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next.unsqueeze(-1)), dim=0)
    bo_best.append(-Y_turbo.max())

    if verbose == True:
        if method.lower() == 'bo':
            print(
                f"{i+1}) Current best value: {-bo_best[-1]:.4e}, new value: {Y_next.numpy():.4e}, new parameters: {paras[-batch_size]:.4e}"
            )
        if method.lower() in ['turbo', 'switchturbo']:
            print(
                f"{i+1}) Current best value: {-bo_best[-1]:.4e}, new value: {Y_next.numpy():.4e}, new parameters: {paras[-batch_size]:.4e}, TR length: {state.length}"
            )
    # return all the evaluated values, corresponding parameters & the final best value as numpy array
    return -Y_turbo.numpy(), paras, bo_best[-1].numpy()
