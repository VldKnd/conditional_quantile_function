import numpy as np
from scipy.stats import qmc
from scipy.stats import norm

from conformal.otcp.functions import T0, learn_psi


def sample_grid_refactor(data, seed=0, positive=False):
    ''' Sample the reference distribution.'''
    n = data.shape[0]
    d = data.shape[1]
    R = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    if positive == False:
        sampler = qmc.Halton(d=d, rng=rng)
        sample_gaussian = sampler.random(n=n + 1)[1:]
        sample_gaussian = norm.ppf(sample_gaussian, loc=0, scale=1)
        mu = []
        for i in range(n):
            Z = sample_gaussian[i]
            Z = Z / np.linalg.norm(Z)
            mu.append(R[i] * Z)
    else:
        mu = []
        for i in range(n):
            Z = rng.exponential(scale=1.0, size=d)
            Z = Z / np.sum(Z)
            mu.append(R[i] * Z)
    return (np.array(mu))


def RankFuncRefactor(x, mu, psi, ksi=0):
    if ksi == 0:
        # For exact recovery of the argsup, one can use T0.
        res = T0(x, mu, psi)
    else:
        # ksi >0 computes a smooth argmax (LogSumExp). ksi is a regularisation parameter, hence approximates the OT map.
        if (len(x.shape) == 1):
            to_max = ((mu @ x) - psi) * ksi
            to_sum = np.exp(to_max - np.max(to_max))
            weights = to_sum / (np.sum(to_sum))
            res = np.sum(mu * weights.reshape(len(weights), 1), axis=0)
        else:
            res = []
            for xi in x:
                to_max = ((mu @ xi) - psi) * ksi
                to_sum = np.exp(to_max - np.max(to_max))
                weights = to_sum / (np.sum(to_sum))
                res.append(np.sum(mu * weights.reshape(len(weights), 1), axis=0))
            res = np.array(res)
    return (res)


def MultivQuantileTresholdRefactor(
    data_calib, data_valid, alpha=0.9, seed=0, positive=False
):
    ''' To change the reference distribution towards a positive one, set positive = True.  '''
    # Solve OT
    mu = sample_grid_refactor(data_calib, seed=seed, positive=positive)
    psi, psi_star = learn_psi(mu, data_calib)

    # QUANTILE TRESHOLDS
    n = len(data_valid)
    Ranks_data_valid = RankFuncRefactor(data_valid, mu, psi)
    Norm_ranks_valid = np.linalg.norm(Ranks_data_valid, axis=1, ord=2)
    Quantile_Treshold = np.quantile(
        Norm_ranks_valid, np.min([np.ceil((n + 1) * alpha) / n, 1])
    )
    return (Quantile_Treshold, mu, psi, psi_star, data_calib)
