import numpy as np
from sklearn.model_selection import train_test_split


# Worst-slab coverage, as introduced in:
# 1. Knowing what You Know: valid and validated confidence sets in multiclass and multilabel prediction
#    M. Cauchois, S. Gupta, J. Duchi - Journal of machine learning research, 2021
#    https://jmlr.csail.mit.edu/papers/volume22/20-753/20-753.pdf
# 2. Classification with Valid and Adaptive Coverage
#    Y. Romano, M. Sesia, E. Candes - Advances in Neural Information Processing Systems 33 (NeurIPS 2020)
#    https://arxiv.org/pdf/2006.02544
# Code adapted from:
# 1. https://github.com/msesia/chr/blob/master/chr/coverage.py
# 2. https://github.com/Vekteur/multi-output-conformal-regression/blob/master/moc/metrics/conditional_coverage_metrics.py

def wsc(reprs, coverages, delta, M=1000, random_state=42, n_cpus=1):

    def wsc_v(reprs, cover, delta, v):
        n = len(cover)
        z = np.dot(reprs, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best, bi_best = 0, n - 1
        cover_min = cover.mean()
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(
                1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best, bi_best = ai, bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def wsc_v_fully_vectorized(v, reprs, cover, delta):
        # Ensure inputs are numpy arrays
        reprs = np.atleast_2d(reprs)
        cover = np.atleast_1d(cover)
        v = np.atleast_1d(v)

        # Calculate z and sort
        z = np.dot(reprs, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]

        # Calculate parameters
        n = len(cover)
        ai_max = int(np.round((1.0 - delta) * n))
        delta_n = int(np.round(delta * n))

        # Prepare arrays for vectorized operations
        ai_range = np.arange(ai_max)
        bi_range = np.arange(n)
        cumsum_cover = np.cumsum(cover_ordered)

        # Create meshgrid for all possible (ai, bi) pairs
        ai_mesh, bi_mesh = np.meshgrid(ai_range, bi_range, indexing='ij')

        # Calculate coverage for all pairs
        denominator = bi_mesh - ai_mesh + 1
        numerator = cumsum_cover[bi_mesh] - np.where(
            ai_mesh > 0, cumsum_cover[ai_mesh - 1], 0)

        coverage = np.full_like(denominator, np.inf, dtype=float)
        valid_mask = (bi_mesh >= ai_mesh) & (denominator > 0)
        coverage[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        # Set coverage to 1 for invalid pairs (bi < ai + delta_n)
        coverage[bi_mesh < ai_mesh + delta_n] = 1

        # Find the minimum coverage and corresponding indices
        min_coverage = np.min(coverage)
        ai_best, bi_best = np.unravel_index(np.argmin(coverage),
                                            coverage.shape)

        return min_coverage, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p, seed=42):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((p, n))
        #v = np.random.randn(p, n)
        v /= np.linalg.norm(v, axis=0)
        return v.T

    def sample_from_normal_approximation(n, X):
        mean = np.mean(X, axis=0)
        covariance = np.cov(X, rowvar=False)
        return np.random.multivariate_normal(mean, covariance, size=n)

    V = sample_sphere(M, reprs.shape[1], seed=random_state)
    # V = sample_from_normal_approximation(M, reprs)
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M

    if n_cpus < 2:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v_fully_vectorized(
                V[m], reprs, coverages, delta)
    else:
        import tqdm_pathos
        res = tqdm_pathos.map(
            wsc_v_fully_vectorized,
            V,
            reprs,
            coverages,
            delta,
            n_cpus=n_cpus,
            #tqdm_kwargs={"disable": not verbose}
        )
        wsc_list, a_list, b_list = zip(*res)

    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(reprs,
                 coverages,
                 delta,
                 M=1000,
                 test_size=0.75,
                 random_state=0,
                 n_cpus=1):

    def wsc_vab(reprs, cover, v, a, b):
        n = len(reprs)
        z = np.dot(reprs, v)
        idx = np.where((a <= z) * (z <= b))
        coverage = np.mean(cover[idx])
        return coverage

    (
        reprs_train,
        reprs_test,
        coverages_train,
        coverages_test,
    ) = train_test_split(reprs,
                         coverages,
                         test_size=test_size,
                         random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(reprs_train,
                                           coverages_train,
                                           delta=delta,
                                           M=M,
                                           random_state=random_state,
                                           n_cpus=n_cpus)
    # print(wsc_star, v_star, a_star, b_star)
    # Estimate coverage
    coverage = wsc_vab(reprs_test, coverages_test, v_star, a_star, b_star)
    return coverage
