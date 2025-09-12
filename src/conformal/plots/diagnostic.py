import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def draw_qq_scores(scores: np.ndarray, ax):
    """
    Draw a Quantile-Quantile plot of the multidimensional scores.
    Assumed standard normal distribution.
    """
    n, d = scores.shape
    for j in range(d):
        stats.probplot(
            scores[:, j],
            dist="norm",
            fit=False,
            plot=ax,
        )
        ax.get_lines()[j].set_markerfacecolor(f'C{j}')
        ax.get_lines()[j].set_markeredgecolor(f'C{j}')
    ax.plot([-3, 3], [-3, 3], ls='--', c='k')


def draw_qq_scores_pair(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    title_1="Calibration",
    title_2="Test",
    save_path: str | None = None
):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    draw_qq_scores(scores_1, ax[0])
    ax[0].set_title(title_1)
    draw_qq_scores(scores_2, ax[1])
    ax[1].set_title(title_2)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    #return fig, ax


def draw_density_scores_pair(
    scores_1,
    scores_2,
    title_1="Calibration",
    title_2="Test",
    save_path: str | None = None
):
    t_min = min(scores_1.min(), scores_2.min())
    t_max = max(scores_1.max(), scores_2.max())
    t = np.linspace(t_min, t_max, 500)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    p0 = ax[0].plot(t, stats.norm.pdf(t), 'k--', label=r"$\mathcal{N}(0, 1)$")
    hp0 = sns.histplot(
        scores_1, alpha=0.3, stat="density", common_norm=False, kde=True, ax=ax[0]
    )
    ax[0].set_title(title_1)
    ax[0].set_xlim(-5, 5)
    hp1 = sns.histplot(
        scores_2, alpha=0.3, stat="density", common_norm=False, kde=True, ax=ax[1]
    )
    p1 = ax[1].plot(t, stats.norm.pdf(t), 'k--', label=r"$\mathcal{N}(0, 1)$")
    ax[1].set_title(title_2)
    ax[1].set_xlim(-5, 5)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    #return fig, ax
