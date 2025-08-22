import argparse
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
from functions import get_out_dirs, set_seed, setup_plotting, get_palette
from conformal.ot_predictors import (
    OTCPGlobal,
    OTCPAdaptiveKNN,
    RectangleGlobal,
)
from conformal.ot_predictors import quantile_map
from conformal.ellipsoidal_predictors import EllipsoidalGlobal


def sample_scores(n=1000):
    pi = np.array([3 / 8, 3 / 8, 1 / 4])
    M = np.array([[5, -5, 0], [0, 0, 0]])
    COV = np.array([[[4, -3], [-3, 4]], [[4, 3], [3, 4]], [[3, 0], [0, 1]]])
    d = 2
    y = np.zeros((n, d))
    cluster = np.random.choice(np.arange(3), size=n, p=pi)
    for i in range(n):
        y[i, :] = np.random.multivariate_normal(M[:, cluster[i]], COV[cluster[i]])
    return y


def mk_contour(otcp: OTCPGlobal, N: int = 100):
    angles = 2 * np.pi * np.linspace(0, 1, N)
    contour = np.array([np.cos(angles), np.sin(angles)]).T * otcp.quantile_threshold_
    return quantile_map(contour, otcp.data_calib_, otcp.psi_star_)


def rectangle_plot_patch(rect_predictor: RectangleGlobal, **kwargs):
    if rect_predictor.list_axis_ is None:
        raise RuntimeError("RectangleGlobal must be fit() before rectangle_plot_patch().")
    from matplotlib.patches import Rectangle

    lower_bounds = rect_predictor.list_axis_.T[1]
    upper_bounds = rect_predictor.list_axis_.T[0]
    width = upper_bounds[0] - lower_bounds[0]
    height = upper_bounds[1] - lower_bounds[1]
    return Rectangle((lower_bounds[0], lower_bounds[1]), width, height, **kwargs)


def _ellipse_matrix_to_param(mat: np.ndarray):
    lambda1 = (mat[0, 0] + mat[1, 1]) / 2 + np.sqrt(((mat[0, 0] - mat[1, 1]) / 2) ** 2 + mat[0, 1] ** 2)
    lambda2 = (mat[0, 0] + mat[1, 1]) / 2 - np.sqrt(((mat[0, 0] - mat[1, 1]) / 2) ** 2 + mat[0, 1] ** 2)
    if mat[0, 1] == 0 and mat[0, 0] >= mat[1, 1]:
        theta = 0
    elif mat[0, 1] == 0 and mat[0, 0] < mat[1, 1]:
        theta = np.pi / 2
    else:
        theta = np.arctan2(lambda1 - mat[0, 0], mat[0, 1])
    return np.sqrt(lambda1), np.sqrt(lambda2), theta


def ellipse_plot_patch(ell_predictor: EllipsoidalGlobal, **kwargs):
    if ell_predictor.cov_ is None:
        raise RuntimeError("EllipsoidalGlobal must be fit() before ellipse_plot_patch().")
    from matplotlib.patches import Ellipse

    width, height, angle = _ellipse_matrix_to_param(np.linalg.inv(ell_predictor.cov_ * ell_predictor.alpha_s_))
    return Ellipse(xy=ell_predictor.data_calib_mean_, width=width, height=height, angle=angle, **kwargs)


def run_synthetic_experiments(args):
    """Run synthetic experiments for OT-CP and baselines.

    Tasks (choose with --task):
      - shapes: Compare shapes/volumes of MK quantile region vs hyperrectangle (independent marginals) vs ellipsoid (Mahalanobis). Saves scatter and contour figure.
      - coverage_efficiency: Repeat calibration/test splits to estimate marginal coverage and average set volume for MK/RECT/ELL, including timing of calibration steps. Saves coverage/volume/time bar charts.
      - 3d_homo: Visualize MK quantile contours under homoscedastic noise; contours are images of circles in latent space under the OT map. Saves 3D plot.
      - 3d_hetero: Same visualization under heteroscedastic noise using local OT fits across a grid of x. Saves 3D plot.
      - conditional_coverage: Evaluate OT-CP+ (KNN-adaptive MK ranks) coverage across x-intervals [0,2], [0.25,0.5], [1.5,2]. Saves bar chart.
      - timing: Report calibration time vs calibration size for MK (global), rectangles, ellipsoids, and OT-CP+.

    Inputs:
      --alpha: confidence level; --seed: RNG seed; task-specific options like --reps, --k-neighbors, --n-test.
    Outputs:
      PDFs under outputs/ and (for coverage_efficiency) CSVs under csv/.
    """
    out_dir, csv_dir = get_out_dirs(args.out_dir, args.csv_dir)
    set_seed(args.seed)
    setup_plotting()
    pal = get_palette()

    if args.task in ("all", "shapes"):
        print("\n--- Running task: shapes ---")
        scores = sample_scores(n=2000) / 10
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.scatter(scores[:, 0], scores[:, 1], c="black", s=4)
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, "synthetic_scores_scatter.pdf"), bbox_inches="tight"
        )

        # Fit predictors
        X_placeholder = np.zeros((scores.shape[0], 1))
        y_scores = scores
        y_pred_zero = np.zeros_like(scores)
        otcp = OTCPGlobal(alpha=args.alpha).fit(X_placeholder, y_scores, y_pred_zero)
        rect_predictor = RectangleGlobal(alpha=args.alpha).fit(X_placeholder, y_scores, y_pred_zero)
        ell_predictor = EllipsoidalGlobal(alpha=args.alpha).fit(X_placeholder, y_scores, y_pred_zero)

        # Get contours and volumes
        contourMK = mk_contour(otcp)
        vol_mk = otcp.metrics(X_placeholder, y_scores, y_pred_zero)["avg_volume"]
        vol_rect = rect_predictor.metrics(X_placeholder, y_scores, y_pred_zero)["avg_volume"]
        vol_ell = ell_predictor.metrics(X_placeholder, y_scores, y_pred_zero)["avg_volume"]

        print("volume MK:", vol_mk)
        print("volume rectangle:", vol_rect)
        print("volume ellipse:", vol_ell)

        fig, ax = plt.subplots(figsize=(5, 5))
        df = pd.DataFrame(scores, columns=["X1", "X2"])
        sns.scatterplot(df, x="X1", y="X2", c="black", s=15, alpha=0.5)
        plt.plot(contourMK.T[0], contourMK.T[1], color=pal["mk"], linewidth=3)
        ax.add_patch(
            rectangle_plot_patch(rect_predictor, edgecolor=pal["y2"], facecolor="none", lw=3)
        )
        ax.add_patch(
            ellipse_plot_patch(ell_predictor, edgecolor=pal["ell"], facecolor="none", lw=3)
        )
        plt.axis("off")
        plt.savefig(os.path.join(out_dir, "quantiles_scores.pdf"), bbox_inches="tight")
        print("--- Task 'shapes' completed. ---")

    if args.task in ("all", "coverage_efficiency"):
        print("\n--- Running task: coverage_efficiency ---")
        results = []
        for rep in range(args.reps):
            scores_cal = sample_scores(n=1000) / 10
            scores_test = sample_scores(n=1000) / 10

            # Placeholders for X and y_pred
            X_cal_placeholder = np.zeros((scores_cal.shape[0], 1))
            X_test_placeholder = np.zeros((scores_test.shape[0], 1))
            y_pred_cal_zero = np.zeros_like(scores_cal)
            y_pred_test_zero = np.zeros_like(scores_test)

            # OT-CP
            t0 = time()
            otcp = OTCPGlobal(alpha=args.alpha).fit(X_cal_placeholder, scores_cal, y_pred_cal_zero)
            time_mk = time() - t0
            metrics_mk = otcp.metrics(X_test_placeholder, scores_test, y_pred_test_zero)

            # Rectangle
            t0 = time()
            rect_predictor = RectangleGlobal(alpha=args.alpha).fit(X_cal_placeholder, scores_cal, y_pred_cal_zero)
            time_rect = time() - t0
            metrics_rect = rect_predictor.metrics(X_test_placeholder, scores_test, y_pred_test_zero)

            # Ellipsoid
            t0 = time()
            ell_predictor = EllipsoidalGlobal(alpha=args.alpha).fit(X_cal_placeholder, scores_cal, y_pred_cal_zero)
            time_ell = time() - t0
            metrics_ell = ell_predictor.metrics(X_test_placeholder, scores_test, y_pred_test_zero)

            results.append(
                {
                    "method": "OT-CP",
                    "coverage": metrics_mk["avg_coverage"],
                    "volume": metrics_mk["avg_volume"],
                    "time": time_mk,
                }
            )
            results.append(
                {
                    "method": "RECT",
                    "coverage": metrics_rect["avg_coverage"],
                    "volume": metrics_rect["avg_volume"],
                    "time": time_rect,
                }
            )
            results.append(
                {
                    "method": "ELL",
                    "coverage": metrics_ell["avg_coverage"],
                    "volume": metrics_ell["avg_volume"],
                    "time": time_ell,
                }
            )

        df_results = pd.DataFrame(results)

        # Plotting
        for metric in ["coverage", "volume", "time"]:
            plt.figure()
            sns.barplot(
                data=df_results,
                x="method",
                y=metric,
                hue="method",
                palette=[pal["mk"], pal["y2"], pal["ell"]],
                capsize=0.3,
            )
            if metric == "coverage":
                plt.hlines(
                    args.alpha,
                    xmin=-0.6,
                    xmax=2.6,
                    linewidth=3,
                    linestyles="dashed",
                    color="black",
                )
                plt.ylim(0, 1)
                plt.ylabel("Marginal coverage")
            elif metric == "volume":
                plt.ylabel("Volume of prediction sets")
                plt.ylim(0.5, 1.5)
            else:
                plt.ylabel("Time")

            sns.despine(trim=True, left=True)
            plt.savefig(
                os.path.join(out_dir, f"Synthetic_{metric.capitalize()}.pdf"),
                bbox_inches="tight",
            )
        print("--- Task 'coverage_efficiency' completed. ---")

    if args.task in ("all", "3d_homo"):
        print("\n--- Running task: 3d_homo ---")
        n_sample = 999
        x = np.linspace(0, 2, n_sample)
        noise = sample_scores(n_sample) / 2
        Y = np.array([2 * x**2, (x + 1) ** 2]).T + noise
        y = Y.T[0]
        z = Y.T[1]
        BayesRegressor = np.array([x, 2 * x**2, (x + 1) ** 2]).T
        scores = Y - BayesRegressor[:, 1:]
        X_placeholder = np.zeros((scores.shape[0], 1))
        y_pred_zero = np.zeros_like(scores)
        otcp = OTCPGlobal(alpha=args.alpha).fit(X_placeholder, scores, y_pred_zero)
        k = 100
        grid = 6
        sphere = np.array(
            [np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)]
        ).T
        levels = np.array([0.2, 0.5, 0.8])
        contours = [
            quantile_map(a * sphere, otcp.data_calib_, otcp.psi_star_)
            for a in levels
        ]
        contours = np.array(contours)
        indices_grid = np.array(np.linspace(0, 0.999, grid) * len(x), dtype=int)
        x_tick = x[indices_grid]
        quantiles = []
        for i in range(len(levels)):
            Q0 = contours[i] + BayesRegressor[:, 1:][indices_grid][0]
            quantile = np.array([np.repeat(x_tick[0], k), Q0.T[0], Q0.T[1]]).T
            for g in range(1, grid):
                Q0 = contours[i] + BayesRegressor[:, 1:][indices_grid][g]
                Q0x = np.array([np.repeat(x_tick[g], k), Q0.T[0], Q0.T[1]]).T
                quantile = np.concatenate([quantile, Q0x])
            quantiles.append(quantile)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="3d")
        ax.scatter3D(x, y, z, color="gray", s=30, alpha=0.2, label="Test samples")
        ax.set_xlabel("X")
        ax.set_ylabel("$Y_1$")
        ax.set_zlabel("$Y_2$")
        ax.plot3D(
            BayesRegressor[:, 0],
            BayesRegressor[:, 1],
            BayesRegressor[:, 2],
            "black",
            alpha=1,
            linewidth=3.0,
            linestyle="-.",
            zorder=5,
            label="Prediction $\\hat{f}(x)$",
        )
        for q in quantiles:
            si = k
            for g in range(grid):
                r1 = g * si
                r2 = si * (g + 1)
                loop = np.zeros((si + 1, 3))
                loop[0:si, :] = q[r1:r2, :]
                loop[si, :] = q[r1, :]
                ax.plot3D(
                    loop[:, 0],
                    loop[:, 1],
                    loop[:, 2],
                    get_palette()["mk"],
                    alpha=1,
                    linewidth=2.0,
                    zorder=2,
                )
        ax.set_box_aspect(None, zoom=0.9)
        plt.savefig(
            os.path.join(out_dir, "HomoscedasticNoise.pdf"), bbox_inches="tight"
        )
        print("--- Task '3d_homo' completed. ---")

    if args.task in ("all", "3d_hetero"):
        print("\n--- Running task: 3d_hetero ---")
        n_sample = 999
        x = np.linspace(0, 2, n_sample)
        noise = sample_scores(n_sample) / 2
        Y = np.array([2 * x**2, (x + 1) ** 2]).T + noise * np.sqrt(x).reshape(
            n_sample, 1
        )
        y = Y.T[0]
        z = Y.T[1]
        n = 100
        BayesRegressor = np.array([x, 2 * x**2, (x + 1) ** 2]).T
        k = 100
        grid = 6
        indices_grid = np.array(np.linspace(0, 0.999, grid) * len(x), dtype=int)
        x_tick = x[indices_grid]
        sphere = np.array(
            [np.cos(2 * np.pi * np.arange(k) / k), np.sin(2 * np.pi * np.arange(k) / k)]
        ).T
        quantile_contours = []
        for i in range(grid):
            scores = Y - BayesRegressor[:, 1:]
            order = np.argsort(np.abs(x - x_tick[i]))
            scores = scores[order][:n]
            X_local_placeholder = np.zeros((scores.shape[0], 1))
            y_pred_local_zero = np.zeros_like(scores)
            otcp_local = OTCPGlobal(alpha=args.alpha).fit(X_local_placeholder, scores, y_pred_local_zero)
            Q0 = mk_contour(otcp_local)
            quantile_contours.append(Q0)
        quantile_contours = np.array(quantile_contours)
        quantiles = []
        g = 0
        Q0 = quantile_contours[g] + BayesRegressor[:, 1:][indices_grid][g]
        quantile = np.array([np.repeat(x_tick[g], k), Q0.T[0], Q0.T[1]]).T
        for g in range(1, grid):
            Q0 = quantile_contours[g] + BayesRegressor[:, 1:][indices_grid][g]
            Q0x = np.array([np.repeat(x_tick[g], k), Q0.T[0], Q0.T[1]]).T
            quantile = np.concatenate([quantile, Q0x])
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="3d")
        ax.scatter3D(x, y, z, color="gray", s=30, alpha=0.2)
        ax.set_xlabel("X")
        ax.set_ylabel("$Y_1$")
        ax.set_zlabel("$Y_2$")
        for g in range(grid):
            si = k
            r1 = g * si
            r2 = si * (g + 1)
            loop = np.zeros((si + 1, 3))
            q = quantile[r1:r2, :]
            loop[0:si, :] = q
            loop[si, :] = q[0]
            ax.plot3D(
                loop[:, 0],
                loop[:, 1],
                loop[:, 2],
                get_palette()["mk"],
                alpha=1,
                linewidth=2.0,
                zorder=2,
            )
        plt.savefig(
            os.path.join(out_dir, "HeteroscedasticNoise.pdf"), bbox_inches="tight"
        )
        print("--- Task '3d_hetero' completed. ---")

    if args.task in ("all", "conditional_coverage"):
        print("\n--- Running task: conditional_coverage ---")
        n_sample = 999
        x = np.linspace(0, 2, n_sample)
        noise2 = sample_scores(n_sample) / 2
        Y = np.array([2 * x**2, (x + 1) ** 2]).T + noise2 * np.sqrt(x).reshape(
            n_sample, 1
        )
        BayesRegressor = np.array([2 * x**2, (x + 1) ** 2]).T
        # Fit OTCPAdaptiveKNN on calibration
        predictor = OTCPAdaptiveKNN(alpha=args.alpha, n_neighbors=args.k_neighbors).fit(
            X_cal=x.reshape(-1, 1), y_cal=Y, y_pred_cal=BayesRegressor
        )

        def eval_interval(a, b):
            X_test = np.linspace(a, b, args.n_test)
            noise2 = sample_scores(args.n_test) / 2
            Y_test = np.array(
                [2 * X_test**2, (X_test + 1) ** 2]
            ).T + noise2 * np.sqrt(X_test).reshape(args.n_test, 1)
            Bayes = np.array([2 * X_test**2, (X_test + 1) ** 2]).T
            mask = predictor.contains(X_test.reshape(-1, 1), Y_test, Bayes)
            return float(np.mean(mask))

        resmarginal = eval_interval(0, 2)
        res1 = eval_interval(0.25, 0.5)
        res2 = eval_interval(1.5, 2)
        df = pd.DataFrame(
            [[resmarginal, res1, res2]], columns=["Marginal", "[0.25,0.5]", "[1.5,2]"]
        )
        g = sns.catplot(
            data=df.melt(var_name="Interval", value_name="Coverage"),
            kind="bar",
            x="Interval",
            y="Coverage",
            palette=[get_palette()["mk"]],
            aspect=0.8,
            capsize=0.4,
        )
        plt.hlines(
            args.alpha,
            xmin=-0.6,
            xmax=2.6,
            linewidth=3,
            linestyles="dashed",
            color="black",
        )
        plt.ylim(0, 1)
        sns.despine(trim=True, left=True)
        plt.savefig(
            os.path.join(out_dir, "SimuConditionalCVG.pdf"), bbox_inches="tight"
        )
        print("--- Task 'conditional_coverage' completed. ---")


def main():
    parser = argparse.ArgumentParser(description="Synthetic experiments for OT-CP.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=[
            "all",
            "shapes",
            "coverage_efficiency",
            "3d_homo",
            "3d_hetero",
            "conditional_coverage",
            "timing",
        ],
    )
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--csv-dir", type=str, default="csv")
    args = parser.parse_args()
    run_synthetic_experiments(args)


if __name__ == "__main__":
    main()
