"""Enhanced EM convergence diagnostics (multi-panel).

P2.5: Three-panel figure showing frequency convergence, mu trajectory,
and T trajectory across EM iterations.
"""

from __future__ import annotations

from pathlib import Path

from ._style import ancestry_colors, ancestry_names, popout_style
from ._loaders import read_summary, read_stats_jsonl


def plot_convergence(
    prefix: str | Path,
    *,
    labels: dict | None = None,
    figsize: tuple[float, float] = (16, 5),
) -> "matplotlib.figure.Figure":
    """Enhanced EM convergence: freq delta, mu trajectory, T trajectory.

    Parameters
    ----------
    prefix : path prefix
    labels : optional labels dict from read_labels_json()
    figsize : figure size
    """
    import matplotlib.pyplot as plt

    prefix = Path(prefix)

    # Try stats.jsonl first (richer data), fall back to summary.json
    stats_path = prefix.with_name(prefix.name + ".stats.jsonl")
    summary_path = prefix.with_name(prefix.name + ".summary.json")

    iterations = []
    max_deltas = []
    mean_deltas = []
    mu_history = []  # list of (iter, [mu_0, mu_1, ...])
    t_history = []   # list of (iter, T)

    if stats_path.exists():
        records = read_stats_jsonl(stats_path)
        for r in records:
            key = r.get("key", "")
            if key == "em/max_delta_freq":
                it = r.get("context", {}).get("iteration", len(max_deltas))
                iterations.append(it)
                max_deltas.append(r["value"])
            elif key == "em/mean_delta_freq":
                mean_deltas.append(r["value"])
            elif key == "em/mu":
                it = r.get("context", {}).get("iteration", len(mu_history))
                mu_history.append((it, r["value"]))
            elif key == "em/T":
                it = r.get("context", {}).get("iteration", len(t_history))
                t_history.append((it, r["value"]))
    elif summary_path.exists():
        summary = read_summary(summary_path)
        for rec in summary.get("em_convergence", []):
            if "max_delta_freq" in rec:
                iterations.append(rec.get("iteration", len(iterations)))
                max_deltas.append(rec["max_delta_freq"])
            if "mean_delta_freq" in rec:
                mean_deltas.append(rec["mean_delta_freq"])

    if not iterations:
        raise ValueError("No convergence data found")

    # Determine number of panels
    has_mu = len(mu_history) > 0
    has_t = len(t_history) > 0
    n_panels = 1 + int(has_mu) + int(has_t)

    with popout_style():
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        # Panel 1: Frequency convergence
        ax = axes[0]
        ax.semilogy(iterations, max_deltas, "o-", color="#2196F3",
                     markersize=3, label="max Δ(freq)")
        if mean_deltas:
            ax.semilogy(iterations[:len(mean_deltas)], mean_deltas, "s--",
                        color="#FF9800", markersize=3, label="mean Δ(freq)")
        ax.axhline(1e-4, color="gray", linestyle=":", alpha=0.5,
                   label="threshold (1e-4)")
        # Annotate final value
        final_val = max_deltas[-1]
        ax.annotate(
            f"{final_val:.2e}", xy=(iterations[-1], final_val),
            xytext=(5, 10), textcoords="offset points", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        )
        ax.set_xlabel("EM Iteration")
        ax.set_ylabel("Allele Frequency Change")
        ax.set_title("Frequency Convergence")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

        panel_idx = 1

        # Panel 2: mu trajectory
        if has_mu:
            ax = axes[panel_idx]
            panel_idx += 1
            mu_iters = [m[0] for m in mu_history]
            mu_vals = [m[1] for m in mu_history]
            n_anc = len(mu_vals[0]) if mu_vals else 0
            colors = ancestry_colors(n_anc)
            names = ancestry_names(n_anc, labels)
            for a in range(n_anc):
                vals = [m[a] for m in mu_vals]
                ax.plot(mu_iters, vals, "o-", color=colors[a],
                        markersize=3, label=names[a])
            ax.set_xlabel("EM Iteration")
            ax.set_ylabel("Ancestry Proportion (μ)")
            ax.set_title("Ancestry Proportions")
            ax.set_ylim(0, None)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.2)

        # Panel 3: T trajectory
        if has_t:
            ax = axes[panel_idx]
            t_iters = [t[0] for t in t_history]
            t_vals = [t[1] for t in t_history]
            ax.plot(t_iters, t_vals, "o-", color="#4CAF50", markersize=4)
            ax.set_xlabel("EM Iteration")
            ax.set_ylabel("Generations Since Admixture (T)")
            ax.set_title("Admixture Time")
            ax.grid(True, alpha=0.2)

        fig.suptitle("EM Convergence Diagnostics", fontsize=13, fontweight="bold")
        fig.tight_layout()
    return fig
