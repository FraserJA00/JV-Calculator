import os
import numpy as np
from typing import Dict, List, Tuple

def sweep_is_reverse(V, J, eps: float = 1e-6) -> bool:
    """Voc-aware would be best, but heuristic: |V| moving toward 0 ⇒ reverse."""
    V = np.asarray(V, float)
    dAbs = np.diff(np.abs(V))
    dAbs[np.abs(dAbs) < eps] = 0.0
    return np.nanmedian(dAbs) < 0

def choose_label_index(sweeps: List[Tuple[np.ndarray, np.ndarray]], mode="auto") -> int | None:
    if not sweeps:
        return None
    try:
        if mode not in ("auto", "reverse", "forward", "none"):
            idx = int(mode)
            return max(0, min(len(sweeps) - 1, idx))
    except Exception:
        pass
    if mode == "none":
        return None
    if len(sweeps) == 1:
        return 0
    if mode in ("auto", "reverse"):
        for i, (V, J) in enumerate(sweeps):
            if sweep_is_reverse(V, J):
                return i
        return len(sweeps) - 1
    if mode == "forward":
        for i, (V, J) in enumerate(sweeps):
            if not sweep_is_reverse(V, J):
                return i
        return 0
    return 0


def apply_plot_style(ax, *, ticks_in=True, border=1.5, tick_len=5, tick_w=1.5,
                     xlabel="Voltage (V)", ylabel="Current Density (mA/cm²)",
                     label_fs=16, legend_fs=14):
    ax.tick_params(axis="both",
                   direction="in" if ticks_in else "out",
                   top=True, right=True,
                   length=tick_len, width=tick_w)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_linewidth(border)
    ax.set_xlabel(xlabel, fontsize=label_fs, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=12)
    ax.xaxis.set_tick_params(labelsize=10, pad=8)
    ax.yaxis.set_tick_params(labelsize=10, pad=8)

def draw_plot(ax, datasets: List[Dict], default_xlim, default_ylim,
              style_opts: Dict, preserve_view: bool):
    if preserve_view:
        prev_xlim = ax.get_xlim(); prev_ylim = ax.get_ylim()

    ax.clear()

    for fi, ds in enumerate(datasets):
        base_label = ds.get("display_name") or os.path.splitext(os.path.basename(ds["path"]))[0]
        color = ds.get("color") or _default_palette()[fi % len(_default_palette())]
        label_idx = choose_label_index(ds["sweeps"], ds.get("label_sweep", "auto"))

        for si, (V, J) in enumerate(ds["sweeps"]):
            me = max(1, len(V) // 30)
            is_rev = sweep_is_reverse(V, J)
            linestyle = "-" if is_rev else "--"
            line_label = base_label if (label_idx is not None and si == label_idx) else "_nolegend_"
            kwargs = dict(label=line_label, linestyle=linestyle, color=color)
            if is_rev:
                kwargs.update(marker="o", markevery=me)
            ax.plot(V, J, **kwargs)

    ax.axhline(0, lw=1.5, color="black")
    ax.legend(frameon=False, loc="best", fontsize=style_opts.get("legend_fs", 14))

    apply_plot_style(ax, **style_opts)

    if preserve_view:
        x0, x1 = prev_xlim; y0, y1 = prev_ylim
        if x0 != x1 and y0 != y1:
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        else:
            ax.set_xlim(*default_xlim); ax.set_ylim(*default_ylim)
    else:
        ax.set_xlim(*default_xlim); ax.set_ylim(*default_ylim)

def _default_palette():
    return ["#d62728", "#1f1f1f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
