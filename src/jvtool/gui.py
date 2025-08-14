# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 13:21:57 2025

@author: Fraser Angus (PGR)
"""

import tkinter as tk
import os
from tkinter import ttk, filedialog, messagebox, colorchooser
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class JVApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JV Data Processing Tool")
        self.geometry("1300x650")

        self.datasets = []                    
        self.default_xlim = (0.0, 1.2)
        self.default_ylim = (-15.0, 27.0)
        self.axis_ticks_inwards = True
        self.graph_border_thickness = 1.5
        self.tick_length = 5.0
        self.tick_width = 1.5

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text="Open File(s) — Plot & Compute", command=self.on_open).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear", command=self.on_clear).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Legend & Colors", command=lambda: self.left_tabs.select(self.tab_legend)).pack(side=tk.LEFT, padx=6)
        self.bind_all("<Control-s>", lambda e: self.on_save_png())

        axes = ttk.Frame(self)
        axes.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(axes, text="x min").pack(side=tk.LEFT)
        self.xmin_var = tk.StringVar(value=str(self.default_xlim[0]))
        ttk.Entry(axes, textvariable=self.xmin_var, width=8).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(axes, text="x max").pack(side=tk.LEFT)
        self.xmax_var = tk.StringVar(value=str(self.default_xlim[1]))
        ttk.Entry(axes, textvariable=self.xmax_var, width=8).pack(side=tk.LEFT, padx=(2, 20))

        ttk.Label(axes, text="y min").pack(side=tk.LEFT)
        self.ymin_var = tk.StringVar(value=str(self.default_ylim[0]))
        ttk.Entry(axes, textvariable=self.ymin_var, width=8).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(axes, text="y max").pack(side=tk.LEFT)
        self.ymax_var = tk.StringVar(value=str(self.default_ylim[1]))
        ttk.Entry(axes, textvariable=self.ymax_var, width=8).pack(side=tk.LEFT, padx=(2, 20))

        ttk.Button(axes, text="Apply axes", command=self.on_apply_axes).pack(side=tk.LEFT, padx=4)
        ttk.Button(axes, text="Reset", command=self.on_reset_axes).pack(side=tk.LEFT, padx=4)

        content = ttk.Frame(self)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.left_tabs = ttk.Notebook(content, width=480)
        self.left_tabs.pack(side=tk.LEFT, fill=tk.Y)
        
        self.tab_params = ttk.Frame(self.left_tabs)
        self.left_tabs.add(self.tab_params, text="Parameters")
        
        ttk.Label(self.tab_params, text="Device Parameters", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(6,2))
        self.metrics_text = tk.Text(self.tab_params, width=34, height=28)
        self.metrics_text.pack(fill=tk.Y, expand=False, padx=6, pady=(0,6))
        self.metrics_text.tag_configure("bold", font=("Segoe UI", 10, "bold"))
        
        self.tab_legend = ttk.Frame(self.left_tabs)
        self.left_tabs.add(self.tab_legend, text="Legend & Colors")
        
        self.legend_canvas = tk.Canvas(self.tab_legend, bd=0, highlightthickness=0)
        self.legend_scroll = ttk.Scrollbar(self.tab_legend, orient="vertical", command=self.legend_canvas.yview)
        self.legend_inner = ttk.Frame(self.legend_canvas)
        self.legend_xscroll = ttk.Scrollbar(self.tab_legend, orient="horizontal",
                                            command=self.legend_canvas.xview)
        self.legend_xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.legend_canvas.configure(xscrollcommand=self.legend_xscroll.set)
        
        legend_btns = ttk.Frame(self.tab_legend)
        legend_btns.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        
        self.legend_inner.bind(
            "<Configure>",
            lambda e: self.legend_canvas.configure(scrollregion=self.legend_canvas.bbox("all"))
        )
        self.legend_canvas.create_window((0, 0), window=self.legend_inner, anchor="nw")
        self.legend_canvas.configure(yscrollcommand=self.legend_scroll.set)
        
        self.legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.legend_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        legend_btns = ttk.Frame(self.tab_legend)
        legend_btns.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        ttk.Button(legend_btns, text="Apply", command=self._apply_legend_changes).pack(side=tk.RIGHT, padx=4)
        ttk.Button(legend_btns, text="Revert", command=self._rebuild_legend_panel).pack(side=tk.RIGHT, padx=4)

        plot_frame = ttk.Frame(content)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.fig = Figure(figsize=(7, 5), dpi=150, constrained_layout=True)
        self.ax = self.fig.add_subplot(111)

        self._apply_plot_style()
        self.ax.set_xlim(*self.default_xlim)
        self.ax.set_ylim(*self.default_ylim)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.draw_idle()
        
 
    def on_open(self):
        paths = filedialog.askopenfilenames(
            title="Open JV data (you can select multiple)",
            initialdir=getattr(self, "last_dir", os.path.expanduser("~")),
            filetypes=[
                ("JV text files", ("*.txt", "*.dat", "*.csv", "*.tsv")),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
    
        self.last_dir = os.path.dirname(paths[0])
        self.datasets = []
    
        loaded = 0
        errors = []
    
        for path in paths:
            try:
                df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                arr = df.to_numpy()
    
                sweeps, summary = self._sweeps_from_array(arr)
                base = os.path.basename(path)
                short = os.path.splitext(base)[0]
                default_color = self._default_color_for(len(self.datasets))
    
                self.datasets.append({
                    "path": path,
                    "sweeps": sweeps,
                    "summary": summary,
                    "display_name": short,
                    "label_sweep": "auto",
                    "color": default_color,
                })
    

                self.data = arr
                self.sweep_arrays = sweeps
    
                loaded += 1
    
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
    
        if loaded:
            self._rebuild_legend_panel()
            self._draw_plot(apply_defaults=True)
            self.on_compute_metrics(refresh_plot=False)
    
        if errors:
            messagebox.showwarning("Some files skipped", "\n".join(errors))
                                 
    """
    Need a way to identify if a file contains all data in 2 columns (i.e. is conctenated)
    Split by vhange in sign of voltage values
    """
    
    def _find_voltage_change(self, V, eps=1e-6, min_len=10):
        V = np.asarray(V, dtype=float)
        if V.size <3:
            return []
        dV = np.diff(V)
        dV[np.abs(dV) < eps] = 0.0
        sign = np.sign(dV)
        
        segments = []
        k = 0
        
        while k < sign.size and sign[k] == 0:
            k += 1
        if k == sign.size:
            return [(0, V.size, 0)]
        
        current_sign = sign[k]
        i0 = 0
        
        for i in range(k + 1, sign.size):
            if sign[i] == 0:
                continue
            if sign[i] != current_sign:
                if (i + 1 - i0) >= min_len:
                    segments.append((i0, i + 1, int(current_sign)))
                i0 = i + 1
                current_sign = sign[i]
                
        if (V.size - i0) >= min_len:
            segments.append((i0, V.size, int(current_sign)))
        
        if not segments:
            segments = [(0, V.size, int(current_sign))]
        return segments
    
    def _draw_plot(self, apply_defaults=True):
    
        preserve_view = not apply_defaults and hasattr(self, "ax")
        if preserve_view:
            prev_xlim = self.ax.get_xlim()
            prev_ylim = self.ax.get_ylim()
    
        self.ax.clear()
    
        colors  = ["#d62728", "#1f1f1f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        markers = ["o", "^", "s", "d", "v", "P"]
    
        for fi, ds in enumerate(self.datasets):
            base_label = ds.get("display_name") or os.path.splitext(os.path.basename(ds["path"]))[0]
            color      = ds.get("color") or colors[fi % len(colors)]
            label_idx  = self._choose_label_index(ds["sweeps"], ds.get("label_sweep", "auto"))
    
            for si, (V, J) in enumerate(ds["sweeps"]):
                me = max(1, len(V) // 30)
                line_label = base_label if (label_idx is not None and si == label_idx) else "_nolegend_"
    
                is_rev    = self._sweep_is_reverse(V, J) 
                linestyle = "-" if is_rev else "--"
                kwargs = dict(label=line_label, linestyle=linestyle, color=color)
                if is_rev:
                    kwargs.update(marker=markers[(fi + si) % len(markers)], markevery=me)
    
                self.ax.plot(V, J, **kwargs)
    
        self.ax.axhline(0, lw=1.5, color="black")
        self.ax.set_xlabel("Voltage (V)", fontsize = 14)
        self.ax.set_ylabel("Current Density (mA/cm²)", fontsize = 14)
        self.ax.legend(frameon=False, loc="best", fontsize = 12)
        self._apply_plot_style()
    
        if preserve_view:
            x0, x1 = prev_xlim
            y0, y1 = prev_ylim
            if x0 != x1 and y0 != y1:
                self.ax.set_xlim(x0, x1)
                self.ax.set_ylim(y0, y1)
            else:
                self.ax.set_xlim(*self.default_xlim)
                self.ax.set_ylim(*self.default_ylim)
        else:
            self.ax.set_xlim(*self.default_xlim)
            self.ax.set_ylim(*self.default_ylim)
    
        self.canvas.draw_idle()
              
    def _find_Jsc(self, V, J, v=0.0):
        order = np.argsort(V)
        V_sorted = V[order]
        J_sorted = J[order]
        return float(np.interp(v, V_sorted, J_sorted))
    
    def _find_Voc(self, V, J):
        s = np.sign(J)
        changes = np.where(np.diff(s) != 0)[0]
        if len(changes) == 0:
            return None
        i = changes[0]
        x0, x1 = V[i], V[i+1]
        y0, y1 = J[i], J[i+1]
        if y1 == y0:
            return float(0.5 * (x0 + x1))
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    
    def _calc_mpp_with_bounds(self, V, J, Voc):
        """
        This should calulate MPP from any dataset, using bounds
        so that the current comes from the photogenerated current
        (i.e. between Voc and Jsc)
        """
        v = np.asarray(V, dtype= float)
        j = np.asarray(J, dtype=float)
        
        lo, hi = (Voc, 0.0) if Voc < 0 else (0.0, Voc)
        mask = (v >= lo) & (v <= hi)
        Vw = v[mask]
        Jw = j[mask]
        order = np.argsort(v)
        v_sorted, j_sorted = v[order], j[order]
        
        def _add_endpoint(Vw, Jw, x):
            if Vw.size and np.any(np.isclose(Vw, x, atol=1e-12)):
                return Vw, Jw
            if (x >= v_sorted[0]) and (x <= v_sorted[-1]):
                Jx = np.interp(x, v_sorted, j_sorted)
                return np.append(Vw, x), np.append(Jw, Jx)
            return Vw, Jw
        
        Vw, Jw = _add_endpoint(Vw, Jw, 0.0)
        Vw, Jw = _add_endpoint(Vw, Jw, Voc)
        
        if Vw.size == 0:
            return None, None, None
        
        P = Vw * Jw  
        idx = np.argmax(np.abs(P))
        return float(Vw[idx]), float(Jw[idx]), float(P[idx])
    
    def _compute_metrics_for(self, V, J):
        Jsc = self._find_Jsc(V, J, v=0.0)
        Voc = self._find_Voc(V, J)
        if Voc is None:
            return {"Jsc": Jsc, "Voc": None, "Vmp": None, "Jmp": None,
                    "Pmp": None, "FF": np.nan, "PCE": np.nan}
        
        Vmp, Jmp, Pmp = self._calc_mpp_with_bounds(V, J, Voc)
        if Vmp is None:
            return {"Jsc": Jsc, "Voc": Voc, "Vmp": None, "Jmp": None,
                    "Pmp": None, "FF": np.nan, "PCE": np.nan}
        
        FF = (abs(Pmp)) / (abs(Voc*Jsc))
        PCE = (Jsc * Voc * FF)
        
        return {"Jsc": Jsc, "Voc": Voc, "Vmp": Vmp, "Jmp": Jmp,
                "Pmp": Pmp, "FF": FF, "PCE": PCE}
    
    def on_compute_metrics(self, refresh_plot=True):
        if getattr(self, "sweep_arrays", None) is None or len(self.sweep_arrays) == 0:
            messagebox.showinfo("Info", "Load a file first, then click Plot.")
            return
        
        if not self.datasets:
            messagebox.showinfo("Info", "Load one or more files first.")
            return
        
        if refresh_plot:
            self._draw_plot()
        self.canvas.draw_idle()
        self.metrics_text.delete("1.0", tk.END)
        
        for fi, ds in enumerate(self.datasets):
            name = ds.get("display_name") or os.path.splitext(os.path.basename(ds["path"]))[0]
            self.metrics_text.insert(tk.END, f"{name}\n", "bold")
            for si, (V, J) in enumerate(ds["sweeps"]):
                m = self._compute_metrics_for(np.asarray(V), np.asarray(J))
                self.metrics_text.insert(tk.END, f"  Sweep {si+1}\n")
                self.metrics_text.insert(tk.END, f"    Jsc: {m['Jsc']:.3f} mA/cm²\n")
                self.metrics_text.insert(tk.END, f"    Voc: {m['Voc']:.3f} V\n" if m['Voc'] is not None else "    Voc: —\n")
                self.metrics_text.insert(tk.END, f"    FF : {m['FF']:.3f}\n")
                self.metrics_text.insert(tk.END, f"    PCE: {m['PCE']:.2f} %\n")
        
            self.metrics_text.insert(tk.END, "\n")
        
        self.canvas.draw_idle()         
        
    def _parse_float(self, s):
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def on_apply_axes(self):
        xmin = self._parse_float(self.xmin_var.get())
        xmax = self._parse_float(self.xmax_var.get())
        ymin = self._parse_float(self.ymin_var.get())
        ymax = self._parse_float(self.ymax_var.get())
        
        cur_xmin, cur_xmax = self.ax.get_xlim()
        cur_ymin, cur_ymax = self.ax.get_ylim()
        
        new_xmin = cur_xmin if xmin is None else xmin
        new_xmax = cur_xmax if xmax is None else xmax
        new_ymin = cur_ymin if ymin is None else ymin
        new_ymax = cur_ymax if ymax is None else ymax
        
        if new_xmin > new_xmax:
            new_xmin, new_xmax = new_xmax, new_xmin
        if new_ymin > new_ymax:
            new_ymin, new_ymax = new_ymax, new_ymin
            
        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)
        self.default_xlim = self.ax.get_xlim()
        self.default_ylim = self.ax.get_ylim()
        self.canvas.draw_idle()
        
    def _apply_plot_style(self):
        ax = self.ax
        
        ticks_in = getattr(self, "axis_ticks_inwards", True)
        border = getattr(self, "graph_border_thickness", 1.5)
        tick_len = getattr(self, "tick_length", 5)
        tick_w = getattr(self, "tick_width", 1.5)
        
        ax.tick_params(
            axis="both",
            direction="in" if ticks_in else "out",
            top=True,
            right=True,
            length=tick_len,
            width=tick_w,
            )
        
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_linewidth(border)
            
        ax.xaxis.set_tick_params(labelsize=10, pad=8)
        ax.yaxis.set_tick_params(labelsize=10, pad=8)
        
    def on_reset_axes(self):
        if not getattr(self, "sweep_arrays", None):
            self.ax.relim()
            self.ax.autoscale()
            self.canvas.draw_idle()
            return
        if self.datasets:
            V_all = np.concatenate([np.asarray(V) for ds in self.datasets for (V, J) in ds["sweeps"]])
            J_all = np.concatenate([np.asarray(J) for ds in self.datasets for (V, J) in ds["sweeps"]])
        elif getattr(self, "sweep_arrays", None):
            V_all = np.concatenate([np.asarray(V) for (V, J) in self.sweep_arrays])
            J_all = np.concatenate([np.asarray(J) for (V, J) in self.sweep_arrays])
                
        def pad(lo, hi, frac=0.05):
            if lo == hi:
                span = 1.0 if hi == 0 else abs(hi) * frac
                return lo - span, hi + span
            span = hi - lo
            return lo - frac * span, hi + frac * span
    
        xmin, xmax = pad(float(V_all.min()), float(V_all.max()), 0.05)
        ymin, ymax = pad(float(J_all.min()), float(J_all.max()), 0.05)
    
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.canvas.draw_idle()
        
    def on_clear(self):
        self.datasets.clear()
        self.metrics_text.delete("1.0", tk.END)
        self.ax.clear()
        self.ax.set_xlabel("Voltage (V)"); self.ax.set_ylabel("Current Density (mA/cm²)")
        self.ax.grid(True, which="both", alpha=0.25)
        self.ax.axhline(0, lw=1, color="black", alpha=0.8); self.ax.axvline(0, lw=1, color="0.8")
        self.ax.set_xlim(*self.default_xlim); self.ax.set_ylim(*self.default_ylim)
        self.canvas.draw_idle()
        
        
    def _sweeps_from_array(self, arr):
        r, c = arr.shape
        summary = f"{r} rows x {c} columns"
    
        if c == 2:
            V = arr[:, 0]
            J = arr[:, 1]
            min_len = max(10, r // 50)
            segs = self._find_voltage_change(V, eps=1e-6, min_len=min_len)
    
            if segs and len(segs) >= 2:
                (s0, e0, sign0) = segs[0]
                (s1, e1, sign1) = segs[1]
                sweeps = [(V[s0:e0], J[s0:e0]), (V[s1:e1], J[s1:e1])]
                summary += f" | 2 column concatenated: segments [{s0}:{e0}] (sign {sign0}), [{s1}:{e1}] (sign {sign1})"
            else:
                sweeps = [(V, J)]
                summary += " | 2 column single sweep"
    
        elif c == 4:
            V1, J1 = arr[:, 0], arr[:, 1]
            V2, J2 = arr[:, 2], arr[:, 3]
            sweeps = [(V1, J1), (V2, J2)]
            summary += " | Detected 2 sweeps: (0, 1), (2, 3)"
    
        else:
            raise ValueError(f"Unsupported column count ({c}). Need 2 or 4.")
    
        return sweeps, summary
        
    def on_save_png(self):
        default = "jv_plot.png"
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            initialfile=default,
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All files", "*.*")]
        )
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight")
        
    """
    Now to define what sweep is the reverse sweep, this has been avoided due to regular and inverted devices
    lets try
    """
    
    def _sweep_is_reverse(self, V, J, eps=1e-6):
        V = np.asarray(V, float)
        J = np.asarray(J, float)
        
        dV = np.diff(V)
        dV[np.abs(dV) < eps] = 0.0
        med_dV = np.nanmedian(dV)
        
        Voc = self._find_Voc(V, J)
        if Voc is not None and Voc != 0:
            return med_dV * np.sign(Voc) < 0
        
        dAbs = np.diff(np.abs(V))
        dAbs[np.abs(dAbs) < eps] = 0.0
        return np.nanmedian(dAbs) < 0
    
    def _choose_label_index(self, sweeps, mode="auto"):
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
                if self._sweep_is_reverse(V, J):
                    return i
            return len(sweeps) - 1
    
        if mode == "forward":
            for i, (V, J) in enumerate(sweeps):
                if not self._sweep_is_reverse(V, J):
                    return i
            return 0
        return 0
    
    def _palette(self):
        return["#d62728", "#1f1f1f", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

    def _default_color_for(self, idx):
        p = self._palette()
        return p[idx % len(p)]     
    
    def _rebuild_legend_panel(self):
        for child in self.legend_inner.winfo_children():
            child.destroy()
    
        self._legend_rows = []
    
        hdr = ttk.Frame(self.legend_inner)
        hdr.pack(fill=tk.X, padx=6, pady=(8, 4))
        ttk.Label(hdr, text="Dataset").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(hdr, text="Legend name").grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Label(hdr, text="Color").grid(row=0, column=2, sticky="w", padx=(0, 10))
        ttk.Label(hdr, text="Label sweep").grid(row=0, column=3, sticky="w")
    
        options = ["Auto", "Reverse", "Forward", "Sweep 1", "Sweep 2", "None"]
    
        for ds in self.datasets:
            row = ttk.Frame(self.legend_inner)
            row.pack(fill=tk.X, padx=6, pady=4)
    
            base = os.path.basename(ds["path"])
            cur_name  = ds.get("display_name") or os.path.splitext(base)[0]
            cur_color = ds.get("color") or self._default_color_for(0)
            mode      = ds.get("label_sweep", "auto")
    
            name_var  = tk.StringVar(value=cur_name)
            color_var = tk.StringVar(value=cur_color)
            mode_var  = tk.StringVar(value=(
                "Auto" if mode == "auto" else
                "Reverse" if mode == "reverse" else
                "Forward" if mode == "forward" else
                "Sweep 1" if str(mode) == "0" else
                "Sweep 2" if str(mode) == "1" else
                "None"
            ))
    
            ttk.Label(row, text=base, width=22).grid(row=0, column=0, sticky="w")
    
            name_entry = ttk.Entry(row, textvariable=name_var, width=20)
            name_entry.grid(row=0, column=1, sticky="w", padx=(0, 10))
            def apply_name(ev=None, d=ds, var=name_var):
                new_name = var.get().strip()
                if new_name:
                    d["display_name"] = new_name
                    self._draw_plot(apply_defaults=False)
                    self.on_compute_metrics(refresh_plot=False)
            name_entry.bind("<Return>", apply_name)
    
            color_frame = ttk.Frame(row)
            color_frame.grid(row=0, column=2, sticky="w", padx=(0, 10))
    
            swatch = tk.Label(color_frame, text="   ", bg=cur_color, relief="groove", width=3)
            swatch.pack(side=tk.LEFT, padx=(0, 5))
    
            def pick_color(d=ds, var=color_var, sw=swatch):
                rgb, hx = colorchooser.askcolor(color=var.get(), parent=self)
                if hx:
                    var.set(hx)
                    sw.config(bg=hx)
                    d["color"] = hx        
                    self._draw_plot(apply_defaults=False)
    
            ttk.Button(color_frame, text="Pick…", command=pick_color).pack(side=tk.LEFT)
    
            cb = ttk.Combobox(row, values=options, state="readonly",
                              textvariable=mode_var, width=10)
            cb.grid(row=0, column=3, sticky="w", padx=(10, 0))
    
            def set_mode(ev=None, d=ds, var=mode_var):
                sel = var.get()
                d["label_sweep"] = (
                    "auto"    if sel == "Auto"    else
                    "reverse" if sel == "Reverse" else
                    "forward" if sel == "Forward" else
                    0         if sel == "Sweep 1" else
                    1         if sel == "Sweep 2" else
                    "none"
                )
                self._draw_plot(apply_defaults=False)
    
            cb.bind("<<ComboboxSelected>>", set_mode)
    
            self._legend_rows.append((ds, name_var, color_var, mode_var))
            # self._draw_plot(apply_defaults=False)   

            
    def _apply_legend_changes(self):
        if not hasattr(self, "_legend_rows"):
            return
    
        for ds, name_var, color_var, mode_var in self._legend_rows:
            new_name = name_var.get().strip()
            new_color = color_var.get().strip()
            if new_name:
                ds["display_name"] = new_name
            if new_color:
                ds["color"] = new_color
    
            sel = mode_var.get()
            ds["label_sweep"] = (
                "auto" if sel == "Auto" else
                "reverse" if sel == "Reverse" else
                "forward" if sel == "Forward" else
                0 if sel == "Sweep 1" else
                1 if sel == "Sweep 2" else
                "none"
            )
    
        self._draw_plot(apply_defaults=False)
        self.on_compute_metrics(refresh_plot=False)


def run_app():
    app = JVApp()
    app.mainloop()
        