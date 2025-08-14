import os
import numpy as np
import pandas as pd

def read_table(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.to_numpy()

def find_voltage_change(V: np.ndarray, eps: float = 1e-6, min_len: int = 10):
    V = np.asarray(V, float)
    if V.size < 2:
        return []
    dV = np.diff(V)
    dV[np.abs(dV) < eps] = 0.0
    signs = np.sign(dV)

    segs = []
    start = 0
    cur = signs[0]
    for i, s in enumerate(signs[1:], start=1):
        if s != cur:
            end = i
            if (end - start + 1) >= min_len:
                segs.append((start, end + 1, int(np.sign(np.nanmedian(signs[start:end+1])))))
            start = i
            cur = s
    end = len(signs)
    if (end - start + 1) >= min_len:
        segs.append((start, end + 1, int(np.sign(np.nanmedian(signs[start:end+1])))))
    return segs

def sweeps_from_array(arr: np.ndarray):
        r, c = arr.shape
        summary = f"{r} rows x {c} columns"

        if c == 4:
            V1, J1 = arr[:, 0], arr[:, 1]
            V2, J2 = arr[:, 2], arr[:, 3]
            sweeps = [(V1, J1), (V2, J2)]
            summary += " | Detected 2 sweeps: (0,1),(2,3)"
            return sweeps, summary

        if c == 2:
            V, J = arr[:, 0], arr[:, 1]
            min_len = max(10, r // 50)
            segs = find_voltage_change(V, min_len=min_len)
            if len(segs) >= 2:
                (s0, e0, sg0), (s1, e1, sg1) = segs[0], segs[1], 
                sweeps = [(V[s0:e0], J[s0:e0]), (V[s1:e1], J[s1:e1])]
                summary += f" | 2-col concatenated: [{s0}:{e0}] (sign {sg0}), [{s1}:{e1}] (sign {sg1})"
            else:
                sweeps = [(V, J)]
                summary += " | 2-col single sweep"
            return sweeps, summary

        raise ValueError(f"Unsupported column count ({c}). Need 2 or 4.")