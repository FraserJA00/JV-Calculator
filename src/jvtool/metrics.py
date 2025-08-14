import numpy as np
from typing import Optional, Tuple, Dict

def j_at_v(v: float, V: np.ndarray, J: np.ndarray) -> float:
    order = np.argsort(V)
    return float(np.interp(v, V[order], J[order]))

def voc(V: np.ndarray, J: np.ndarray) -> Optional[float]:
    s = np.sign(J)
    changes = np.where(np.diff(s) != 0)[0]
    if len(changes) == 0:
        return None
    i = int(changes[0])
    x0, x1 = V[i], V[i+1]
    y0, y1 = J[i], J[i+1]
    if y1 == y0:
        return float((x0 + x1) / 2.0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))

def _interp_endpoint(V: np.ndarray, J: np.ndarray, x: float):
    order = np.argsort(V)
    return float(x), float(np.interp(x, V[order], J[order]))

def mpp_between_0_and_voc(V: np.ndarray, J: np.ndarray,
                          Voc: Optional[float] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if Voc is None:
        Voc = voc(V, J)
    if Voc is None:
        return None, None, None

    V = np.asarray(V, float)
    J = np.asarray(J, float)

    vmin = min(0.0, Voc)
    vmax = max(0.0, Voc)
    mask = (V >= vmin) & (V <= vmax)
    V_sel = V[mask]
    J_sel = J[mask]

    ptsV, ptsJ = [], []
    if V_sel.size:
        ptsV.extend([V_sel[0], V_sel[-1]])
        ptsJ.extend([J_sel[0], J_sel[-1]])
    for xv in (0.0, Voc):
        x, y = _interp_endpoint(V, J, xv)
        ptsV.append(x); ptsJ.append(y)

    V_all = np.concatenate([V_sel, np.array(ptsV, dtype=float)]) if V_sel.size else np.array(ptsV, dtype=float)
    J_all = np.concatenate([J_sel, np.array(ptsJ, dtype=float)]) if V_sel.size else np.array(ptsJ, dtype=float)

    if V_all.size == 0:
        return None, None, None

    P = V_all * J_all
    idx = int(np.argmax(np.abs(P)))
    return float(V_all[idx]), float(J_all[idx]), float(P[idx])

def compute_metrics_for(V: np.ndarray, J: np.ndarray) -> Dict[str, float]:
    V = np.asarray(V, float); J = np.asarray(J, float)
    Jsc = j_at_v(0.0, V, J)
    Voc_ = voc(V, J)
    Vmp = Jmp = Pmp = None
    if Voc_ is not None:
        Vmp, Jmp, Pmp = mpp_between_0_and_voc(V, J, Voc_)
    if (Voc_ is not None and Jsc != 0 and Pmp is not None):
        FF = abs(Pmp) / abs(Voc_ * Jsc)
        PCE = abs(Pmp)
    else:
        FF = np.nan; PCE = np.nan

    return {
        "Jsc": float(Jsc),
        "Voc": float(Voc_) if Voc_ is not None else np.nan,
        "Vmp": float(Vmp) if Vmp is not None else np.nan,
        "Jmp": float(Jmp) if Jmp is not None else np.nan,
        "Pmp": float(Pmp) if Pmp is not None else np.nan,
        "FF": float(FF),
        "PCE": float(PCE),
    }