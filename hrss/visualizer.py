# hrss/visualizer.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import h5py  # only used for context HDF lookup (in pyproject dependencies)
except Exception:
    h5py = None  # handled gracefully below

__all__ = [
    "plot_polar_ppi",
    "plot_polar_strip",
    "plot_polar_animation",
    "context_block_for_window",
    "print_context_summary",
    "plot_context_timeseries",
]

# ----------------------------- small helpers -----------------------------

import numpy as _np
import numpy.ma as _ma
import matplotlib.pyplot as _plt
import pandas as _pd

def _robust_limits(vals_like, q=(1, 99)):
    vals = _np.asarray(vals_like, dtype=float)
    vals = vals[_np.isfinite(vals)]
    if vals.size == 0:
        return (-30.0, 70.0)
    lo = float(_np.nanpercentile(vals, q[0]))
    hi = float(_np.nanpercentile(vals, q[1]))
    if not _np.isfinite(lo) or not _np.isfinite(hi) or lo >= hi:
        lo, hi = float(_np.nanmin(vals)), float(_np.nanmax(vals))
    return (lo, hi)

def _xy_from_az_rng(az_deg_1d, rng_m_1d):
    """Return (Xc, Yc) gate-center coords shaped (H, W) from 1-D azimuth/range."""
    azr = _np.deg2rad(az_deg_1d)[:, None]  # (H, 1)
    R   = _np.asarray(rng_m_1d, dtype=float)[None, :]  # (1, W)
    Xc  = R * _np.sin(azr)   # east
    Yc  = R * _np.cos(azr)   # north
    return Xc, Yc

def plot_polar_ppi(
    frame_hw_c: _np.ndarray,
    geometry: dict,
    *,
    t: int = 0,
    ch: int = 0,
    units: str = "dBZ",
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: _plt.Axes | None = None,
    cmap: str = "turbo",
    rings: bool = True,
    percentiles: tuple[float, float] = (5, 99),
):
    """
    Polar fallback PPI: draw gate centers in Cartesian (x east, y north) with pcolormesh.

    frame_hw_c: one time slice (H, W, C)
    geometry: must have azimuth_deg (T,H) and range_m (T,W); we index with t
    """
    H, W = frame_hw_c.shape[:2]
    az = _np.asarray(geometry["azimuth_deg"][t], _np.float32)[:H]  # (H,)
    rg = _np.asarray(geometry["range_m"][t],     _np.float32)[:W]  # (W,)

    # sort rays by wrapped azimuth so the mesh is orderly
    order = _np.argsort(_np.mod(az, 360.0))
    az_sorted = az[order]
    img = _ma.masked_invalid(frame_hw_c[order, :, ch])

    # color limits (robust) if not provided
    if vmin is None or vmax is None:
        vmin, vmax = _robust_limits(img.compressed(), q=percentiles)

    # gate-center coordinates
    Xc, Yc = _xy_from_az_rng(az_sorted, rg)

    created = False
    if ax is None:
        fig, ax = _plt.subplots(figsize=(6, 6))
        created = True

    pm = ax.pcolormesh(Xc, Yc, img, shading="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_aspect("equal", adjustable="box")

    # limits from robust radius
    r_xy = float(_np.nanpercentile(_np.hypot(Xc, Yc), 99.5))
    if not _np.isfinite(r_xy) or r_xy <= 0:
        r_xy = float(_np.nanmax(rg))
    ax.set_xlim(-r_xy, r_xy); ax.set_ylim(-r_xy, r_xy)

    # optional range rings
    if rings and _np.isfinite(r_xy):
        for rr in _np.linspace(r_xy * 0.25, r_xy, 4):
            ax.add_artist(_plt.Circle((0, 0), rr, fill=False, lw=0.6, alpha=0.25))

    # title + colorbar
    if title:
        ax.set_title(title, fontsize=11)
    cb = _plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(units)

    ax.set_xlabel("x (m, east)"); ax.set_ylabel("y (m, north)")
    if created:
        _plt.tight_layout()
    return ax

def plot_polar_strip(
    window_thwc: _np.ndarray,
    geometry: dict,
    *,
    ch: int = 0,
    t_indices: Sequence[int] | None = None,
    units: str = "dBZ",
    suptitle: str | None = None,
    max_cols: int = 6,
    figsize_per: tuple[float, float] = (3.6, 3.6),
    cmap: str = "turbo",
    percentiles: tuple[float, float] = (5, 99),
):
    """
    Row of polar PPI frames (x/y pcolormesh). Consistent color scale across the row.
    """
    T = window_thwc.shape[0]
    if t_indices is None:
        t_indices = list(range(T))
    t_indices = list(t_indices)[:max_cols]
    n = len(t_indices)

    # compute robust color scale over selected frames
    vals = []
    for ti in t_indices:
        vals.append(_np.asarray(window_thwc[ti, :, :, ch], dtype=float))
    vmin, vmax = _robust_limits(_np.stack(vals, axis=0), q=percentiles)

    fig, axes = _plt.subplots(1, n, figsize=(figsize_per[0]*n, figsize_per[1]), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, ti in zip(axes, t_indices):
        title = f"t={ti}, ch={ch}"
        plot_polar_ppi(window_thwc[ti], geometry, t=ti, ch=ch, units=units,
                       title=title, vmin=vmin, vmax=vmax, ax=ax, cmap=cmap)

    if suptitle:
        fig.suptitle(suptitle, y=1.03, fontsize=12)
    return fig, axes

def plot_polar_animation(
    window_thwc: _np.ndarray,
    geometry: dict,
    *,
    ch: int = 0,
    units: str = "dBZ",
    interval_ms: int = 250,
    save_path: str | None = None,
    cmap: str = "turbo",
    percentiles: tuple[float, float] = (5, 99),
    rings: bool = True,
):
    """
    Polar PPI animation (fallback). Keeps fixed color limits and axes limits across frames.
    Replaces the QuadMesh each frame (accommodates changing ray counts / azimuths).
    """
    import matplotlib.animation as _animation

    T = int(window_thwc.shape[0])

    # --- Global color limits across the whole window (robust) ---
    vmin, vmax = _robust_limits(window_thwc[..., ch], q=percentiles)

    # --- Compute a fixed radius that works for all frames ---
    def _frame_radius(i: int) -> float:
        H, W = window_thwc.shape[1:3]
        az = _np.asarray(geometry["azimuth_deg"][i], _np.float32)[:H]
        rg = _np.asarray(geometry["range_m"][i],     _np.float32)[:W]
        if az.size == 0 or rg.size == 0:
            return _np.nan
        order = _np.argsort(_np.mod(az, 360.0))
        az_sorted = az[order]
        Xc, Yc = _xy_from_az_rng(az_sorted, rg)
        rad = _np.nanpercentile(_np.hypot(Xc, Yc), 99.5)
        return float(rad)

    r_vals = []
    for i in range(T):
        try:
            r = _frame_radius(i)
            if _np.isfinite(r):
                r_vals.append(r)
        except Exception:
            pass
    # Fallback to max range if needed
    if r_vals:
        r_lim = max(r_vals)
    else:
        try:
            r_lim = float(_np.nanmax(_np.asarray(geometry["range_m"][0], _np.float32)))
        except Exception:
            r_lim = 1.0  # last resort to avoid zero/NaN

    # --- Figure / Axes setup once ---
    fig, ax = _plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-r_lim, r_lim)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")

    # Range rings once
    ring_artists = []
    if rings and _np.isfinite(r_lim) and r_lim > 0:
        for rr in _np.linspace(r_lim * 0.25, r_lim, 4):
            ring = _plt.Circle((0, 0), rr, fill=False, lw=0.6, alpha=0.25)
            ax.add_artist(ring)
            ring_artists.append(ring)

    # --- First frame (build pm + colorbar once) ---
    def _quadmesh_for(i: int):
        H, W = window_thwc.shape[1:3]
        az = _np.asarray(geometry["azimuth_deg"][i], _np.float32)[:H]
        rg = _np.asarray(geometry["range_m"][i],     _np.float32)[:W]
        order = _np.argsort(_np.mod(az, 360.0))
        az_sorted = az[order]
        img = _ma.masked_invalid(window_thwc[i, order, :, ch])
        Xc, Yc = _xy_from_az_rng(az_sorted, rg)
        pm = ax.pcolormesh(Xc, Yc, img, shading="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        return pm

    pm_holder = [ _quadmesh_for(0) ]   # keep current QuadMesh here
    title_obj = ax.set_title(f"t=0, ch={ch}")
    cbar = fig.colorbar(pm_holder[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(units)

    # --- Per-frame updater (replace only the QuadMesh, keep limits/colorbar) ---
    def _update(i: int):
        # Remove previous QuadMesh
        if pm_holder[0] is not None:
            try:
                pm_holder[0].remove()
            except Exception:
                pass
        # Add new QuadMesh
        pm_holder[0] = _quadmesh_for(i)
        # Keep fixed axes limits
        ax.set_xlim(-r_lim, r_lim)
        ax.set_ylim(-r_lim, r_lim)
        # Update colorbar to point at the new mappable
        cbar.update_normal(pm_holder[0])
        # Update title
        title_obj.set_text(f"t={i}, ch={ch}")
        # Return artists that changed (no blitting, but harmless)
        return (pm_holder[0], title_obj)

    ani = _animation.FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=False)

    if save_path:
        ani.save(save_path, dpi=120)

    return ani

# ----------------------------- context helpers (used by notebooks) -----------------------------

def _read_context_df_from_product(meta: dict) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Find and load the context HDF associated with meta['path_in'].
    Returns (df, path) or (None, None) if not found.
    """
    prod_path = Path(meta["path_in"])
    ctx_path: Optional[Path] = None

    # Preferred: product file attr points to context
    if h5py is not None:
        try:
            with h5py.File(prod_path, "r") as h5:
                ctx_rel = h5.attrs.get("context_relpath", None)
                if isinstance(ctx_rel, (bytes, bytearray)):
                    ctx_rel = ctx_rel.decode("utf-8", errors="ignore")
                if isinstance(ctx_rel, str) and ctx_rel.strip():
                    ctx_path = (prod_path.parent / ctx_rel).resolve()
        except Exception:
            ctx_path = None

    # Fallback: glob in the same storm dir
    if ctx_path is None or not ctx_path.exists():
        cands = sorted(prod_path.parent.glob("*context*.h5"))
        ctx_path = cands[0] if cands else None

    if ctx_path is None or not ctx_path.exists():
        return None, None

    df = pd.read_hdf(ctx_path, key="context")

    # Ensure time_unix_ms exists
    if "time_unix_ms" not in df.columns:
        if "time" in df.columns:
            t = pd.to_datetime(df["time"], utc=True, errors="coerce")
            try:
                df["time_unix_ms"] = (t.astype("int64") // 1_000_000)
            except Exception:
                df["time_unix_ms"] = (t.view("int64") // 1_000_000)
        else:
            df["time_unix_ms"] = np.int64(-1)

    return df, ctx_path

def context_block_for_window(meta: dict, columns: Optional[Iterable[str]] = None) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """
    Return a filtered + ordered context DataFrame aligned to the X window [t0..t1].
    """
    df, ctx_path = _read_context_df_from_product(meta)
    if df is None:
        return None, None

    t0 = int(meta["t0_ms"]); t1 = int(meta["t1_ms"])
    block = df[(df["time_unix_ms"] >= t0) & (df["time_unix_ms"] <= t1)].copy()

    # Friendly UTC time (first column)
    block.insert(0, "time_utc",
                 pd.to_datetime(block["time_unix_ms"], unit="ms", utc=True)
                   .dt.strftime("%Y-%m-%d %H:%M:%S"))

    default_cols = [
        "time_utc", "storm_id", "radar_site",
        "latitude", "longitude",
        "u_motion", "v_motion",
        "column_max_refl",
        "10_dbz_echo_top", "20_dbz_echo_top", "30_dbz_echo_top", "40_dbz_echo_top",
        "min_lat", "max_lat", "min_lon", "max_lon",
    ]
    cols = list(columns) if columns is not None else default_cols
    present = [c for c in cols if c in block.columns]
    block = block[present]

    return block.reset_index(drop=True), ctx_path

def print_context_summary(block: Optional[pd.DataFrame]) -> None:
    """Pretty numeric summary across the window."""
    if block is None or len(block) == 0:
        print("[ctx] no rows in window.")
        return

    def smean(name): return float(pd.to_numeric(block.get(name), errors="coerce").mean())
    def smax(name):  return float(pd.to_numeric(block.get(name), errors="coerce").max())

    mu = smean("u_motion") if "u_motion" in block.columns else np.nan
    mv = smean("v_motion") if "v_motion" in block.columns else np.nan
    zcol = smax("column_max_refl") if "column_max_refl" in block.columns else np.nan
    et10 = smax("10_dbz_echo_top") if "10_dbz_echo_top" in block.columns else np.nan
    et20 = smax("20_dbz_echo_top") if "20_dbz_echo_top" in block.columns else np.nan
    et30 = smax("30_dbz_echo_top") if "30_dbz_echo_top" in block.columns else np.nan
    et40 = smax("40_dbz_echo_top") if "40_dbz_echo_top" in block.columns else np.nan

    print("— Context summary (window) —")
    if np.isfinite(mu) and np.isfinite(mv):
        spd = (mu**2 + mv**2) ** 0.5
        # meteorological direction (from which it blows): convert (u,v) to deg
        import math
        dir_deg = (270 - math.degrees(math.atan2(mv, mu))) % 360
        print(f"  Storm motion: u={mu:.2f} m/s, v={mv:.2f} m/s  |  speed={spd:.2f} m/s, dir={dir_deg:.0f}°")
    if np.isfinite(zcol):
        print(f"  Column-max reflectivity (max over window): {zcol:.1f} dBZ")
    tops = [et10, et20, et30, et40]; lbls = ["10", "20", "30", "40"]
    tops_text = ", ".join(f"{l} dBZ: {v:.1f} km" for l, v in zip(lbls, tops) if np.isfinite(v))
    if tops_text:
        print(f"  Echo tops (max over window): {tops_text}")

def plot_context_timeseries(
    block: pd.DataFrame,
    cols: Sequence[str] = ("column_max_refl", "u_motion", "v_motion"),
    *,
    figsize: Tuple[float, float] = (8, 3),
    title: Optional[str] = None,
):
    """Tiny timeseries panel for a few context columns within the window."""
    if block is None or len(block) == 0:
        print("[ctx] no rows to plot.")
        return None, None

    t = pd.to_datetime(block["time_utc"])
    fig, ax = plt.subplots(figsize=figsize)
    for c in cols:
        if c in block.columns:
            ax.plot(t, pd.to_numeric(block[c], errors="coerce"), marker="o", lw=1, label=c)
    ax.set_xlabel("UTC time"); ax.grid(True, alpha=0.3); ax.legend()
    if title:
        ax.set_title(title)
    fig.autofmt_xdate()
    return fig, ax
