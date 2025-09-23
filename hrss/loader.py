# hrss/loader.py
from __future__ import annotations

import dataclasses as dc
import logging
import math
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd

try:
    import h5py  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("h5py is required for hrss.loader. Install with `pip install h5py`.") from e

from . import io as hio
# public things re-exported so users can import from hrss.loader if they want
HRSSIOError = hio.HRSSIOError
HRSSIntegrityError = hio.HRSSIntegrityError

log = logging.getLogger("hrss.loader")
log.addHandler(logging.NullHandler())

# ----------------------------- small utils -----------------------------


def _summ_times_ms(times_ms: np.ndarray) -> str:
    """Compact description of a time vector for debug logging."""
    NAT = np.iinfo(np.int64).max
    n = times_ms.size
    if n == 0:
        return "times: [empty]"
    n_nat = int(np.sum(times_ms == NAT))
    n_val = n - n_nat
    if n_val >= 2:
        v = times_ms[times_ms != NAT]
        deltas = (v[1:] - v[:-1]) // 1000
        return (f"times: n={n} val={n_val} nat={n_nat} "
                f"dt_sec[min/med/max]={deltas.min()}/{int(np.median(deltas))}/{deltas.max()}")
    return f"times: n={n} val={n_val} nat={n_nat}"

def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _candidate_starts_from_times_ms(
    times_ms: np.ndarray,
    need: int,
    cadence_min: int,
    tol_sec: int,
    stride: int,
) -> np.ndarray:
    """
    Return start indices for sliding windows (length=need) that satisfy cadence and have no NaTs.
    times_ms: int64 (ms since epoch). NaT represented by np.iinfo(np.int64).max
    """
    if times_ms.size < need:
        return np.empty((0,), dtype=np.int32)

    NAT = np.iinfo(np.int64).max
    valid = times_ms != NAT
    pair_valid = valid[:-1] & valid[1:]
    deltas_sec = (times_ms[1:] - times_ms[:-1]) // 1000

    target = int(cadence_min) * 60
    ok_dt = pair_valid & (deltas_sec >= (target - tol_sec)) & (deltas_sec <= (target + tol_sec))

    # fast rolling sums
    def _movsum(b, w):
        if w <= 0 or b.size == 0:
            return np.zeros((0,), dtype=np.int32)
        c = np.cumsum(b.astype(np.int32))
        return c[w-1:] - np.concatenate(([0], c[:-w]))

    ok_runs = _movsum(ok_dt, need - 1)
    valid_runs = _movsum(valid.astype(np.int32), need)
    L = min(ok_runs.size, valid_runs.size)
    starts = np.where((ok_runs[:L] == (need - 1)) & (valid_runs[:L] == need))[0]
    if stride > 1:
        starts = starts[::stride]
    return starts.astype(np.int32)

def _compute_km_per_pixel(min_lat, max_lat, min_lon, max_lon, H: int, W: int) -> Tuple[float, float]:
    vals = [min_lat, max_lat, min_lon, max_lon]
    if any(pd.isna(v) for v in vals) or H <= 0 or W <= 0:
        return (float("nan"), float("nan"))
    lat_span = float(max_lat) - float(min_lat)
    lon_span = float(max_lon) - float(min_lon)
    if lat_span <= 0 or lon_span <= 0:
        return (float("nan"), float("nan"))
    mean_lat = (float(min_lat) + float(max_lat)) / 2.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * float(np.cos(np.deg2rad(mean_lat)))
    ky = (lat_span * km_per_deg_lat) / float(H)
    kx = (lon_span * km_per_deg_lon) / float(W)
    return (float(ky), float(kx))


# ----------------------------- dataset index -----------------------------

@dc.dataclass(slots=True)
class _ProductFile:
    handle: hio.SourceHandle      # which archive/dir it came from
    relpath: str                  # POSIX relpath as in manifest
    local_path: Path              # absolute path on local FS (extracted if zip)
    kind: str                     # e.g., "reflectivity", "reflectivity_composite"
    site: str
    storm_id: str
    T: int
    H: int
    W: int
    C: int
    times_ms: np.ndarray          # int64 ms since epoch (length T)
    context_relpath: Optional[str]  # relpath within storm_dir
    storm_dir_rel: Optional[str]    # relpath to the storm directory (prefix of file relpath)

@dc.dataclass(slots=True)
class _Sample:
    p_in: _ProductFile            # input product file
    start: int                    # start index within p_in.times_ms
    t_in: int                     # frames for X
    t_out: int                    # frames for Y
    target_mode: str              # "future" or "product"
    p_tgt: Optional[_ProductFile] # when target_mode == "product"
    site: str
    storm_id: str
    t0_ms: int
    t1_ms: int

# ----------------------------- public dataset -----------------------------

class HRSSDataset:
    """
    A logical dataset assembled from one or more HRSS archives (dir/zip/DOI).
    Yields windowed samples: (X, Y, meta) lazily from HDF5.

    Typical use:
      ds = HRSSDataset(source, input_product="reflectivity", target="future", t_in=6, t_out=6)
      x, y, meta = ds[0]
      X, Y, groups, times, of_params = ds.as_arrays(split="all", compute_of_params=True)
    """
    def __init__(
        self,
        source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        *,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        years: Optional[Iterable[int]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        token: Optional[str] = None,
        include_sites: Optional[Iterable[str]] = None,
        include_storms: Optional[Iterable[Union[str, int]]] = None,
        input_product: str = "reflectivity",
        target: Union[str, Literal["future"]] = "future",  # another product kind or "future"
        t_in: int = 6,
        t_out: int = 6,
        stride: int = 1,
        cadence_min: int = 5,
        cadence_tol_sec: int = 90,
        fill_nan: Optional[float] = None,
        transform_X: Optional[Callable[[np.ndarray, Dict], np.ndarray]] = None,
        transform_Y: Optional[Callable[[np.ndarray, Dict], np.ndarray]] = None,
        joint_transform: Optional[Callable[[np.ndarray, np.ndarray, Dict], Tuple[np.ndarray, np.ndarray]]] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.input_product = str(input_product)
        self.target_mode = "future" if str(target).lower() == "future" else "product"
        self.target_product = None if self.target_mode == "future" else str(target)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.need = self.t_in + self.t_out
        self.stride = max(1, int(stride))
        self.cadence_min = int(cadence_min)
        self.cadence_tol_sec = int(cadence_tol_sec)
        self.fill_nan = fill_nan
        self.transform_X = transform_X
        self.transform_Y = transform_Y
        self.joint_transform = joint_transform
        self.seed = seed

        src_list: List[Union[str, os.PathLike]] = _as_list(source)
        log.debug("HRSSDataset init: sources=%s, years=%s, date_range=[%s..%s], input_product=%s, target=%s, T_in=%d, T_out=%d, stride=%d",
                src_list, list(years) if years else None, date_from, date_to,
                self.input_product, (self.target_product or "future"), self.t_in, self.t_out, self.stride)

        # Resolve -> handles (always a list with your latest io.py)
        handles: List[hio.SourceHandle] = []
        for src in src_list:
            got = hio.resolve_source(
                src,
                cache_dir=cache_dir,
                years=years,
                date_from=date_from,
                date_to=date_to,
                token=token,
            )
            handles.extend(got)
            log.debug("Resolved %s -> %d handle(s)", src, len(got))

        if not handles:
            raise HRSSIOError("No sources resolved.")

        # Load indices (manifest/catalog)
        self._handles = []
        self._manifest = []
        self._catalog = []
        for h in handles:
            h2 = hio.load_index(h)
            self._handles.append(h2)
            self._manifest.append(h2.manifest.copy())
            self._catalog.append(h2.catalog.copy())
            log.debug("Loaded index for handle kind=%s cache_dir=%s: manifest=%s rows, catalog=%s rows",
                    h2.kind, h2.cache_dir, len(h2.manifest), len(h2.catalog))

        man = pd.concat(self._manifest, ignore_index=True)
        cat = pd.concat(self._catalog, ignore_index=True)
        log.debug("Merged manifest rows=%d, catalog rows=%d", len(man), len(cat))

        # Filters
        if include_sites:
            before = len(man)
            include_sites = {str(s).upper() for s in include_sites}
            man = man[man["site"].astype(str).str.upper().isin(include_sites)]
            cat = cat[cat["site"].astype(str).str.upper().isin(include_sites)]
            log.debug("Filtered by sites (%s): manifest %d -> %d", include_sites, before, len(man))
        if include_storms:
            before = len(man)
            inc = {str(int(s)) if str(s).isdigit() else str(s) for s in include_storms}
            man = man[man["storm_id"].astype(str).isin(inc)]
            cat = cat[cat["storm_id"].astype(str).isin(inc)]
            log.debug("Filtered by storms (%s): manifest %d -> %d", inc, before, len(man))

        # Build product map and samples
        self._pmap = {}
        self._build_product_map(man)
        self._samples = []
        self._build_samples()
        self._groups = [s.storm_id for s in self._samples]

        log.debug("Dataset ready: samples=%d, unique storms=%d, sites=%s",
                len(self._samples), len(set(self._groups)),
                sorted(set(k[0] for k in self._pmap.keys())))


    # ----------- building internals -----------

    def _handle_for_relpath(self, relpath: str) -> hio.SourceHandle:
        # Pick any handle whose manifest contains relpath (fast: we already merged)
        # Fallback: try all handles based on expected cache layout — but we keep it simple here:
        for h in self._handles:
            if h.manifest is not None and (h.manifest["relpath"] == relpath).any():
                return h
        # Should not happen if manifest was merged correctly
        return self._handles[0]

    def _ensure_local(self, handle: hio.SourceHandle, relpath: str) -> Path:
        return hio.ensure_local_file(handle, relpath, verify=True)

    def _read_times_from_h5(self, p: Path) -> np.ndarray:
        """
        Returns int64 ms since epoch (NaT -> max int). Prefers /time_unix_ms if present,
        otherwise decodes /time (strings/bytes) robustly (pandas >= 2.0 friendly).
        """
        NAT = np.iinfo(np.int64).max
        try:
            with h5py.File(p, "r") as h5:
                # 1) Fast path: explicit ms
                if "time_unix_ms" in h5:
                    arr = np.asarray(h5["time_unix_ms"][...], dtype="int64")
                    bad = ~np.isfinite(arr)
                    if bad.any():
                        arr[bad] = NAT
                    log.debug("read_times_from_h5(%s): used time_unix_ms (T=%d, bad=%d)",
                            p, arr.shape[0], int(bad.sum()))
                    return arr

                # 2) Fallback: string/bytes time → parse → ns → ms
                if "time" in h5:
                    raw = h5["time"][...]
                    # decode bytes if needed
                    if isinstance(raw, np.ndarray) and raw.dtype.kind in ("S", "O", "U"):
                        s = [t.decode("utf-8", errors="ignore") if isinstance(t, (bytes, bytearray)) else str(t) for t in raw]
                    else:
                        s = [str(t) for t in np.asarray(raw).tolist()]
                    ts = pd.to_datetime(s, utc=True, errors="coerce")

                    # pandas 2.x friendly conversion to ns
                    if hasattr(ts, "asi8"):
                        ns = np.asarray(ts.asi8, dtype=np.int64)  # NaT -> int64 min
                    else:
                        ns = np.asarray(ts.astype("int64", copy=False), dtype=np.int64)

                    mask_nat = ns == np.iinfo(np.int64).min
                    ms = (ns // 1_000_000).astype("int64", copy=False)
                    ms[mask_nat] = NAT

                    nT = int(ms.shape[0])
                    nNaT = int(mask_nat.sum())
                    log.debug("read_times_from_h5(%s): parsed time strings (T=%d, NaT=%d)", p, nT, nNaT)
                    return ms
        except Exception as e:
            log.debug("read_times_from_h5(%s) failed with %s; will fall back to T-only.", p, e)

        # 3) Last resort: shape from /data, mark all NaT
        try:
            with h5py.File(p, "r") as h5:
                T = int(h5["data"].shape[0])
        except Exception:
            T = 0
        log.debug("read_times_from_h5(%s): fallback path, marking %d frames as NaT", p, T)
        return np.full((T,), NAT, dtype="int64")


    def _storm_dir_from_rel(self, rel: str) -> Optional[str]:
        # rel like: KDLH/storm_2898/....h5 → return "KDLH/storm_2898"
        p = Path(rel)
        if "storm_" in p.as_posix():
            parts = p.parts
            for i in range(len(parts)):
                if parts[i].lower().startswith("storm_"):
                    return Path(*parts[: i + 1]).as_posix()
        return p.parent.as_posix()  # best-effort

    def _build_product_map(self, man: pd.DataFrame) -> None:
        # Robust HDF5 row selection (accept 'h5', 'hdf5', '.h5', etc.)
        ft = man.get("file_type")
        rp = man.get("relpath")
        ft_lc = ft.astype(str).str.lower() if ft is not None else pd.Series("", index=man.index)
        rp_lc = rp.astype(str).str.lower() if rp is not None else pd.Series("", index=man.index)
        is_h5 = ft_lc.str.contains("h5", regex=False) | rp_lc.str.endswith(".h5")

        rows = man[is_h5].copy()
        if "size_bytes" not in rows.columns:
            rows["size_bytes"] = 0
        rows["size_bytes"] = pd.to_numeric(rows["size_bytes"], errors="coerce").fillna(0).astype(int)

        rows = rows.sort_values(
            ["site", "storm_id", "kind", "size_bytes"],
            ascending=[True, True, True, False]
        )

        picked: Dict[Tuple[str, str, str], Dict] = {}
        for _, r in rows.iterrows():
            key = (str(r["site"]), str(r["storm_id"]), str(r["kind"]))
            if key not in picked:
                picked[key] = r.to_dict()

        totals: Dict[str, int] = {}
        for (site, storm, kind), r in picked.items():
            rel = str(r["relpath"])
            handle = self._handle_for_relpath(rel)
            local = self._ensure_local(handle, rel)

            with h5py.File(local, "r") as h5:
                if "data" not in h5 or h5["data"].ndim != 4:
                    log.debug("Skip (no /data or wrong rank) %s", local)
                    continue
                T, H, W, C = map(int, h5["data"].shape)
                ctx_rel = None
                try:
                    ctx_rel = h5.attrs.get("context_relpath", None)
                    if isinstance(ctx_rel, (bytes, bytearray)):
                        ctx_rel = ctx_rel.decode("utf-8", errors="ignore")
                    if ctx_rel:
                        ctx_rel = str(ctx_rel)
                except Exception:
                    ctx_rel = None

            times_ms = self._read_times_from_h5(local)
            storm_dir_rel = self._storm_dir_from_rel(rel)

            pf = _ProductFile(
                handle=handle,
                relpath=rel,
                local_path=Path(local),
                kind=kind,
                site=site,
                storm_id=str(storm),
                T=T, H=H, W=W, C=C,
                times_ms=times_ms,
                context_relpath=ctx_rel,
                storm_dir_rel=storm_dir_rel,
            )
            self._pmap.setdefault((site, str(storm)), {})[kind] = pf
            totals[kind] = totals.get(kind, 0) + 1
            log.debug("Picked product: site=%s storm=%s kind=%s T=%d HxW=%dx%d C=%d rel=%s",
                    site, storm, kind, T, H, W, C, rel)

        have = sum(1 for _k, m in self._pmap.items() if self.input_product in m)
        if have == 0:
            uniq_ft = sorted(set(ft_lc[ft_lc.notna()].tolist()))
            log.debug("No '%s' found. file_type uniques seen: %s", self.input_product, uniq_ft)
            raise HRSSIOError(f"No product '{self.input_product}' found in sources.")

        if self.target_mode == "product":
            miss = sum(1 for _k, m in self._pmap.items() if self.target_product not in m)
            if miss > 0:
                log.warning("Target product '%s' missing for %d storm(s); those storms will contribute no samples.",
                            self.target_product, miss)

        log.debug("Product map built: storms=%d, kinds_count=%s", len(self._pmap), totals)



    def _build_samples(self) -> None:
        need = self.need
        total_windows = 0
        storms_considered = 0

        for (site, storm), kinds in self._pmap.items():
            if self.input_product not in kinds:
                continue
            storms_considered += 1
            p_in = kinds[self.input_product]

            starts = _candidate_starts_from_times_ms(
                p_in.times_ms, need, self.cadence_min, self.cadence_tol_sec, self.stride
            )

            log.debug("Windows for site=%s storm=%s: candidates=%d (T=%d, cadence=%d±%d sec, stride=%d)",
                    site, storm, int(starts.size), int(p_in.T), int(self.cadence_min*60),
                    self.cadence_tol_sec, self.stride)

            if starts.size == 0:
                continue

            p_tgt = None
            if self.target_mode == "product":
                if self.target_product not in kinds:
                    log.debug("Storm (%s,%s) lacks target '%s' — skipping",
                            site, storm, self.target_product)
                    continue
                p_tgt = kinds[self.target_product]
                if p_tgt.T != p_in.T:
                    log.warning("Storm (%s,%s) target T=%d != input T=%d; skipping.",
                                site, storm, p_tgt.T, p_in.T)
                    continue

            for s in starts:
                t0 = int(p_in.times_ms[s])
                t1 = int(p_in.times_ms[s + need - 1])
                self._samples.append(_Sample(
                    p_in=p_in, start=int(s), t_in=self.t_in, t_out=self.t_out,
                    target_mode=self.target_mode, p_tgt=p_tgt,
                    site=site, storm_id=str(storm), t0_ms=t0, t1_ms=t1
                ))
            total_windows += int(starts.size)

        log.debug("Built samples: storms_considered=%d, total_windows=%d, samples=%d",
                storms_considered, total_windows, len(self._samples))


    # ----------- python dataset protocol -----------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        s = self._samples[idx]
        X = self._read_window(s.p_in.local_path, s.start, s.t_in)

        if s.target_mode == "future":
            Y = self._read_window(s.p_in.local_path, s.start + s.t_in, s.t_out)
            tgt_path = s.p_in.local_path
        else:
            assert s.p_tgt is not None
            Y = self._read_window(s.p_tgt.local_path, s.start + s.t_in, s.t_out)
            tgt_path = s.p_tgt.local_path

        meta = {
            "site": s.site,
            "storm_id": s.storm_id,
            "t0_ms": s.t0_ms,
            "t1_ms": s.t1_ms,
            "input_product": self.input_product,
            "target": self.target_product if self.target_mode == "product" else "future",
            "HWC_in": X.shape[1:],
            "HWC_out": Y.shape[1:],
            "path_in": str(s.p_in.local_path),
            "path_tgt": str(tgt_path),
        }

        if self.fill_nan is not None:
            X = np.nan_to_num(X, nan=self.fill_nan, posinf=self.fill_nan, neginf=self.fill_nan)
            Y = np.nan_to_num(Y, nan=self.fill_nan, posinf=self.fill_nan, neginf=self.fill_nan)

        if self.joint_transform is not None:
            X, Y = self.joint_transform(X, Y, meta)
        else:
            if self.transform_X is not None:
                X = self.transform_X(X, meta)
            if self.transform_Y is not None:
                Y = self.transform_Y(Y, meta)

        log.debug("__getitem__(%d): X%s Y%s site=%s storm=%s start=%d", idx, X.shape, Y.shape, s.site, s.storm_id, s.start)
        return X, Y, meta


    def _read_window(self, local_path: Path, start: int, length: int) -> np.ndarray:
        with h5py.File(local_path, "r") as h5:
            d = h5["data"]
            stop = start + length
            arr = d[start:stop, :, :, :]
            out = np.array(arr, dtype=np.float32, copy=False)
        log.debug("_read_window(%s, [%d:%d]) -> %s", local_path, start, start+length, out.shape)
        return out

    # ----------- summaries / splitting / export -----------

    def info(self) -> Dict[str, object]:
        storms = sorted(set(self._groups))
        dims = {}
        for (site, storm), kinds in self._pmap.items():
            if self.input_product in kinds:
                p = kinds[self.input_product]
                dims.setdefault((p.H, p.W, p.C), 0)
                dims[(p.H, p.W, p.C)] += 1
        return {
            "samples": len(self),
            "storms": len(storms),
            "sites": sorted(set(k[0] for k in self._pmap.keys())),
            "dims_counts": {f"{k}": v for k, v in dims.items()},
            "input_product": self.input_product,
            "target": self.target_product if self.target_mode == "product" else "future",
            "t_in": self.t_in,
            "t_out": self.t_out,
            "cadence_min": self.cadence_min,
            "stride": self.stride,
        }

    def split_indices_by_storm(
        self,
        *,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        seed = self.seed if seed is None else seed
        storms = np.array(sorted(set(self._groups)))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(storms))
        storms = storms[perm]

        n = len(storms)
        n_test = int(round(n * float(test_size)))
        n_val = int(round((n - n_test) * float(val_size)))

        test_storms = set(storms[:n_test])
        val_storms = set(storms[n_test: n_test + n_val])
        train_storms = set(storms[n_test + n_val:])

        idx = np.arange(len(self), dtype=int)
        groups = np.array(self._groups)
        pick = lambda S: idx[np.isin(groups, list(S))]
        parts = {
            "train": pick(train_storms),
            "val":   pick(val_storms),
            "test":  pick(test_storms),
        }
        log.debug("split_indices_by_storm: storms=%d -> train=%d val=%d test=%d (seed=%s)",
                n, len(parts["train"]), len(parts["val"]), len(parts["test"]), seed)
        return parts


    def as_arrays(
        self,
        *,
        split: str = "all",
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: Optional[int] = None,
        crop: str = "min",
        compute_of_params: bool = True,
        limit: Optional[int] = None,
        fill_nan: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[pd.Timestamp, pd.Timestamp]], Dict[str, np.ndarray]]:
        if split != "all":
            parts = self.split_indices_by_storm(val_size=val_size, test_size=test_size, seed=seed)
            if split not in parts:
                raise ValueError(f"Unknown split={split}")
            indices = parts[split]
        else:
            indices = np.arange(len(self), dtype=int)

        if limit is not None:
            indices = indices[: int(limit)]

        if indices.size == 0:
            log.debug("as_arrays(%s): empty selection", split)
            empty_X = np.empty((0, self.t_in, 0, 0, 0), dtype=np.float32)
            empty_Y = np.empty((0, self.t_out, 0, 0, 0), dtype=np.float32)
            return empty_X, empty_Y, [], [], {
                "u_bg": np.empty((0,)), "v_bg": np.empty((0,)),
                "ky_km": np.empty((0,)), "kx_km": np.empty((0,)),
                "dt_sec": np.array(self.cadence_min * 60, dtype=np.float32)
            }

        Hs, Ws = [], []
        for i in indices:
            s = self._samples[i]
            Hs.append(s.p_in.H)
            Ws.append(s.p_in.W)
        Hmin, Wmin = min(Hs), min(Ws)
        Cx = self._samples[indices[0]].p_in.C
        if self._samples[indices[0]].target_mode == "product":
            C_y = self._samples[indices[0]].p_tgt.C if self._samples[indices[0]].p_tgt else 1
        else:
            C_y = Cx

        log.debug("as_arrays(%s): N=%d, crop=min-> (%d,%d), Cx=%d, Cy=%d", split, len(indices), Hmin, Wmin, Cx, C_y)

        X = np.empty((len(indices), self.t_in, Hmin, Wmin, Cx), dtype=np.float32)
        Y = np.empty((len(indices), self.t_out, Hmin, Wmin, C_y), dtype=np.float32)

        groups: List[str] = []
        times: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        u_bg = np.full((len(indices),), np.nan, dtype=np.float32)
        v_bg = np.full((len(indices),), np.nan, dtype=np.float32)
        ky_km = np.full((len(indices),), np.nan, dtype=np.float32)
        kx_km = np.full((len(indices),), np.nan, dtype=np.float32)

        n_ctx_hits = 0
        for j, i in enumerate(indices):
            x, y, meta = self[i]
            X[j] = x[:, :Hmin, :Wmin, :]
            Y[j] = y[:, :Hmin, :Wmin, :]

            groups.append(str(meta["storm_id"]))
            t0 = pd.to_datetime(int(meta["t0_ms"]), unit="ms", utc=True, errors="coerce")
            t1 = pd.to_datetime(int(meta["t1_ms"]), unit="ms", utc=True, errors="coerce")
            times.append((t0, t1))

            if compute_of_params:
                try:
                    pf = self._samples[i].p_in
                    ctx_rel = pf.context_relpath
                    ctx_full_rel = Path(pf.storm_dir_rel or "").joinpath(ctx_rel).as_posix() if ctx_rel else None
                    handle = pf.handle
                    if ctx_full_rel:
                        ctx_local = hio.ensure_local_file(handle, ctx_full_rel, verify=False)
                    else:
                        storm_dir = Path(pf.relpath).parent.as_posix()
                        m = handle.manifest
                        cand = m[(m["relpath"].str.startswith(storm_dir)) &
                                (m["kind"].astype(str).str.lower() == "context")]
                        ctx_local = hio.ensure_local_file(handle, str(cand.iloc[0]["relpath"]), verify=False) if len(cand) > 0 else None

                    if ctx_local and Path(ctx_local).exists():
                        df = pd.read_hdf(ctx_local, key="context", mode="r")
                        if "time_unix_ms" in df.columns:
                            block = df[(df["time_unix_ms"] >= int(meta["t0_ms"])) &
                                    (df["time_unix_ms"] <= int(meta["t1_ms"]))]
                        else:
                            block = None
                            if "time" in df.columns:
                                tt = pd.to_datetime(df["time"], utc=True, errors="coerce")
                                tms = (tt.view("int64") // 1_000_000)  # ok on Series; but we can be safe:
                                try:
                                    tms = (tt.astype("int64", copy=False) // 1_000_000)
                                except Exception:
                                    tms = (tt.view("int64") // 1_000_000)
                                df = df.assign(time_unix_ms=tms)
                                block = df[(df["time_unix_ms"] >= int(meta["t0_ms"])) &
                                        (df["time_unix_ms"] <= int(meta["t1_ms"]))]

                        if block is not None and len(block) > 0:
                            n_ctx_hits += 1
                            u = pd.to_numeric(block.get("u_motion"), errors="coerce")
                            v = pd.to_numeric(block.get("v_motion"), errors="coerce")
                            u_bg[j] = float(np.nanmean(u)) if len(u) else np.nan
                            v_bg[j] = float(np.nanmean(v)) if len(v) else np.nan

                            min_lat = float(pd.to_numeric(block.get("min_lat"), errors="coerce").median())
                            max_lat = float(pd.to_numeric(block.get("max_lat"), errors="coerce").median())
                            min_lon = float(pd.to_numeric(block.get("min_lon"), errors="coerce").median())
                            max_lon = float(pd.to_numeric(block.get("max_lon"), errors="coerce").median())
                            ky_km[j], kx_km[j] = _compute_km_per_pixel(min_lat, max_lat, min_lon, max_lon, Hmin, Wmin)
                except Exception as e:
                    log.debug("OF params computation failed for sample %d: %s", i, e)

        if fill_nan is not None:
            X = np.nan_to_num(X, nan=fill_nan, posinf=fill_nan, neginf=fill_nan)
            Y = np.nan_to_num(Y, nan=fill_nan, posinf=fill_nan, neginf=fill_nan)

        of_params = {
            "u_bg": u_bg, "v_bg": v_bg, "ky_km": ky_km, "kx_km": kx_km,
            "dt_sec": np.array(self.cadence_min * 60, dtype=np.float32),
        }
        log.debug("as_arrays(%s): X%s Y%s groups=%d ctx_hits=%d", split, X.shape, Y.shape, len(groups), n_ctx_hits)
        return X, Y, groups, times, of_params


    # ----------- optional adapters -----------

    def to_torch(self):
        """
        Return a torch.utils.data.Dataset view that pulls from this HRSSDataset.
        Torch is optional; we import lazily.
        """
        try:
            import torch  # type: ignore
            from torch.utils.data import Dataset  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyTorch is not installed. Install torch to use to_torch().") from e

        ds = self

        class TorchDataset(Dataset):  # type: ignore
            def __len__(self): return len(ds)
            def __getitem__(self, i):
                X, Y, meta = ds[i]
                # Channels-last → keep as-is; users can permute in collate if needed.
                return torch.from_numpy(X), torch.from_numpy(Y), meta

        return TorchDataset()


# ----------------------------- convenience factory -----------------------------

def load(
    source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    **kwargs,
) -> HRSSDataset:
    """
    Shorthand for HRSSDataset(...). You can pass all HRSSDataset kwargs here.
    Example:
      ds = hrss.loader.load("10.5072/zenodo.12345", input_product="reflectivity", target="future", t_in=6, t_out=6)
    """
    return HRSSDataset(source, **kwargs)


# ----------------------------- simple group splits (export helpers) -----------------------------

def split_train_val_by_storm(
    X: np.ndarray, Y: np.ndarray, groups: Sequence[str], val_size: float = 0.2, seed: Optional[int] = 42
):
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * float(val_size)))
    val = set(uniq[:n_val])
    idx = np.arange(len(groups))
    val_idx = idx[np.isin(groups, list(val))]
    tr_idx = idx[~np.isin(groups, list(val))]
    pick = lambda I: (X[I], Y[I], [groups[i] for i in I])
    return pick(tr_idx), pick(val_idx)

def split_train_val_test_by_storm(
    X: np.ndarray, Y: np.ndarray, groups: Sequence[str], val_size: float = 0.1, test_size: float = 0.1, seed: Optional[int] = 42
):
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n_test = int(round(len(uniq) * float(test_size)))
    test = set(uniq[:n_test])
    rem = uniq[n_test:]
    n_val = int(round(len(rem) * float(val_size)))
    val = set(rem[:n_val])
    train = set(rem[n_val:])

    idx = np.arange(len(groups))
    pick = lambda S: idx[np.isin(groups, list(S))]
    tr = pick(train); va = pick(val); te = pick(test)
    sel = lambda I: (X[I], Y[I], [groups[i] for i in I])
    return sel(tr), sel(va), sel(te)
