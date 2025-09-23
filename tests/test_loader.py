# tests/test_loader.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import pytest

from hrss import loader as hloader
from hrss import io as hio

# -----------------------------
# ENV wiring (mirrors test_io.py)
# -----------------------------
ENV_YEAR_DIR = os.getenv("HRSS_TEST_YEAR_DIR")              # dir that contains manifest.csv/catalog.csv
ENV_ZIP       = os.getenv("HRSS_TEST_ZIP")                  # path to hrss-starter-2017.zip
ENV_MD5_ARCH  = os.getenv("HRSS_TEST_MD5_ARCHIVE")          # md5 for the zip (optional here)
ENV_MD5_MAN   = os.getenv("HRSS_TEST_MD5_MANIFEST")         # md5 for manifest.csv (unused here)
ENV_MD5_CAT   = os.getenv("HRSS_TEST_MD5_CATALOG")          # md5 for catalog.csv  (unused here)

ENV_ZEN_DOI   = os.getenv("HRSS_TEST_ZENODO_DOI")           # DOI or https://.../records/<id>
ENV_ZEN_TOKEN = os.getenv("HRSS_TEST_ZENODO_TOKEN")         # optional

# -----------------------------
# Local HRSS cache per test module
# -----------------------------
@pytest.fixture
def patched_cache_env(tmp_path, monkeypatch):
    """
    Give this test module its own HRSS_CACHE root. Your conftest.py post-session
    cleanup will remove these automatically unless --hrss-keep-cache is set.
    """
    monkeypatch.setenv("HRSS_CACHE", str(tmp_path / "hrss_cache"))
    return tmp_path

# -----------------------------
# Helpers
# -----------------------------
def _pick_first_available_product_kind(source: str) -> str:
    """
    Prefer 'reflectivity' if present; else return the first non-context 'kind'
    seen across the resolved source(s). Robust to file_type values like 'h5',
    'HDF5', '.h5', etc. Prints a small debug summary when nothing is found.
    """
    handles = hio.resolve_source(source)  # always List[SourceHandle]
    kinds: List[str] = []

    for h in handles:
        h = hio.load_index(h)
        df = h.manifest
        if df is None or len(df) == 0:
            continue

        # Robust HDF5 selector (use scalar fallbacks to avoid length mismatch)
        ft = df.get("file_type")
        rp = df.get("relpath")
        ft_lc = ft.astype(str).str.lower() if ft is not None else pd.Series("", index=df.index)
        rp_lc = rp.astype(str).str.lower() if rp is not None else pd.Series("", index=df.index)
        is_h5 = ft_lc.str.contains("h5", regex=False) | rp_lc.str.endswith(".h5")

        # Exclude context products
        kind_col = df["kind"] if "kind" in df.columns else pd.Series("", index=df.index)
        non_context = kind_col.astype(str).str.lower() != "context"

        kk = (
            df[is_h5 & non_context]
            .loc[:, "kind"]
            .astype(str)
            .str.lower()
            .tolist()
        )
        kinds.extend(kk)

    if not kinds:
        # helpful printouts when running with -s
        print("[test helper] No products found. Unique file_type values per handle:")
        for h in handles:
            hh = hio.load_index(h)
            df = hh.manifest
            if df is None or len(df) == 0:
                print("  - (empty manifest)")
                continue
            uniq_ft = sorted(map(str, set(df.get("file_type", []))))
            print("  -", uniq_ft)
        raise hloader.HRSSIOError("No product HDF5 files found in source.")

    if "reflectivity" in kinds:
        return "reflectivity"
    return kinds[0]



def _has_both_products_for_any_storm(ds: hloader.HRSSDataset, a: str, b: str) -> bool:
    """Introspect dataset internals to see if any storm has both products."""
    for (_, _), km in ds._pmap.items():  # type: ignore[attr-defined]
        if a in km and b in km:
            return True
    return False

# -----------------------------
# Basic dataset from YEAR DIR
# -----------------------------
@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_loader_dir_basic_build_and_getitem(patched_cache_env):
    # pick a valid product (prefer reflectivity if available)
    prod = _pick_first_available_product_kind(ENV_YEAR_DIR)

    ds = hloader.load(
        ENV_YEAR_DIR,
        input_product=prod,
        target="future",
        t_in=3, t_out=2, stride=2, cadence_min=5, cadence_tol_sec=120,
    )

    # Basic shape / length
    assert len(ds) > 0
    info = ds.info()
    assert "samples" in info and info["samples"] == len(ds)
    assert info["input_product"] == prod
    assert info["t_in"] == 3 and info["t_out"] == 2

    # First sample
    X, Y, meta = ds[0]
    assert X.ndim == 4 and Y.ndim == 4  # (T,H,W,C)
    assert X.shape[0] == 3 and Y.shape[0] == 2
    assert isinstance(meta.get("site", ""), str)
    assert "storm_id" in meta
    assert meta["input_product"] == prod
    assert meta["target"] == "future"

@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_loader_dir_as_arrays_and_of_params(patched_cache_env):
    prod = _pick_first_available_product_kind(ENV_YEAR_DIR)
    ds = hloader.load(ENV_YEAR_DIR, input_product=prod, target="future", t_in=3, t_out=2)

    # Keep it fast: cap to a handful of samples
    X, Y, groups, times, of_params = ds.as_arrays(limit=5, compute_of_params=True, crop="min")

    # Shapes & alignment
    assert X.shape[0] == Y.shape[0] == len(groups) == len(times)
    assert X.shape[1] == 3 and Y.shape[1] == 2
    assert X.dtype == np.float32 and Y.dtype == np.float32

    # OF params present and aligned
    for k in ("u_bg", "v_bg", "ky_km", "kx_km", "dt_sec"):
        assert k in of_params
    assert of_params["u_bg"].shape[0] == X.shape[0]
    assert of_params["dt_sec"] > 0

@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_loader_dir_grouped_splits(patched_cache_env):
    prod = _pick_first_available_product_kind(ENV_YEAR_DIR)
    ds = hloader.load(ENV_YEAR_DIR, input_product=prod, target="future", t_in=3, t_out=2, stride=2)

    parts = ds.split_indices_by_storm(val_size=0.2, test_size=0.2, seed=123)
    tr, va, te = parts["train"], parts["val"], parts["test"]

    # Disjoint and cover all
    all_idx = set(tr) | set(va) | set(te)
    assert len(all_idx) == len(ds)
    assert set(tr).isdisjoint(set(va)) and set(tr).isdisjoint(set(te)) and set(va).isdisjoint(set(te))

    # Materialize a split quickly
    Xtr, Ytr, gtr, ttr, _ = ds.as_arrays(split="train", val_size=0.2, test_size=0.2, seed=123, limit=5)
    assert Xtr.shape[0] == len(gtr) == len(ttr)
    assert Xtr.shape[1] == 3 and Ytr.shape[1] == 2

# -----------------------------
# Optional: product→product target if both exist
# -----------------------------
@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_loader_dir_product_to_product_when_available(patched_cache_env):
    # Build a dataset to inspect available products per storm
    prod = _pick_first_available_product_kind(ENV_YEAR_DIR)
    ds_probe = hloader.load(ENV_YEAR_DIR, input_product=prod, target="future", t_in=2, t_out=1)

    # Try a couple common combos
    candidates = [
        ("reflectivity", "reflectivity_composite"),
        ("reflectivity_composite", "reflectivity"),
    ]
    chosen: Optional[Tuple[str, str]] = None
    for a, b in candidates:
        if _has_both_products_for_any_storm(ds_probe, a, b):
            chosen = (a, b)
            break

    if chosen is None:
        pytest.skip("No storm with both reflectivity and reflectivity_composite products; skipping cross-product target test.")

    a, b = chosen
    ds = hloader.load(ENV_YEAR_DIR, input_product=a, target=b, t_in=2, t_out=1)
    assert len(ds) > 0

    # Ensure Y channel count can differ and shapes are sensible
    X, Y, meta = ds[0]
    assert X.shape[0] == 2 and Y.shape[0] == 1
    assert meta["input_product"] == a and meta["target"] == b

# -----------------------------
# Basic dataset from local ZIP
# -----------------------------
@pytest.mark.skipif(not ENV_ZIP, reason="HRSS_TEST_ZIP not set")
def test_loader_zip_basic(patched_cache_env):
    prod = _pick_first_available_product_kind(ENV_ZIP)
    ds = hloader.load(ENV_ZIP, input_product=prod, target="future", t_in=3, t_out=2)
    assert len(ds) > 0
    X, Y, meta = ds[0]
    assert X.shape[0] == 3 and Y.shape[0] == 2

# -----------------------------
# Optional: Zenodo E2E (heavy; skipped unless DOI provided)
# -----------------------------
@pytest.mark.skipif(not ENV_ZEN_DOI, reason="HRSS_TEST_ZENODO_DOI not set")
def test_loader_zenodo_minimal_e2e(patched_cache_env):
    # Keep it tiny: small T, few samples. Uses API /content path (handled by io.py).
    prod = "reflectivity"  # typical; fall back to auto-pick if missing
    try:
        # If reflectivity not available, auto-pick a kind from manifest.
        prod = _pick_first_available_product_kind(ENV_ZEN_DOI)
    except Exception:
        pass

    ds = hloader.load(
        ENV_ZEN_DOI,
        token=ENV_ZEN_TOKEN,
        input_product=prod,
        target="future",
        t_in=2, t_out=1, stride=3,
    )
    # Just touch a few items to avoid long runs
    n = min(3, len(ds))
    for i in range(n):
        X, Y, meta = ds[i]
        assert X.shape[0] == 2 and Y.shape[0] == 1
