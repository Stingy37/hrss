# tests/test_io.py
from __future__ import annotations
import os
import re
from pathlib import Path
import pytest

from hrss.io import (
    resolve_source,
    load_index,
    ensure_local_file,
    read_schema,
    peek_h5_attrs,
    HRSSIOError,
    HRSSIntegrityError,
)

# -----------------------------
# ENV wiring
# -----------------------------
ENV_YEAR_DIR = os.getenv("HRSS_TEST_YEAR_DIR")              # dir that contains manifest.csv/catalog.csv
ENV_ZIP       = os.getenv("HRSS_TEST_ZIP")                  # path to hrss-starter-2017.zip
ENV_MD5_ARCH  = os.getenv("HRSS_TEST_MD5_ARCHIVE")          # md5 for the zip
ENV_MD5_MAN   = os.getenv("HRSS_TEST_MD5_MANIFEST")         # md5 for manifest.csv
ENV_MD5_CAT   = os.getenv("HRSS_TEST_MD5_CATALOG")          # md5 for catalog.csv

ENV_ZEN_DOI   = os.getenv("HRSS_TEST_ZENODO_DOI")           # DOI or https://.../records/<id>
ENV_ZEN_TOKEN = os.getenv("HRSS_TEST_ZENODO_TOKEN")         # optional

# -----------------------------
# Helpers
# -----------------------------
def pick_smallest_relpath(manifest_df, *, suffix=".h5"):
    """Pick a tiny file to keep tests fast (often a context .h5 is smallest)."""
    df = manifest_df.copy()
    if suffix:
        df = df[df["relpath"].str.lower().str.endswith(suffix)]
    if len(df) == 0:
        # fall back: any file
        df = manifest_df
    df = df.sort_values("size_bytes", ascending=True)
    return df.iloc[0]["relpath"]

def pick_context_relpath(manifest_df):
    ctx = manifest_df[manifest_df["kind"] == "context"]
    if len(ctx) == 0:
        return None
    ctx = ctx.sort_values("size_bytes", ascending=True)
    return ctx.iloc[0]["relpath"]

def pick_product_relpath(manifest_df):
    """
    Pick the smallest non-context .h5 file so tests stay fast.
    """
    df = manifest_df.copy()
    df = df[
        (df["relpath"].str.lower().str.endswith(".h5")) &
        (df["kind"].astype(str).str.lower() != "context")
    ]
    if len(df) == 0:
        return None
    df = df.sort_values("size_bytes", ascending=True)
    return df.iloc[0]["relpath"]

# pytest will create a new temp dir per test function
@pytest.fixture
def patched_cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HRSS_CACHE", str(tmp_path / "hrss_cache"))
    return tmp_path

# -----------------------------
# Local DIRECTORY tests
# -----------------------------
@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_dir_resolve_and_load_index(patched_cache_env):
    handle = resolve_source(
        ENV_YEAR_DIR,
        expected_manifest_md5=ENV_MD5_MAN,
        expected_catalog_md5=ENV_MD5_CAT,
    )
    assert handle.kind == "dir"
    handle = load_index(handle)
    assert handle.index_loaded
    assert handle.manifest is not None and handle.catalog is not None
    # sanity: manifest rows exist
    assert len(handle.manifest) > 0
    assert set(["relpath","sha256","file_type"]).issubset(handle.manifest.columns)

@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_dir_ensure_local_file_and_verify(patched_cache_env):
    handle = load_index(resolve_source(ENV_YEAR_DIR))
    rel = pick_smallest_relpath(handle.manifest)
    p = ensure_local_file(handle, rel)
    # For directory sources: file should live *under* the year_root
    assert str(p).startswith(str(handle.year_root))
    assert p.exists()

@pytest.mark.skipif(not ENV_YEAR_DIR, reason="HRSS_TEST_YEAR_DIR not set")
def test_dir_read_product_attrs_and_optional_context_schema(patched_cache_env):
    handle = load_index(resolve_source(ENV_YEAR_DIR))
    prod_rel = pick_product_relpath(handle.manifest)
    if prod_rel is None:
        pytest.skip("No product .h5 present in manifest")
    prod_path = ensure_local_file(handle, prod_rel)

    attrs = peek_h5_attrs(prod_path)
    assert isinstance(attrs, dict)

    # Product files should expose at least one of these identifying attrs
    expected_any = {"product_prefix", "shape_T_H_W_C", "dataset_version", "radar_site"}
    assert any(k in attrs for k in expected_any)

    # If the product links to a context file, try reading its schema sidecar
    ctx_rel = attrs.get("context_relpath")
    if ctx_rel:
        ctx_path = (prod_path.parent / str(ctx_rel)).resolve()
        schema = read_schema(ctx_path)
        if schema is not None:
            assert "columns" in schema


@pytest.mark.skipif(not ENV_YEAR_DIR or not ENV_MD5_MAN, reason="Need year dir and MD5 to test failure")
def test_dir_bad_manifest_md5_raises(patched_cache_env):
    with pytest.raises(HRSSIntegrityError):
        resolve_source(ENV_YEAR_DIR, expected_manifest_md5="deadbeef" * 4)

# -----------------------------
# Local ZIP tests
# -----------------------------
@pytest.mark.skipif(not ENV_ZIP, reason="HRSS_TEST_ZIP not set")
def test_zip_resolve_and_load_index(patched_cache_env):
    handle = resolve_source(
        ENV_ZIP,
        expected_archive_md5=ENV_MD5_ARCH,
        expected_manifest_md5=ENV_MD5_MAN,
        expected_catalog_md5=ENV_MD5_CAT,
    )
    assert handle.kind == "zip"
    handle = load_index(handle)
    assert handle.index_loaded
    assert handle.manifest is not None and handle.catalog is not None
    assert handle.zip_year_root_in_archive is not None

@pytest.mark.skipif(not ENV_ZIP, reason="HRSS_TEST_ZIP not set")
def test_zip_ensure_local_file_extracts_once(patched_cache_env):
    handle = load_index(resolve_source(ENV_ZIP))
    rel = pick_smallest_relpath(handle.manifest)
    p1 = ensure_local_file(handle, rel)
    assert p1.exists()
    # For zip sources: file should live *under* the cache_dir, mirroring relpath
    assert str(p1).startswith(str(handle.cache_dir))

    # Second call should be idempotent; same path, no re-extract
    p2 = ensure_local_file(handle, rel)
    assert p2 == p1

@pytest.mark.skipif(not ENV_ZIP, reason="HRSS_TEST_ZIP not set")
def test_zip_read_product_attrs_and_optional_context_schema(patched_cache_env):
    handle = load_index(resolve_source(ENV_ZIP))
    prod_rel = pick_product_relpath(handle.manifest)
    if prod_rel is None:
        pytest.skip("No product .h5 present in manifest")
    prod_path = ensure_local_file(handle, prod_rel)

    attrs = peek_h5_attrs(prod_path)
    assert isinstance(attrs, dict)

    expected_any = {"product_prefix", "shape_T_H_W_C", "dataset_version", "radar_site"}
    assert any(k in attrs for k in expected_any)

    ctx_rel = attrs.get("context_relpath")
    if ctx_rel:
        ctx_path = (prod_path.parent / str(ctx_rel)).resolve()
        schema = read_schema(ctx_path)
        if schema is not None:
            assert "columns" in schema

@pytest.mark.skipif(not ENV_ZIP, reason="HRSS_TEST_ZIP not set")
def test_zip_missing_relpath_raises(patched_cache_env):
    handle = load_index(resolve_source(ENV_ZIP))
    with pytest.raises(HRSSIOError):
        ensure_local_file(handle, "KFAKE/storm_999/fake_thing.h5")

@pytest.mark.skipif(not ENV_ZIP or not ENV_MD5_ARCH, reason="Need zip and archive MD5 to test failure")
def test_zip_bad_archive_md5_raises(patched_cache_env):
    with pytest.raises(HRSSIntegrityError):
        resolve_source(ENV_ZIP, expected_archive_md5="deadbeef" * 4)

# -----------------------------
# Optional: Zenodo E2E (skipped unless DOI is provided)
# -----------------------------
@pytest.mark.skipif(not ENV_ZEN_DOI, reason="HRSS_TEST_ZENODO_DOI not set")
def test_zenodo_resolve_download_and_extract(patched_cache_env):
    # Note: this will download the largest .zip once into temp HRSS_CACHE
    handle = resolve_source(
        ENV_ZEN_DOI,
        expected_manifest_md5=ENV_MD5_MAN,   # optional
        expected_catalog_md5=ENV_MD5_CAT,    # optional
        token=ENV_ZEN_TOKEN,                 # only if needed
    )
    assert handle.kind == "zip"
    handle = load_index(handle)
    rel = pick_smallest_relpath(handle.manifest)
    local = ensure_local_file(handle, rel)
    assert local.exists()
