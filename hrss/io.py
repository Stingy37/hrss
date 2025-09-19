# hrss/io.py
from __future__ import annotations

import csv
import dataclasses as dc
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

try:
    import zipfile
except Exception as e:  # pragma: no cover
    raise

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pandas is required for hrss.io (for reading manifest/catalog). "
        "Install with `pip install pandas`."
    ) from e

# Optional dependency; only needed for DOI/URL fetching
try:
    import requests  # noqa: F401
except Exception:
    requests = None  # We guard before use.

log = logging.getLogger("hrss.io")
log.addHandler(logging.NullHandler())

# Canonical HRSS filename pattern (matches your writer)
_FNAME_RE = re.compile(
    r"^(?P<site>[A-Z0-9]{4})_(?P<storm>\d+?)_"
    r"(?P<kind>context|[a-zA-Z0-9_]+?)"
    r"(?:_T(?P<T>\d+))?"
    r"(?:_(?P<H>\d+)x(?P<W>\d+)x(?P<C>\d+)ch)?"
    r"_(?P<t0>\d{8}T\d{6}Z)_(?P<t1>\d{8}T\d{6}Z)\.h5$"
)

# ----------------------------- Exceptions -----------------------------

class HRSSIOError(Exception):
    """Generic I/O/plumbing error."""


class HRSSIntegrityError(HRSSIOError):
    """Checksum or integrity failure."""


# ----------------------------- Data classes -----------------------------

@dc.dataclass(slots=True)
class SourceHandle:
    """
    Opaque handle returned by resolve_source(). It normalizes all sources so the
    rest of the package can treat them uniformly.

    Attributes
    ----------
    kind : {"dir","zip"}
        After resolution we only distinguish between a directory tree and a local zip.
        (Zenodo/DOI is downloaded -> local zip.)
    year_root : Path
        Local filesystem directory that *contains* manifest.csv & catalog.csv,
        or a synthetic placeholder for zips (see `zip_year_root_in_archive`).
    archive_path : Optional[Path]
        Path to the local zip archive if kind=="zip", else None.
    cache_dir : Path
        Directory where on-demand extractions are written, mirroring the relpaths from manifest.
    zip_year_root_in_archive : Optional[str]
        For zips: the internal directory inside the archive that corresponds to `year_root`.
        e.g., "hrss-starter-2017/2017"
    index_loaded : bool
        Whether manifest/catalog have been loaded and parsed.
    manifest : Optional[pd.DataFrame]
    catalog : Optional[pd.DataFrame]
    expected_archive_md5 : Optional[str]
    expected_manifest_md5 : Optional[str]
    expected_catalog_md5 : Optional[str]
    """
    kind: str  # "dir" | "zip"
    year_root: Path
    archive_path: Optional[Path]
    cache_dir: Path
    zip_year_root_in_archive: Optional[str] = None
    index_loaded: bool = False
    manifest: Optional[pd.DataFrame] = None
    catalog: Optional[pd.DataFrame] = None
    expected_archive_md5: Optional[str] = None
    expected_manifest_md5: Optional[str] = None
    expected_catalog_md5: Optional[str] = None

    # fast lookup map: relpath -> sha256 (filled in load_index)
    _sha_by_rel: Optional[Dict[str, str]] = None

    def manifest_row_for(self, relpath: str) -> Optional[pd.Series]:
        if self.manifest is None:
            return None
        m = self.manifest
        # manifest relpath uses forward slashes regardless of OS
        rel_norm = str(Path(relpath).as_posix())
        hits = m.loc[m["relpath"] == rel_norm]
        if len(hits) == 0:
            return None
        return hits.iloc[0]

    def file_sha256(self, relpath: str) -> Optional[str]:
        if self._sha_by_rel is None:
            return None
        # normalize key as posix
        key = str(Path(relpath).as_posix())
        return self._sha_by_rel.get(key)


# ----------------------------- Public API -----------------------------

def resolve_source(
    source: Union[str, os.PathLike],
    *,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    expected_archive_md5: Optional[str] = None,
    expected_manifest_md5: Optional[str] = None,
    expected_catalog_md5: Optional[str] = None,
    zenodo_host: str = "zenodo.org",
    sandbox_host: str = "sandbox.zenodo.org",
    token: Optional[str] = None,
) -> SourceHandle:
    """
    Normalize a user-provided source into a SourceHandle.

    Parameters
    ----------
    source
        - Path to a local *year directory* (contains manifest.csv/catalog.csv under it), OR
        - Path to a local .zip (your Zenodo bundle), OR
        - A Zenodo DOI like "10.5281/zenodo.1234567" (prod) or "10.5072/zenodo.1234567" (sandbox), OR
        - A Zenodo record URL like "https://zenodo.org/records/1234567".
    cache_dir
        Where to extract files from zip/doi; defaults to ~/.cache/hrss/<archive_key>/.
    expected_*_md5
        Optional MD5s for the archive (zip) and the manifest/catalog CSVs. If provided,
        they'll be checked and raise HRSSIntegrityError on mismatch.
    zenodo_host, sandbox_host
        Hosts for public and sandbox Zenodo. Used to form API endpoints when `source` is a DOI/URL.
    token
        Optional Zenodo access token for private/embargoed records (as dataset creator).

    Returns
    -------
    SourceHandle

    Notes
    -----
    - If `source` is a DOI/record URL, this will *download the zip once* into cache.
    - For a *local directory*, we treat it as the `year_root` (the directory that contains manifest/catalog).
    """
    src = str(source).strip()

    # 1) Local directory?
    p = Path(src)
    if p.exists() and p.is_dir():
        year_root = _ensure_year_root_dir(p)
        _check_top_level_csv_md5s(year_root, expected_manifest_md5, expected_catalog_md5)
        return SourceHandle(
            kind="dir",
            year_root=year_root,
            archive_path=None,
            cache_dir=_default_cache_dir(cache_dir, archive_key=f"localdir-{_safe_key(str(year_root))}"),
            expected_archive_md5=None,
            expected_manifest_md5=expected_manifest_md5,
            expected_catalog_md5=expected_catalog_md5,
        )

    # 2) Local zip?
    if p.exists() and p.is_file() and p.suffix.lower() == ".zip":
        _check_archive_md5(p, expected_archive_md5)
        # For zips we don't have a real year_root yet; we store a placeholder
        # and record where the year_root lives *inside* the archive when we load the index.
        handle = SourceHandle(
            kind="zip",
            year_root=Path("<zip>"),
            archive_path=p,
            cache_dir=_default_cache_dir(cache_dir, archive_key=_archive_key_for_zip(p)),
            expected_archive_md5=expected_archive_md5,
            expected_manifest_md5=expected_manifest_md5,
            expected_catalog_md5=expected_catalog_md5,
        )
        return handle

    # 3) Zenodo DOI/URL -> download zip
    if _looks_like_zenodo(src):
        if requests is None:
            raise HRSSIOError("requests is required to open DOI/URL sources. Install with `pip install requests`.")
        archive_path = _download_zenodo_zip_to_cache(
            src,
            cache_dir=_default_cache_dir(cache_dir, archive_key="zenodo"),
            zenodo_host=zenodo_host,
            sandbox_host=sandbox_host,
            token=token,
        )
        _check_archive_md5(archive_path, expected_archive_md5)
        return SourceHandle(
            kind="zip",
            year_root=Path("<zip>"),
            archive_path=archive_path,
            cache_dir=_default_cache_dir(cache_dir, archive_key=_archive_key_for_zip(archive_path)),
            expected_archive_md5=expected_archive_md5,
            expected_manifest_md5=expected_manifest_md5,
            expected_catalog_md5=expected_catalog_md5,
        )

    raise HRSSIOError(
        f"Unsupported source: {source!r}. "
        "Provide a local year directory, a .zip file, or a Zenodo DOI/record URL."
    )


def load_index(handle: SourceHandle) -> SourceHandle:
    """
    Load manifest.csv + catalog.csv into the handle (and find their location).
    Also builds a fast relpath->sha256 map for integrity checks.
    """
    if handle.kind == "dir":
        year_root = handle.year_root
        manifest_path = year_root / "manifest.csv"
        catalog_path = year_root / "catalog.csv"
        if not manifest_path.exists():
            raise HRSSIOError(f"manifest.csv not found under: {year_root}")
        if not catalog_path.exists():
            raise HRSSIOError(f"catalog.csv not found under: {year_root}")

        _check_top_level_csv_md5s(year_root, handle.expected_manifest_md5, handle.expected_catalog_md5)

        man = pd.read_csv(manifest_path)
        cat = pd.read_csv(catalog_path)

        _normalize_manifest(man)
        _normalize_catalog(cat)
        handle.manifest, handle.catalog = man, cat
        handle._sha_by_rel = dict(zip(man["relpath"].map(str), man["sha256"].map(str)))
        handle.index_loaded = True
        return handle

    # Zip mode: locate CSVs inside the zip and read them
    if handle.kind == "zip":
        assert handle.archive_path is not None
        with zipfile.ZipFile(handle.archive_path, "r") as zf:
            manifest_member = _find_member(zf, "manifest.csv")
            catalog_member = _find_member(zf, "catalog.csv")
            if manifest_member is None or catalog_member is None:
                raise HRSSIOError(
                    f"Could not find manifest/catalog inside zip: {handle.archive_path}"
                )

            # Derive the archive's "year root" (the parent directory of manifest.csv)
            yr_root = Path(manifest_member).parent  # e.g., hrss-starter-2017/2017
            handle.zip_year_root_in_archive = str(yr_root).replace("\\", "/")

            # Integrity (optional MD5 for the CSV entries themselves)
            _check_zip_member_md5(zf, manifest_member, handle.expected_manifest_md5)
            _check_zip_member_md5(zf, catalog_member, handle.expected_catalog_md5)

            # Load CSVs directly from the zip stream
            with zf.open(manifest_member) as f:
                man = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
            with zf.open(catalog_member) as f:
                cat = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))

        _normalize_manifest(man)
        _normalize_catalog(cat)
        handle.manifest, handle.catalog = man, cat
        handle._sha_by_rel = dict(zip(man["relpath"].map(str), man["sha256"].map(str)))
        handle.index_loaded = True
        return handle

    raise HRSSIOError(f"Unknown handle.kind: {handle.kind}")


def ensure_local_file(
    handle: SourceHandle,
    relpath: Union[str, os.PathLike],
    *,
    verify: bool = True,
) -> Path:
    """
    Ensure that a given `relpath` (as listed in manifest.csv) exists on the local filesystem,
    and return its absolute Path. For directory sources, this is a no-op; for zips, it
    extracts exactly that file into the cache (mirroring the relpath under the cache dir).

    Parameters
    ----------
    handle : SourceHandle
        From resolve_source() + load_index().
    relpath : str | Path
        Path as recorded in manifest.csv (POSIX style); e.g. "KUEX/storm_123/....h5"
    verify : bool
        If True, will verify the file's sha256 against the manifest (fast path).
        If the manifest row is missing a sha256, verification is skipped.

    Returns
    -------
    Path : absolute local path to the file
    """
    if not handle.index_loaded:
        raise HRSSIOError("Call load_index(handle) before ensure_local_file().")

    rel = Path(relpath).as_posix()

    if handle.kind == "dir":
        p = handle.year_root / rel
        if not p.exists():
            raise HRSSIOError(f"File not found under local directory: {p}")
        if verify:
            _verify_sha256_against_manifest(handle, p, rel)
        return p.resolve()

    if handle.kind == "zip":
        assert handle.archive_path is not None, "zip handle missing archive_path"
        if handle.zip_year_root_in_archive is None:
            raise HRSSIOError("Zip year root not set. Did you call load_index(handle)?")

        # Where inside the zip?
        arc_rel = f"{handle.zip_year_root_in_archive}/{rel}".replace("\\", "/")

        # Where in cache should we place it?
        dest = (Path(handle.cache_dir) / rel).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            # already extracted; verify hash if requested
            if verify:
                _verify_sha256_against_manifest(handle, dest, rel)
            return dest

        # Extract just this file atomically
        with zipfile.ZipFile(handle.archive_path, "r") as zf:
            try:
                info = zf.getinfo(arc_rel)
            except KeyError:
                raise HRSSIOError(f"Relpath not found in zip: {arc_rel}") from None
            tmp = dest.with_suffix(dest.suffix + ".partial")
            with zf.open(info, "r") as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            os.replace(tmp, dest)

        if verify:
            _verify_sha256_against_manifest(handle, dest, rel)

        return dest

    raise HRSSIOError(f"Unknown handle.kind: {handle.kind}")


def read_schema(local_context_path: Union[str, os.PathLike]) -> Optional[dict]:
    """
    Read a context schema JSON given a local context .h5 path.

    Returns dict or None if the sidecar isn't found.
    """
    p = Path(local_context_path)
    schema_path = p.with_suffix(".schema.json")
    if not schema_path.exists():
        return None
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HRSSIOError(f"Failed to read schema: {schema_path}") from e


def peek_h5_attrs(local_h5_path: Union[str, os.PathLike]) -> Dict[str, object]:
    """
    Open an HDF5 file and return only its root-attribute dict (no /data reads).
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise HRSSIOError("h5py is required for peek_h5_attrs(). Install with `pip install h5py`.") from e

    attrs: Dict[str, object] = {}
    p = Path(local_h5_path)
    if not p.exists():
        raise HRSSIOError(f"HDF5 file not found: {p}")
    try:
        with h5py.File(p, "r") as h5:
            for k, v in h5.attrs.items():
                attrs[str(k)] = _pretty_attr(v)
    except Exception as e:
        raise HRSSIOError(f"Failed to open HDF5 attrs: {p}") from e
    return attrs


# ----------------------------- Helpers -----------------------------

def _pretty_attr(val):
    # mirror your writer's humanization to keep things readable
    import numpy as _np
    try:
        if isinstance(val, (bytes, _np.bytes_)):
            return val.decode("utf-8", errors="replace")
        if isinstance(val, (_np.bool_, _np.integer, _np.floating)):
            return val.item()
        if isinstance(val, _np.ndarray):
            if val.ndim == 0:
                return val.item()
            if val.size > 16:
                return f"array(shape={val.shape}, dtype={val.dtype})"
            return _np.array2string(val, threshold=16)
        return val
    except Exception:
        return str(val)


def _default_cache_dir(cache_dir: Optional[Union[str, os.PathLike]], *, archive_key: str) -> Path:
    base = Path(cache_dir) if cache_dir else Path(os.environ.get("HRSS_CACHE", Path.home() / ".cache" / "hrss"))
    return (base / _safe_key(archive_key)).resolve()


def _safe_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def _archive_key_for_zip(zip_path: Path) -> str:
    # Use name + size + mtime for a quick unique key; you can switch to sha256 if desired.
    try:
        st = zip_path.stat()
        return f"zip-{zip_path.name}-{st.st_size}-{int(st.st_mtime)}"
    except Exception:
        return f"zip-{zip_path.name}"


def _ensure_year_root_dir(p: Path) -> Path:
    """
    Normalize a given directory to the actual 'year root' (the dir that contains manifest.csv/catalog.csv).
    Accept either directly the year directory, or a parent that contains exactly one year dir with manifest/catalog.
    """
    # direct?
    if (p / "manifest.csv").exists() and (p / "catalog.csv").exists():
        return p.resolve()

    # try to auto-detect a single child year dir that has manifest/catalog
    candidates = [c for c in p.iterdir() if c.is_dir() and (c / "manifest.csv").exists() and (c / "catalog.csv").exists()]
    if len(candidates) == 1:
        return candidates[0].resolve()

    raise HRSSIOError(
        f"Could not find manifest.csv & catalog.csv under: {p}\n"
        "Pass the year directory that contains those files."
    )


def _looks_like_zenodo(s: str) -> bool:
    if s.startswith("10."):
        return "zenodo." in s
    u = urlparse(s)
    return "zenodo.org" in (u.netloc or "") or "sandbox.zenodo.org" in (u.netloc or "")


def _download_zenodo_zip_to_cache(
    doi_or_url: str,
    cache_dir: Path,
    *,
    zenodo_host: str,
    sandbox_host: str,
    token: Optional[str],
) -> Path:
    """
    Resolve a Zenodo DOI/URL to a record, pick the largest .zip file,
    and download it into <cache_dir>/archives/<record_id>/<filename>.zip.

    Notes
    -----
    - We purposely keep this simple and robust:
      * DOI of the form 10.5281/zenodo.<id> or 10.5072/zenodo.<id> (sandbox)
      * Record URL of the form https://<host>/records/<id>
    - Requires `requests`. Raises HRSSIOError on errors.
    """
    if requests is None:
        raise HRSSIOError("requests is required to download from Zenodo. Install with `pip install requests`.")

    rec_id, host = _zenodo_record_id_and_host(doi_or_url, zenodo_host, sandbox_host)
    api_url = f"https://{host}/api/records/{rec_id}"

    headers = {}
    params = {}
    if token:
        # Zenodo accepts token via query or header; we use header.
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = requests.get(api_url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        meta = r.json()
    except Exception as e:
        raise HRSSIOError(f"Failed to query Zenodo API for record {rec_id} at {host}") from e

    files = meta.get("files") or meta.get("hits", {}).get("hits", [{}])[0].get("files")  # be tolerant
    if not files:
        raise HRSSIOError(f"No files listed for Zenodo record {rec_id} at {host}")

    # pick the largest .zip
    zips = [f for f in files if str(f.get('key', '')).lower().endswith(".zip")]
    if not zips:
        # fallback: pick the largest file
        zips = files
    zips.sort(key=lambda f: int(f.get("size", 0)), reverse=True)
    best = zips[0]
    filename = best.get("key") or best.get("filename") or f"{rec_id}.zip"

    # Build a direct download URL
    link = None
    # new records: get 'links': {'self': ... , 'download': ...}
    if "links" in best and "download" in best["links"]:
        link = best["links"]["download"]
    elif "links" in meta and "latest" in meta["links"]:
        # fallback to record download link pattern
        link = f"https://{host}/records/{rec_id}/files/{filename}?download=1"
    else:
        link = f"https://{host}/records/{rec_id}/files/{filename}?download=1"

    dst_dir = cache_dir / "archives" / str(rec_id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename

    if dst.exists() and dst.stat().st_size == int(best.get("size", dst.stat().st_size or 0)):
        log.debug("Zenodo zip already cached: %s", dst)
        return dst.resolve()

    try:
        with requests.get(link, headers=headers, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            tmp = dst.with_suffix(".partial")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dst)
    except Exception as e:
        raise HRSSIOError(f"Failed to download Zenodo asset: {link}") from e

    return dst.resolve()


def _zenodo_record_id_and_host(doi_or_url: str, prod_host: str, sandbox_host: str) -> Tuple[int, str]:
    s = doi_or_url.strip()
    # DOI forms
    m = re.search(r"10\.(?:5281|5072)/zenodo\.(\d+)", s)
    if m:
        rec_id = int(m.group(1))
        host = prod_host if "5281" in s else sandbox_host
        return rec_id, host

    # URL forms
    u = urlparse(s)
    if u.netloc in {prod_host, sandbox_host} and u.path:
        m = re.search(r"/records/(\d+)", u.path)
        if m:
            return int(m.group(1)), u.netloc

    raise HRSSIOError(f"Could not parse Zenodo record id from: {doi_or_url}")


def _find_member(zf: zipfile.ZipFile, filename: str) -> Optional[str]:
    """
    Find a member path in the zip whose basename matches `filename`.
    Returns the first match (deep search).
    """
    fname = filename.lower()
    for n in zf.namelist():
        if n.lower().endswith("/" + fname) or n.lower() == fname:
            return n
    return None


def _normalize_manifest(df: pd.DataFrame) -> None:
    required = {"relpath", "file_type", "size_bytes", "sha256", "site", "storm_id", "kind"}
    missing = required - set(df.columns)
    if missing:
        raise HRSSIOError(f"manifest.csv missing columns: {sorted(missing)}")
    # Ensure posix-style relpaths for cross-platform consistency
    df["relpath"] = df["relpath"].astype(str).str.replace("\\", "/", regex=False)


def _normalize_catalog(df: pd.DataFrame) -> None:
    required = {"site", "storm_id", "storm_dir", "t0_utc", "t1_utc", "T", "n_products", "products", "dims", "total_bytes"}
    missing = required - set(df.columns)
    if missing:
        raise HRSSIOError(f"catalog.csv missing columns: {sorted(missing)}")
    df["storm_dir"] = df["storm_dir"].astype(str).str.replace("\\", "/", regex=False)


def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _check_archive_md5(zip_path: Path, expected_md5: Optional[str]) -> None:
    if not expected_md5:
        return
    md5 = _file_md5(zip_path)
    if md5.lower() != expected_md5.lower():
        raise HRSSIntegrityError(
            f"Archive MD5 mismatch for {zip_path.name}: got {md5}, expected {expected_md5}"
        )


def _check_top_level_csv_md5s(year_root: Path, expected_manifest_md5: Optional[str], expected_catalog_md5: Optional[str]) -> None:
    if expected_manifest_md5:
        got = _file_md5(year_root / "manifest.csv")
        if got.lower() != expected_manifest_md5.lower():
            raise HRSSIntegrityError(f"manifest.csv MD5 mismatch: got {got}, expected {expected_manifest_md5}")
    if expected_catalog_md5:
        got = _file_md5(year_root / "catalog.csv")
        if got.lower() != expected_catalog_md5.lower():
            raise HRSSIntegrityError(f"catalog.csv MD5 mismatch: got {got}, expected {expected_catalog_md5}")


def _check_zip_member_md5(zf: zipfile.ZipFile, member: str, expected_md5: Optional[str]) -> None:
    if not expected_md5:
        return
    # Stream the member and compute MD5
    h = hashlib.md5()
    with zf.open(member, "r") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got.lower() != expected_md5.lower():
        raise HRSSIntegrityError(f"Zip entry MD5 mismatch for {member}: got {got}, expected {expected_md5}")


def _verify_sha256_against_manifest(handle: SourceHandle, local_path: Path, relpath: str) -> None:
    expected = handle.file_sha256(relpath)
    if not expected:
        log.debug("No sha256 in manifest for %s; skipping verification", relpath)
        return
    got = _file_sha256(local_path)
    if got.lower() != expected.lower():
        raise HRSSIntegrityError(
            f"sha256 mismatch for {relpath}: got {got}, expected {expected} "
            f"({local_path})"
        )
