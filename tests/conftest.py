import os, shutil, glob, tempfile, getpass, logging
from pathlib import Path

import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--hrss-keep-cache",
        action="store_true",
        help="Keep HRSS test caches (skip cleanup)."
    )

def pytest_sessionfinish(session, exitstatus):
    # Opt-out via flag or env
    if session.config.getoption("--hrss-keep-cache") or os.getenv("HRSS_TEST_KEEP_CACHE"):
        logging.getLogger("hrss.io").info("Keeping HRSS test caches (flag/env set).")
        return

    # Only delete caches created under pytest's temp roots
    tmp_root = Path(tempfile.gettempdir()) / f"pytest-of-{getpass.getuser()}"
    if not tmp_root.exists():
        return

    removed_any = False
    for p in tmp_root.rglob("hrss_cache"):
        # extra safety: only remove dirs named exactly hrss_cache
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            logging.getLogger("hrss.io").debug("Removed test cache: %s", p)
            removed_any = True

    if not removed_any:
        logging.getLogger("hrss.io").debug("No hrss_cache directories found under %s", tmp_root)
