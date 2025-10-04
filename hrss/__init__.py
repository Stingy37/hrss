from importlib.metadata import version as _version, PackageNotFoundError as _PNF
import logging as _logging
from typing import TYPE_CHECKING as _TC

# ---- version ----
try:
    __version__ = _version("hrss")
except _PNF:
    __version__ = "0.0.0"

# ---- logging (library best practice) ----
# don't force logging onto user, let user choose 
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# ---- public API (documented) ----
#     - HRSSDataset ->        main abstraction for dataset operations (construction, indexing, .asarray(), etc.)
#     - HRSSIOError ->        explicitly catch IO errors
#     - HRSSIntegrityError -> explicitly catch integrity (checksums w/ sha256 + manifest) errors
#     - load ->               one line quickstart building and returning HRSSDataset
_PUBLIC = {"HRSSDataset", "HRSSIOError", "HRSSIntegrityError", "load"}
__all__ = sorted(_PUBLIC | {"__version__"})

# ---- type-checker visibility without runtime imports ----
# convinent for IDEs, etc. b/c allows them to "see" modules without actually importing 
#     - works together with lazy-loading (dont import unless user needs it, but now IDE typing works)
if _TC:
    from .loader import HRSSDataset, HRSSIOError, HRSSIntegrityError, load  

# ---- lazy re-exports to keep import fast ----
# delay importing specific attributes
#     - ex. once someone accesses hrss.load, then we import hrss.loader
def __getattr__(name: str):
    if name in _PUBLIC:
        from . import loader  # import on first access only
        obj = getattr(loader, name)
        globals()[name] = obj  # cache for subsequent lookups
        return obj
    raise AttributeError(name)


def __dir__():
    return sorted(set(globals()) | _PUBLIC)
