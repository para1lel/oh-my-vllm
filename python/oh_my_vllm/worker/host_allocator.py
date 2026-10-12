"""Optional glibc allocator observations, separate from complete process RSS."""

import ctypes
import sys
import time
from functools import lru_cache

FIELDS = (
    "arena",
    "ordblks",
    "smblks",
    "hblks",
    "hblkhd",
    "usmblks",
    "fsmblks",
    "uordblks",
    "fordblks",
    "keepcost",
)


class Mallinfo2(ctypes.Structure):
    """glibc malloc.h declares these ten fields as size_t, in this order."""

    _fields_ = [(name, ctypes.c_size_t) for name in FIELDS]


@lru_cache(maxsize=1)
def allocator():
    """Keep the library owner alive, without loading an alternative allocator."""
    if sys.platform != "linux":
        return None, None
    try:
        library = ctypes.CDLL(None)
        version = library.gnu_get_libc_version
        version.argtypes, version.restype = [], ctypes.c_char_p
        text = version().decode("ascii")
        query = library.mallinfo2
        query.argtypes, query.restype = [], Mallinfo2
        return library, text
    except (AttributeError, OSError):
        return None, None


def host_allocator_snapshot() -> dict:
    """Observe allocator storage. Thread caches and other allocators limit scope."""
    started = time.perf_counter()
    library, version = allocator()
    if library is None:
        return {
            "available": False,
            "scope": "glibc_malloc_only",
            "reason": "glibc mallinfo2 is unavailable",
            "elapsed_seconds": time.perf_counter() - started,
        }
    values = library.mallinfo2()
    fields = {name: int(getattr(values, name)) for name in FIELDS}
    return {
        "available": True,
        "scope": "glibc_malloc_only",
        "libc_version": version,
        "fields": fields,
        "used_or_retained_bytes": fields["uordblks"] + fields["hblkhd"],
        "elapsed_seconds": time.perf_counter() - started,
    }
