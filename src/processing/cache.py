"""Coordination for first-time publication of shared dataset caches."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def cache_creation_lock(path):
    """Lock a missing cache; callers recheck existence before creating it.

    Existing cache reads take no lock. Keep independent loading and graph
    reconstruction outside this context. Published cache files must be atomic.
    """
    path = Path(path)
    if path.exists():
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def cache_access_lock(path, *, ready):
    """Allow concurrent opaque cache readers, excluding cold constructors.

    Third-party constructors may publish files in place. Check readiness under
    a shared lock so those partially written files cannot be mistaken for a
    warm cache while another constructor owns the exclusive lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
        try:
            if not ready():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                if ready():
                    fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
