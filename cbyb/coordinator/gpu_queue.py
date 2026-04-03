"""GPU request queue for serializing MLX evaluator access.

The evaluator pipeline is the only local GPU workload. This module provides
a simple FIFO lock so concurrent requests wait their turn rather than
competing for Apple Silicon GPU memory.
"""

import threading
import logging

logger = logging.getLogger(__name__)

# Module-level state — shared across all requests in the Flask process.
_lock = threading.Lock()
_waiting = 0           # number of requests queued (not yet holding the lock)
_waiting_lock = threading.Lock()  # guards the counter


def queue_depth() -> int:
    """Return the number of requests currently waiting for the GPU."""
    with _waiting_lock:
        return _waiting


def acquire(max_depth: int = 3) -> bool:
    """Try to enter the GPU queue.

    If the queue is already at *max_depth*, returns False immediately
    (caller should reject the request).  Otherwise increments the
    waiting counter and blocks until the lock is available.

    Returns True once the lock is held.  Caller MUST call release()
    when done.
    """
    global _waiting

    with _waiting_lock:
        if _waiting >= max_depth:
            return False
        _waiting += 1

    logger.info("GPU queue: waiting (%d in queue)", queue_depth())
    _lock.acquire()

    with _waiting_lock:
        _waiting -= 1

    logger.info("GPU queue: acquired (depth now %d)", queue_depth())
    return True


def release():
    """Release the GPU lock."""
    try:
        _lock.release()
        logger.info("GPU queue: released")
    except RuntimeError:
        pass  # already released — defensive
