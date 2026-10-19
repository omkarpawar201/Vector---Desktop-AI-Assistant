"""
Cancellation primitives for Vector's streaming generation pipeline.

A single :class:`CancellationToken` is created per generation request by the UI
and threaded through the background worker, the router and the provider client.
Every layer only ever *reads* the token (via :meth:`CancellationToken.is_cancelled`),
so cancellation never blocks: the UI flips the flag and each layer stops at its
next safe checkpoint.
"""

import threading
from typing import Optional


class CancellationToken:
    """
    A thread-safe, one-shot cancellation flag shared across one generation.

    ``cancel()`` may be called from any thread (the Qt UI thread handles the
    Stop control) while the provider loop polls :meth:`is_cancelled` on its own
    worker thread.  A token is bound to exactly one generation, so cancelling it
    can never affect a later generation: each request gets its own token.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation.  Idempotent and safe to call from any thread."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """True once cancellation has been requested."""
        return self._event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until cancelled or ``timeout`` elapses; returns the current state."""
        return self._event.wait(timeout)

    def reset(self) -> None:
        """Clear the flag (used only by tests that deliberately reuse a token)."""
        self._event.clear()
