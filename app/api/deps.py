"""Shared FastAPI dependencies.

Anything injected via `Depends(...)` in more than one router lives here, so
routers import from one place rather than reaching into pipeline internals
directly.
"""

from ..pipeline.identity import get_user_id
from ..pipeline.upload_limiter import acquire as acquire_upload_slot

__all__ = ["get_user_id", "acquire_upload_slot"]
