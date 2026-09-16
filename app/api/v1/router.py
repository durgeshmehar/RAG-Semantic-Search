"""Aggregates every v1 route so app/main.py mounts one router, not several."""

from fastapi import APIRouter

from . import search, upload

router = APIRouter()
router.include_router(upload.router)
router.include_router(search.router)
