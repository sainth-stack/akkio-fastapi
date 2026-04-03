"""Akkio API subset: multi-model workspace + usage (sidebar)."""
from fastapi import APIRouter

from .multi_model_api import router as multi_model_router
from .usage_api import usage_router

akkio_router = APIRouter()
akkio_router.include_router(multi_model_router)
akkio_router.include_router(usage_router)
