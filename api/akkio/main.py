from fastapi import APIRouter
from .upload_api import upload_router
from .explore_api import explore_router
from .user_api import user_router
from .database_chat import database_chat_router

akkio_router = APIRouter()
akkio_router.include_router(upload_router)
akkio_router.include_router(explore_router)
akkio_router.include_router(user_router)
akkio_router.include_router(database_chat_router)


