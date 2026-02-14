from fastapi import APIRouter
from .upload_api import upload_router
from .explore_api import explore_router
from .user_api import user_router
from .database_chat import database_chat_router
from .image_classification import image_classification_router
from .multi_model_api import router as multi_model_router
from .usage_api import usage_router
from .settings_api import settings_router
from .sap_api import sap_router

akkio_router = APIRouter()
akkio_router.include_router(upload_router)
akkio_router.include_router(explore_router)
akkio_router.include_router(user_router)
akkio_router.include_router(database_chat_router)
akkio_router.include_router(image_classification_router)
akkio_router.include_router(multi_model_router)
akkio_router.include_router(usage_router)
akkio_router.include_router(settings_router)
akkio_router.include_router(sap_router)


