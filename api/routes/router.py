from fastapi import APIRouter

from api.routes import heartbeat, detector

router = APIRouter()
router.include_router(heartbeat.router, tags=["health"], prefix="/v1/health")
router.include_router(detector.router, tags=["detect"], prefix="/v1/process")


