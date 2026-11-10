from fastapi import APIRouter

from hr_assistant.api.routes.approvals import router as approvals_router
from hr_assistant.api.routes.auth import router as auth_router
from hr_assistant.api.routes.candidates import router as candidates_router
from hr_assistant.api.routes.chat import router as chat_router
from hr_assistant.api.routes.health import router as health_router
from hr_assistant.api.routes.maintenance import router as maintenance_router
from hr_assistant.api.routes.resumes import hr_router as resumes_router
from hr_assistant.api.routes.resumes import router as public_resumes_router
from hr_assistant.api.routes.vacancies import router as vacancies_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(health_router)
api_router.include_router(vacancies_router, prefix="/vacancies", tags=["vacancies"])
api_router.include_router(resumes_router, prefix="/resumes", tags=["resumes"])
api_router.include_router(public_resumes_router, prefix="/resumes", tags=["resumes"])
api_router.include_router(candidates_router, prefix="/candidates", tags=["candidates"])
api_router.include_router(approvals_router, prefix="/approvals", tags=["approvals"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(maintenance_router, prefix="/maintenance", tags=["maintenance"])
