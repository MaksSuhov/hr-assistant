import secrets
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select

from hr_assistant.api.dependencies import get_container
from hr_assistant.application.auth import authenticated_user
from hr_assistant.infrastructure.db.models import UserRecord

router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(min_length=2, max_length=200)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def _set_session(response: Response, container, user: UserRecord, refresh_token: str) -> None:
    settings = container.settings
    common = {"secure": settings.auth_cookie_secure, "samesite": "lax", "path": "/"}
    response.set_cookie(
        "access_token",
        container.auth.access_token(user),
        httponly=True,
        max_age=settings.access_token_minutes * 60,
        **common,
    )
    response.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        max_age=settings.refresh_token_days * 86400,
        **common,
    )
    response.set_cookie(
        "csrf_token",
        secrets.token_urlsafe(32),
        httponly=False,
        max_age=settings.refresh_token_days * 86400,
        **common,
    )


def _clear_session(response: Response, secure: bool) -> None:
    for name in ("access_token", "refresh_token", "csrf_token"):
        response.delete_cookie(name, path="/", secure=secure, samesite="lax")


@router.post("/register", status_code=201)
async def register(data: RegisterRequest, request: Request, response: Response):
    container = get_container(request)
    user = await container.auth.register(data.email, data.password, data.full_name)
    _set_session(response, container, user, await container.auth.issue_refresh(user))
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}


@router.post("/login")
async def login(data: LoginRequest, request: Request, response: Response):
    container = get_container(request)
    user = await container.auth.authenticate(data.email, data.password)
    _set_session(response, container, user, await container.auth.issue_refresh(user))
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}


@router.post("/refresh")
async def refresh(request: Request, response: Response):
    container = get_container(request)
    raw = request.cookies.get("refresh_token")
    if not raw:
        raise HTTPException(401, "Refresh token is missing.")
    csrf_cookie = request.cookies.get("csrf_token", "")
    if not csrf_cookie or csrf_cookie != request.headers.get("x-csrf-token", ""):
        raise HTTPException(403, "CSRF token is missing or invalid.")
    user, next_token = await container.auth.rotate_refresh(raw)
    _set_session(response, container, user, next_token)
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response):
    container = get_container(request)
    csrf_cookie = request.cookies.get("csrf_token", "")
    if csrf_cookie and csrf_cookie != request.headers.get("x-csrf-token", ""):
        raise HTTPException(403, "CSRF token is invalid.")
    await container.auth.revoke_refresh(request.cookies.get("refresh_token"))
    _clear_session(response, container.settings.auth_cookie_secure)
    response.status_code = 204
    return response


@router.get("/me")
async def me(user: Annotated[UserRecord, Depends(authenticated_user)]):
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}


@router.post("/users", status_code=201)
async def create_recruiter(
    data: RegisterRequest,
    request: Request,
    current_user: Annotated[UserRecord, Depends(authenticated_user)],
):
    if current_user.role != "admin":
        raise HTTPException(403, "Administrator role required.")
    user = await get_container(request).auth.provision_recruiter(
        data.email, data.password, data.full_name
    )
    return {"id": user.id, "email": user.email, "full_name": user.full_name, "role": user.role}


@router.get("/oauth/google")
async def google_login(request: Request):
    settings = get_container(request).settings
    if not settings.oauth_google_client_id or not settings.oauth_google_redirect_uri:
        raise HTTPException(503, "Google OAuth is not configured.")
    state = secrets.token_urlsafe(32)
    response = RedirectResponse(
        "https://accounts.google.com/o/oauth2/v2/auth?"
        + urlencode(
            {
                "client_id": settings.oauth_google_client_id,
                "redirect_uri": settings.oauth_google_redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "prompt": "select_account",
            }
        )
    )
    response.set_cookie(
        "oauth_state",
        state,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        max_age=600,
        path="/",
    )
    return response


@router.get("/oauth/google/callback")
async def google_callback(request: Request, code: str, state: str):
    container = get_container(request)
    settings = container.settings
    if not secrets.compare_digest(state, request.cookies.get("oauth_state", "")):
        raise HTTPException(400, "OAuth state validation failed.")
    if (
        not settings.oauth_google_client_id
        or not settings.oauth_google_client_secret
        or not settings.oauth_google_redirect_uri
    ):
        raise HTTPException(503, "Google OAuth is not configured.")
    async with httpx.AsyncClient(timeout=10) as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.oauth_google_client_id,
                "client_secret": settings.oauth_google_client_secret.get_secret_value(),
                "redirect_uri": settings.oauth_google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_response.is_error:
            raise HTTPException(400, "Google authorization could not be completed.")
        access_token = token_response.json().get("access_token")
        profile_response = await client.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if profile_response.is_error:
            raise HTTPException(400, "Google profile could not be loaded.")
        profile = profile_response.json()
    subject, email = profile.get("sub"), profile.get("email", "").lower()
    if not subject or not email or not profile.get("email_verified"):
        raise HTTPException(400, "Google account email is not verified.")
    async with container.database.session() as session:
        user = await session.scalar(
            select(UserRecord).where(
                UserRecord.oauth_provider == "google", UserRecord.oauth_subject == subject
            )
        )
        if not user:
            user = await session.scalar(select(UserRecord).where(UserRecord.email == email))
            if user:
                user.oauth_provider, user.oauth_subject = "google", subject
            else:
                users_exist = await session.scalar(select(func.count()).select_from(UserRecord))
                user = UserRecord(
                    id=secrets.token_hex(16),
                    email=email,
                    full_name=profile.get("name") or email.split("@")[0],
                    oauth_provider="google",
                    oauth_subject=subject,
                    role="recruiter" if users_exist else "admin",
                )
                session.add(user)
        await session.commit()
        await session.refresh(user)
    if not user.is_active:
        raise HTTPException(403, "Account is disabled.")
    response = RedirectResponse(settings.oauth_frontend_url)
    response.delete_cookie("oauth_state", path="/")
    _set_session(response, container, user, await container.auth.issue_refresh(user))
    return response
