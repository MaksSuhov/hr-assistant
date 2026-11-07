"""Cookie based access and rotating refresh token authentication."""

import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import HTTPException, Request, status
from sqlalchemy import func, select

from hr_assistant.core.config import Settings
from hr_assistant.infrastructure.db.database import Database
from hr_assistant.infrastructure.db.models import RefreshTokenRecord, UserRecord


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}${derived.hex()}"


def _password_matches(password: str, encoded: str | None) -> bool:
    if not encoded:
        return False
    try:
        salt_hex, digest = encoded.split("$", 1)
        derived = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(derived.hex(), digest)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class AuthService:
    def __init__(self, database: Database, settings: Settings) -> None:
        self.database = database
        self.settings = settings

    def access_token(self, user: UserRecord) -> str:
        now = datetime.now(UTC)
        return jwt.encode(
            {
                "sub": user.id,
                "email": user.email,
                "role": user.role,
                "iat": now,
                "exp": now + timedelta(minutes=self.settings.access_token_minutes),
                "type": "access",
            },
            self.settings.auth_secret_key.get_secret_value(),
            algorithm="HS256",
        )

    async def register(self, email: str, password: str, full_name: str) -> UserRecord:
        normalized_email = email.strip().lower()
        async with self.database.session() as session:
            user_count = await session.scalar(select(func.count()).select_from(UserRecord))
            if user_count:
                raise HTTPException(
                    403, "Registration is closed. Ask an administrator to provision your account."
                )
            if await session.scalar(select(UserRecord).where(UserRecord.email == normalized_email)):
                raise HTTPException(409, "An account with this email already exists.")
            user = UserRecord(
                id=str(uuid.uuid4()),
                email=normalized_email,
                full_name=full_name.strip(),
                password_hash=_password_hash(password),
                role="admin",
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def provision_recruiter(self, email: str, password: str, full_name: str) -> UserRecord:
        normalized_email = email.strip().lower()
        async with self.database.session() as session:
            if await session.scalar(select(UserRecord).where(UserRecord.email == normalized_email)):
                raise HTTPException(409, "An account with this email already exists.")
            user = UserRecord(
                id=str(uuid.uuid4()),
                email=normalized_email,
                full_name=full_name.strip(),
                password_hash=_password_hash(password),
                role="recruiter",
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def authenticate(self, email: str, password: str) -> UserRecord:
        async with self.database.session() as session:
            user = await session.scalar(
                select(UserRecord).where(UserRecord.email == email.strip().lower())
            )
            if (
                not user
                or not user.is_active
                or not _password_matches(password, user.password_hash)
            ):
                raise HTTPException(401, "Invalid email or password.")
            return user

    async def find_user(self, user_id: str) -> UserRecord | None:
        async with self.database.session() as session:
            return await session.get(UserRecord, user_id)

    async def issue_refresh(self, user: UserRecord) -> str:
        raw = secrets.token_urlsafe(48)
        async with self.database.session() as session:
            session.add(
                RefreshTokenRecord(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    token_hash=_token_hash(raw),
                    expires_at=datetime.now(UTC) + timedelta(days=self.settings.refresh_token_days),
                )
            )
            await session.commit()
        return raw

    async def rotate_refresh(self, raw: str) -> tuple[UserRecord, str]:
        async with self.database.session() as session:
            record = await session.scalar(
                select(RefreshTokenRecord).where(RefreshTokenRecord.token_hash == _token_hash(raw))
            )
            now = datetime.now(UTC)
            if not record or record.revoked_at or record.expires_at.replace(tzinfo=UTC) <= now:
                raise HTTPException(401, "Refresh token is invalid or expired.")
            user = await session.get(UserRecord, record.user_id)
            if not user or not user.is_active:
                raise HTTPException(401, "Account is unavailable.")
            record.revoked_at = now
            next_raw = secrets.token_urlsafe(48)
            session.add(
                RefreshTokenRecord(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    token_hash=_token_hash(next_raw),
                    expires_at=now + timedelta(days=self.settings.refresh_token_days),
                )
            )
            await session.commit()
            return user, next_raw

    async def revoke_refresh(self, raw: str | None) -> None:
        if not raw:
            return
        async with self.database.session() as session:
            record = await session.scalar(
                select(RefreshTokenRecord).where(RefreshTokenRecord.token_hash == _token_hash(raw))
            )
            if record and not record.revoked_at:
                record.revoked_at = datetime.now(UTC)
                await session.commit()


async def authenticated_user(request: Request) -> UserRecord:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required.")
    try:
        payload = jwt.decode(
            token,
            request.app.state.container.settings.auth_secret_key.get_secret_value(),
            algorithms=["HS256"],
            options={"require": ["sub", "exp", "type"]},
        )
        if payload.get("type") != "access":
            raise jwt.InvalidTokenError("wrong token type")
    except jwt.InvalidTokenError as error:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Access token is invalid or expired."
        ) from error
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        csrf_cookie = request.cookies.get("csrf_token", "")
        csrf_header = request.headers.get("x-csrf-token", "")
        if not csrf_cookie or not hmac.compare_digest(csrf_cookie, csrf_header):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF token is missing or invalid.")
    user = await request.app.state.container.auth.find_user(payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account is unavailable.")
    return user
