"""Dependency-free password hashing and HS256 bearer-token helpers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PASSWORD_SCRYPT_N = 2**14
PASSWORD_SCRYPT_R = 8
PASSWORD_SCRYPT_P = 1
PASSWORD_DERIVED_BYTES = 32
ACCESS_TOKEN_SECONDS = 30 * 60
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class AuthenticationConfigurationError(RuntimeError):
    """Raised when authentication secrets are missing or unsafe."""


class InvalidAccessToken(ValueError):
    """Raised for malformed, invalid or expired access tokens."""


def _jwt_secret() -> bytes:
    secret = os.environ.get("AUTH_JWT_SECRET", "")
    encoded = secret.encode("utf-8")
    if len(encoded) < 32:
        raise AuthenticationConfigurationError(
            "AUTH_JWT_SECRET must contain at least 32 bytes"
        )
    return encoded


def hash_password(password: str) -> str:
    """Hash a password with scrypt and a fresh random salt."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=PASSWORD_SCRYPT_N,
        r=PASSWORD_SCRYPT_R,
        p=PASSWORD_SCRYPT_P,
        dklen=PASSWORD_DERIVED_BYTES,
        maxmem=64 * 1024 * 1024,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    derived_text = base64.urlsafe_b64encode(derived).decode("ascii").rstrip("=")
    return f"scrypt${salt_text}${derived_text}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a scrypt hash without exposing password comparison timing."""
    if not isinstance(password, str) or not isinstance(password_hash, str):
        return False
    try:
        algorithm, salt_text, derived_text = password_hash.split("$")
        if algorithm != "scrypt":
            return False
        salt = _decode_base64url(salt_text)
        expected = _decode_base64url(derived_text)
        if len(salt) != 16 or len(expected) != PASSWORD_DERIVED_BYTES:
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=PASSWORD_SCRYPT_N,
            r=PASSWORD_SCRYPT_R,
            p=PASSWORD_SCRYPT_P,
            dklen=PASSWORD_DERIVED_BYTES,
            maxmem=64 * 1024 * 1024,
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, binascii.Error):
        return False


def create_access_token(subject: str, *, now: int | None = None) -> tuple[str, int]:
    """Return a short-lived HS256 JWT and its lifetime in seconds."""
    if not subject:
        raise ValueError("token subject is required")
    issued_at = int(time.time() if now is None else now)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "iat": issued_at,
        "exp": issued_at + ACCESS_TOKEN_SECONDS,
    }
    signing_input = ".".join(
        (_encode_json(header), _encode_json(payload))
    ).encode("ascii")
    signature = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()
    token = signing_input.decode("ascii") + "." + _encode_base64url(signature)
    return token, ACCESS_TOKEN_SECONDS


def decode_access_token(token: str, *, now: int | None = None) -> str:
    """Validate an HS256 JWT and return its user subject."""
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        header = json.loads(_decode_base64url(encoded_header))
        payload: dict[str, Any] = json.loads(_decode_base64url(encoded_payload))
        if not isinstance(header, dict) or header.get("alg") != "HS256":
            raise InvalidAccessToken("unsupported access-token algorithm")
        if header.get("typ") != "JWT" or not isinstance(payload, dict):
            raise InvalidAccessToken("invalid access-token format")
        signature = _decode_base64url(encoded_signature)
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        expected = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise InvalidAccessToken("invalid access-token signature")
        subject = payload.get("sub")
        expires_at = payload.get("exp")
        current_time = int(time.time() if now is None else now)
        if not isinstance(subject, str) or not subject:
            raise InvalidAccessToken("access-token subject is missing")
        if isinstance(expires_at, bool) or not isinstance(expires_at, int):
            raise InvalidAccessToken("access-token expiry is invalid")
        if expires_at <= current_time:
            raise InvalidAccessToken("access token has expired")
        return subject
    except AuthenticationConfigurationError:
        raise
    except InvalidAccessToken:
        raise
    except (ValueError, TypeError, UnicodeDecodeError, binascii.Error) as error:
        raise InvalidAccessToken("malformed access token") from error


def _encode_json(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _encode_base64url(encoded)


def _encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_base64url(value: str) -> bytes:
    if not isinstance(value, str) or not _B64URL_RE.fullmatch(value):
        raise ValueError("invalid base64url data")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
