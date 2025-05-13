import json
import logging
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Security settings
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-key")
RBAC_CONFIG_PATH = (
    Path(__file__).parent.parent.parent / "deployment" / "security" / "rbac.json"
)


# Load RBAC configuration
def load_rbac_config() -> dict[str, Any]:
    try:
        if RBAC_CONFIG_PATH.exists():
            with open(RBAC_CONFIG_PATH) as f:
                return json.load(f)
        else:
            logger.warning(f"RBAC configuration file not found at {RBAC_CONFIG_PATH}")
            return {"roles": {}, "users": {}}
    except Exception as e:
        logger.exception(f"Error loading RBAC configuration: {e}")
        return {"roles": {}, "users": {}}


rbac_config = load_rbac_config()

# Security bearer token
security = HTTPBearer()


def get_user_permissions(user_id: str) -> set[str]:
    """Get permissions for a user based on their roles."""
    permissions: set[str] = set()

    user_config = rbac_config.get("users", {}).get(user_id, {})
    user_roles = user_config.get("roles", [])

    for role in user_roles:
        role_permissions = (
            rbac_config.get("roles", {}).get(role, {}).get("permissions", [])
        )
        permissions.update(role_permissions)

    return permissions


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    """Verify JWT token and return payload."""
    try:
        token = credentials.credentials
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as e:
        logger.exception(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def check_permission(required_permission: str) -> Callable[[dict[str, Any]], None]:
    """Check if user has required permission."""

    def _check(payload: dict[str, Any]) -> None:
        user_id = payload.get("sub", "")
        permissions = get_user_permissions(user_id)

        # Check for wildcard permission or specific permission
        if "*" in permissions or required_permission in permissions:
            return

        logger.warning(
            f"User {user_id} attempted to access {required_permission} without permission",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    return _check


def requires_permission(permission: str) -> Callable[[F], F]:
    """Decorator to check if user has required permission."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = next((arg for arg in args if isinstance(arg, Request)), None)
            if not request:
                for arg_value in kwargs.values():
                    if isinstance(arg_value, Request):
                        request = arg_value
                        break

            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found in function arguments",
                )

            # Skip auth in development mode if configured
            if os.getenv("SKIP_AUTH", "").lower() in ("true", "1", "yes"):
                return await func(*args, **kwargs)

            auth_header = request.headers.get("Authorization")
            if not auth_header:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authorization header is missing",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                check_permission(permission)(payload)
            except (IndexError, jwt.PyJWTError) as e:
                logger.exception(f"Authentication error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return await func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
