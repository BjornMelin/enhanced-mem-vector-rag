#!/usr/bin/env python3
"""Generate JWT tokens for user authentication with the EMVR MCP server."""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import jwt

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def load_env() -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    env_path = Path(__file__).parent.parent / ".env"

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key] = value

    return env_vars


def generate_token(user_id: str, expiry_days: int = 30) -> str:
    """Generate a JWT token for the specified user."""
    env_vars = load_env()
    jwt_secret = env_vars.get("JWT_SECRET") or os.environ.get(
        "JWT_SECRET",
        "your-jwt-secret-key",
    )

    # Check if user exists in RBAC config
    rbac_path = Path(__file__).parent.parent / "security" / "rbac.json"
    if rbac_path.exists():
        with open(rbac_path) as f:
            rbac_config = json.load(f)
            if user_id not in rbac_config.get("users", {}):
                print(f"Warning: User '{user_id}' not found in RBAC configuration")

    # Set expiration time
    expiry = datetime.datetime.utcnow() + datetime.timedelta(days=expiry_days)

    # Create token payload
    payload = {
        "sub": user_id,
        "iat": datetime.datetime.utcnow(),
        "exp": expiry,
    }

    # Generate token
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def main() -> None:
    """Parse arguments and generate token."""
    parser = argparse.ArgumentParser(
        description="Generate JWT token for EMVR authentication",
    )
    parser.add_argument("user_id", help="User ID to generate token for")
    parser.add_argument(
        "--expiry",
        type=int,
        default=30,
        help="Token expiry in days (default: 30)",
    )
    parser.add_argument("--output", help="Output file path (optional)")

    args = parser.parse_args()

    token = generate_token(args.user_id, args.expiry)

    if args.output:
        with open(args.output, "w") as f:
            f.write(token)
        print(f"Token written to {args.output}")
    else:
        print("\nGenerated JWT Token:")
        print("=" * 40)
        print(token)
        print("=" * 40)
        print(f"\nThis token will expire in {args.expiry} days")
        print("\nUse this token in the Authorization header:")
        print(f"Authorization: Bearer {token}")


if __name__ == "__main__":
    main()
