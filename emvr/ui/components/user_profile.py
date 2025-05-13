"""
User profile component for EMVR UI.

This module provides functionality for managing user profiles and preferences.
"""

import logging
from typing import Any

import chainlit as cl
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class UserProfile(BaseModel):
    """
    User profile model.

    This class represents a user profile with preferences and settings.
    """

    user_id: str
    display_name: str | None = None
    preferences: dict[str, Any] = {}
    api_keys: dict[str, str] = {}

    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.

        Args:
        ----
            key: The preference key
            default: Default value if not found

        Returns:
        -------
            The preference value

        """
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference.

        Args:
        ----
            key: The preference key
            value: The preference value

        """
        self.preferences[key] = value

    def get_api_key(self, service: str) -> str | None:
        """
        Get an API key for a service.

        Args:
        ----
            service: The service name

        Returns:
        -------
            The API key if available

        """
        return self.api_keys.get(service)

    def set_api_key(self, service: str, key: str) -> None:
        """
        Set an API key for a service.

        Args:
        ----
            service: The service name
            key: The API key

        """
        self.api_keys[service] = key


def get_current_user_profile() -> UserProfile | None:
    """
    Get the current user's profile.

    Returns
    -------
        The user profile if available

    """
    profile_data = cl.user_session.get("user_profile")
    if profile_data is None:
        return None

    return UserProfile(**profile_data)


async def create_user_profile(
    user_id: str,
    display_name: str | None = None,
) -> UserProfile:
    """
    Create a new user profile.

    Args:
    ----
        user_id: The user ID
        display_name: Optional display name

    Returns:
    -------
        The created user profile

    """
    profile = UserProfile(
        user_id=user_id,
        display_name=display_name or user_id,
    )

    # Store the profile in the session
    cl.user_session.set("user_profile", profile.dict())

    logger.info(f"Created user profile for {user_id}")

    return profile


async def update_user_profile(profile: UserProfile) -> None:
    """
    Update a user profile in the session.

    Args:
    ----
        profile: The updated profile

    """
    # Store the updated profile in the session
    cl.user_session.set("user_profile", profile.dict())

    logger.info(f"Updated user profile for {profile.user_id}")


async def show_profile_settings() -> None:
    """Show the profile settings UI."""
    profile = get_current_user_profile()
    if profile is None:
        await cl.Message(
            content="No user profile found. Please refresh the page.",
            author="System",
        ).send()
        return

    # Create settings elements
    elements = [
        cl.Text(
            name="display_name",
            label="Display Name",
            initial=profile.display_name or "",
        ),
        cl.Select(
            name="theme",
            label="Theme",
            values=["light", "dark"],
            initial=profile.get_preference("theme", "light"),
        ),
        cl.Toggle(
            name="notifications",
            label="Enable Notifications",
            initial=profile.get_preference("notifications", True),
        ),
        cl.Password(
            name="openai_api_key",
            label="OpenAI API Key (Optional)",
            initial=profile.get_api_key("openai") or "",
        ),
    ]

    await cl.Message(
        content="Your profile settings:",
        elements=elements,
        author="EMVR",
    ).send()
