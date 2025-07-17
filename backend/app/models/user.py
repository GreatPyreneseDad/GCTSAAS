"""Dataclasses representing users in the system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class User:
    """Simple user record used by the API and database layer."""

    id: str
    org_id: Optional[str]
    email: str
    full_name: str
    role: Optional[str] = None
    baseline_measurements: Optional[Dict[str, Any]] = None
    privacy_settings: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_record(self) -> Dict[str, Any]:
        """Serialize the user for database insertion."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "baseline_measurements": self.baseline_measurements,
            "privacy_settings": self.privacy_settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "User":
        """Create a ``User`` instance from a database record or dict."""
        return cls(
            id=record["id"],
            org_id=record.get("org_id"),
            email=record["email"],
            full_name=record.get("full_name", ""),
            role=record.get("role"),
            baseline_measurements=record.get("baseline_measurements"),
            privacy_settings=record.get("privacy_settings"),
            created_at=record.get("created_at", datetime.utcnow()),
            updated_at=record.get("updated_at", datetime.utcnow()),
        )
