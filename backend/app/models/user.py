"""User data model placeholder"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

@dataclass
class User:
    id: str
    org_id: Optional[str]
    email: str
    full_name: str
    role: Optional[str] = None
    baseline_measurements: Optional[Dict] = None
    privacy_settings: Optional[Dict] = None
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
