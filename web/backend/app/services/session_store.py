"""In-memory session management."""

import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from threading import Lock

from ..config import settings


@dataclass
class Session:
    """User session data."""
    id: str
    requests_used: int = 0
    requests_limit: int = field(default_factory=lambda: settings.RATE_LIMIT_REQUESTS)
    created_at: datetime = field(default_factory=datetime.utcnow)
    model: str = "qwen2.5:7b"  # Default to Qwen 2.5 7B (best tool calling)
    # All 5 built-in tools enabled by default
    tools: List[str] = field(default_factory=lambda: [
        "Calculator", "DateTime", "UnitConverter", "TextProcessor", "RandomGenerator"
    ])
    messages: List[dict] = field(default_factory=list)
    
    @property
    def requests_remaining(self) -> int:
        return max(0, self.requests_limit - self.requests_used)
    
    @property
    def is_expired(self) -> bool:
        expiry = self.created_at + timedelta(hours=settings.SESSION_EXPIRY_HOURS)
        return datetime.utcnow() > expiry
    
    @property
    def is_rate_limited(self) -> bool:
        return self.requests_used >= self.requests_limit


class SessionStore:
    """Thread-safe in-memory session store."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()
    
    def create(self) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(id=session_id)
        
        with self._lock:
            self._sessions[session_id] = session
            self._cleanup_expired()
        
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session and session.is_expired:
                del self._sessions[session_id]
                return None
            
            return session
    
    def get_or_create(self, session_id: Optional[str]) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get(session_id)
            if session:
                return session
        
        return self.create()
    
    def increment_requests(self, session_id: str) -> bool:
        """Increment request count. Returns True if successful, False if rate limited."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            if session.is_rate_limited:
                return False
            
            session.requests_used += 1
            return True
    
    def update_config(self, session_id: str, model: Optional[str] = None, tools: Optional[List[str]] = None):
        """Update session configuration."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if model:
                    session.model = model
                if tools is not None:
                    session.tools = tools
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session history."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    def _cleanup_expired(self):
        """Remove expired sessions (called within lock)."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired
        ]
        for sid in expired:
            del self._sessions[sid]
    
    def stats(self) -> dict:
        """Get store statistics."""
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": sum(
                    1 for s in self._sessions.values()
                    if not s.is_expired and s.requests_used > 0
                )
            }


# Global session store instance
session_store = SessionStore()
