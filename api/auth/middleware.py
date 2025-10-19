from typing import Optional
from fastapi import Request, HTTPException, status, Depends
import os
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from api.db.database import get_db
from api.db.models import User
from api.auth.security import decode_access_token

security = HTTPBearer(auto_error=False)

def get_token_from_cookie(request: Request) -> Optional[str]:
    """Extract JWT token from cookie."""
    return request.cookies.get("access_token")

def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current authenticated user from JWT token in cookie."""
    token = get_token_from_cookie(request)
    
    if not token:
        return None
    
    payload = decode_access_token(token)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    user = db.query(User).filter(User.id == user_id).first()
    return user

def require_auth(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """Dependency that requires authentication. Raises 401 if not authenticated."""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Optional authentication - returns None if not authenticated instead of raising error."""
    return get_current_user(request, db)

def is_dev_user(user: Optional[User]) -> bool:
    if not user or not getattr(user, 'email', None):
        return False
    # Allow configuring dev emails via env; default to requested address
    raw = os.environ.get('DEV_EMAILS', 'duncan.w.yu@gmail.com')
    allowed = {e.strip().lower() for e in raw.split(',') if e.strip()}
    try:
        return user.email.lower() in allowed
    except Exception:
        return False

def require_dev(
    request: Request,
    db: Session = Depends(get_db)
) -> User:
    """Require a developer/admin user. 403 if not privileged."""
    user = require_auth(request, db)
    if not is_dev_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges",
        )
    return user
