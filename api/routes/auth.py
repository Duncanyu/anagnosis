from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from api.db.database import get_db
from api.db.models import User, Session as DBSession
from api.auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    validate_password,
    ACCESS_TOKEN_EXPIRE_DAYS
)
from api.auth.middleware import get_current_user, require_auth, is_dev_user

router = APIRouter(prefix="/api/auth", tags=["auth"])

class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    
    class Config:
        from_attributes = True

@router.post("/signup", response_model=UserResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    """Create a new user account."""
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    is_valid, error_msg = validate_password(request.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    hashed_password = get_password_hash(request.password)
    user = User(email=request.email, password_hash=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return UserResponse(id=str(user.id), email=user.email)

@router.post("/login")
def login(request: LoginRequest, response: Response, db: Session = Depends(get_db)):
    """Login and receive JWT token in cookie."""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )
    
    return {
        "ok": True,
        "user": {"id": str(user.id), "email": user.email, "is_dev": bool(is_dev_user(user))}
    }

@router.post("/logout")
def logout(response: Response):
    """Logout by clearing the JWT cookie."""
    response.delete_cookie(key="access_token")
    return {"ok": True, "message": "Logged out successfully"}

@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(require_auth)):
    """Get current authenticated user."""
    return UserResponse(id=str(user.id), email=user.email)

@router.get("/check")
def check_auth(request: Request, db: Session = Depends(get_db)):
    """Check if user is authenticated."""
    user = get_current_user(request, db)
    if user:
        return {
            "authenticated": True,
            "user": {"id": str(user.id), "email": user.email, "is_dev": bool(is_dev_user(user))}
        }
    return {"authenticated": False}
