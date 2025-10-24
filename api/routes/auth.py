from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from api.db.database import get_db
from api.db.models import User, Session as DBSession, EmailVerificationToken
from api.auth.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    validate_password,
    ACCESS_TOKEN_EXPIRE_DAYS
)
from api.auth.middleware import get_current_user, require_auth, is_dev_user
from api.services.email import send_verification_email, is_email_configured
import os
import secrets
import pathlib

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Normalize BASE_URL from env or request
def _get_base_url(req: Request) -> str:
    env_url = (os.getenv("BASE_URL") or "").strip()
    if env_url:
        url = env_url.strip()
        # Remove trailing slash
        while url.endswith('/'):
            url = url[:-1]
        # If someone mistakenly set BASE_URL to include /api, strip it
        if url.lower().endswith('/api'):
            url = url[:-4]
        return url
    # Fallback to request base_url
    return str(req.base_url).rstrip('/')

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
def signup(request: SignupRequest, req: Request, db: Session = Depends(get_db)):
    """Create a new user account and send verification email."""
    # Check if email is banned
    email_lower = request.email.strip().lower()
    banned_file = pathlib.Path("artifacts") / "banned_emails.txt"
    if banned_file.exists():
        try:
            banned = set(line.strip().lower() for line in banned_file.read_text(encoding="utf-8").splitlines() if line.strip())
            if email_lower in banned:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This email address is not allowed to register"
                )
        except HTTPException:
            raise
        except Exception:
            pass
    
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

    # Proceed to create user
    hashed_password = get_password_hash(request.password)

    # Check if this is a dev email - auto-verify if so
    dev_emails_raw = os.getenv('DEV_EMAILS', 'duncan.w.yu@gmail.com')
    dev_emails = {e.strip().lower() for e in dev_emails_raw.split(',') if e.strip()}
    is_dev_email = request.email.strip().lower() in dev_emails

    # User starts unverified unless they're a dev
    user = User(
        email=request.email,
        password_hash=hashed_password,
        email_verified="true" if is_dev_email else "false"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    import logging
    logger = logging.getLogger("uvicorn")

    if is_dev_email:
        logger.info(f"🔧 Dev account created and auto-verified: {user.email}")
    else:
        # Generate verification token for non-dev accounts
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        verification_token = EmailVerificationToken(
            user_id=user.id,
            token=token,
            expires_at=expires_at
        )
        db.add(verification_token)
        db.commit()

        # Send verification email (if configured)
        base_url = _get_base_url(req)

        # Debug: check what env vars are set
        smtp_host = os.getenv("SMTP_HOST", "")
        smtp_user = os.getenv("SMTP_USERNAME", "")
        smtp_pass = os.getenv("SMTP_PASSWORD", "")
        logger.info(f"🔍 SMTP Config - Host: {'✓' if smtp_host else '✗'}, User: {'✓' if smtp_user else '✗'}, Pass: {'✓' if smtp_pass else '✗'}")

        if is_email_configured():
            logger.info(f"📧 Attempting to send verification email to {user.email}")
            success = send_verification_email(user.email, token, base_url)
            if success:
                logger.info(f"✅ Verification email sent successfully to {user.email}")
            else:
                logger.error(f"❌ Failed to send verification email to {user.email}")
        else:
            # Dev mode: log the verification link
            logger.info(
                f"[DEV] Email not configured. Verification link for {user.email}: {base_url}/verify-email?token={token}"
            )
    
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
    
    # Check if email is verified (only if email is configured)
    require_verification = is_email_configured() and os.getenv("REQUIRE_EMAIL_VERIFICATION", "true").lower() in {"1", "true", "yes", "on"}
    if require_verification and user.email_verified != "true" and not is_dev_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email address before logging in"
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    
    cookie_secure = os.getenv("COOKIE_SECURE", "false").strip().lower() in {"1","true","yes","on"}
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        samesite="lax",
        secure=cookie_secure
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

@router.post("/verify-email")
def verify_email(token: str | None = Form(None), req: Request = None, db: Session = Depends(get_db)):
    """Verify email address with token."""
    # Accept token via form body or query string for flexibility
    if not token and req is not None:
        token = req.query_params.get("token")
    if not token:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing verification token")
    verification = db.query(EmailVerificationToken).filter(
        EmailVerificationToken.token == token
    ).first()
    
    if not verification:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    if not verification.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired"
        )
    
    user = verification.user
    user.email_verified = "true"
    
    # Delete used token
    db.delete(verification)
    db.commit()
    
    return {"ok": True, "message": "Email verified successfully"}

class ResendVerificationRequest(BaseModel):
    email: EmailStr

@router.post("/resend-verification")
def resend_verification(request: ResendVerificationRequest, req: Request, db: Session = Depends(get_db)):
    """Resend verification email."""
    user = db.query(User).filter(User.email == request.email).first()
    
    if not user:
        # Don't reveal if email exists
        return {"ok": True, "message": "If the email exists, a verification link has been sent"}
    
    if user.email_verified == "true":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )
    
    # Delete old tokens
    db.query(EmailVerificationToken).filter(
        EmailVerificationToken.user_id == user.id
    ).delete()
    
    # Generate new token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=24)
    verification_token = EmailVerificationToken(
        user_id=user.id,
        token=token,
        expires_at=expires_at
    )
    db.add(verification_token)
    db.commit()
    
    # Send email
    base_url = os.getenv("BASE_URL") or str(req.base_url).rstrip("/")
    if is_email_configured():
        send_verification_email(user.email, token, base_url)
    else:
        import logging
        logging.getLogger("uvicorn").info(
            f"[DEV] Email verification link for {user.email}: {base_url}/verify-email?token={token}"
        )
    
    return {"ok": True, "message": "If the email exists, a verification link has been sent"}
