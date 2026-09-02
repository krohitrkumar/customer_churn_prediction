from fastapi import APIRouter, HTTPException, status, Header, Depends
from typing import Optional
from database.config import settings
from database.connection import db_dependency
from schemas.auth import UserRegister, UserLogin, Token, UserOut, ChangePasswordRequest, SendOTPRequest, VerifyOTPRequest,ResetPasswordWithOTPRequest
from database.security import hash_password, create_access_token, verify_password, get_current_user
from models.user import User, UserRole, OTPVerification
from services.email_service import email_service
import random
from datetime import datetime, timedelta, timezone


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# 1. Register User
@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(
    payload: UserRegister,
    db: db_dependency
):
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{payload.email}' already exists."
        )
    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        first_name=payload.first_name,
        last_name=payload.last_name,
        role=payload.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

from fastapi.security import OAuth2PasswordRequestForm

# 2. Login User (Powers Swagger UI Authorize Dialog)
@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
def login_user(
    db: db_dependency,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive."
        )

    access_token = create_access_token(subject=user.id)
    return Token(access_token=access_token, token_type="bearer")

# 3. Get Current User Profile

@router.get("/me", response_model=UserOut, status_code=status.HTTP_200_OK)
def get_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Returns the profile of the current logged-in user."""
    return current_user

# 4. Change Password Endpoint
@router.post("/change_password", status_code=status.HTTP_200_OK)
def change_password(
    payload: ChangePasswordRequest,
    db: db_dependency,
    current_user: User = Depends(get_current_user)
):
    """Allows a logged-in user to securely change their password."""
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect."
        )
    if verify_password(payload.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password cannot be the same as your old password."
        )

    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()
    return {"message": "Password changed successfully."}

# 5. Send Email OTP
@router.post("/send_otp", status_code=status.HTTP_200_OK)
def send_otp(
    payload: SendOTPRequest,
    db: db_dependency
):
    otp_code = f"{random.randint(100000, 999999)}"
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)

    db.query(OTPVerification).filter(
        OTPVerification.email == payload.email,
        OTPVerification.is_verified == False
    ).delete()

    otp_record = OTPVerification(
        email=payload.email,
        otp_code=otp_code,
        expires_at=expires_at,
        is_verified=False
    )
    db.add(otp_record)
    db.commit()

    success = email_service.send_otp_email(payload.email, otp_code)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email. Please check SMTP settings."
        )

    return {"message": f"Verification code sent to {payload.email} (valid for 10 minutes)."}

# 6. Verify OTP
@router.post("/verify_otp", status_code=status.HTTP_200_OK)
def verify_otp(
    payload: VerifyOTPRequest,
    db: db_dependency
):
    now = datetime.now(timezone.utc)
    otp_record = db.query(OTPVerification).filter(
        OTPVerification.email == payload.email,
        OTPVerification.otp_code == payload.otp_code,
        OTPVerification.is_verified == False,
        OTPVerification.expires_at > now
    ).first()

    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code."
        )

    otp_record.is_verified = True
    db.commit()

    user = db.query(User).filter(User.email == payload.email).first()
    if user:
        access_token = create_access_token(subject=user.id)
        return {
            "message": "Email verified successfully.",
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user.id,
            "role": user.role
        }

    return {"message": "Email verified successfully."}


@router.post("/reset_password", status_code=status.HTTP_200_OK)
def reset_password_with_otp(
    payload: ResetPasswordWithOTPRequest,
    db: db_dependency
):
    now = datetime.now(timezone.utc)
    
    
    otp_record = db.query(OTPVerification).filter(
        OTPVerification.email == payload.email,
        OTPVerification.otp_code == payload.otp_code,
        OTPVerification.expires_at > now
    ).first()
    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code."
        )

    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    user.hashed_password = hash_password(payload.new_password)
    
    # 4. Invalidate / delete the used OTP record
    db.delete(otp_record)
    db.commit()
    return {"message": "Password has been successfully reset."}