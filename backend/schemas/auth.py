from datetime import datetime
import re
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
from models.user import UserRole

class UserRegister(BaseModel): 
    email: EmailStr = Field(..., examples=["jane.doe@example.com"])
    password: str = Field(
        ..., 
        min_length=8, 
        max_length=72, 
        description="Password must be between 8 and 72 characters", 
        examples=["SecurePass123!"]
    )
    first_name: str = Field(..., min_length=1, max_length=100, examples=["Jane"])
    last_name: str = Field(..., min_length=1, max_length=100, examples=["Doe"])
    role: Optional[UserRole] = UserRole.USER

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must contain at least one uppercase letter (A-Z).")
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must contain at least one lowercase letter (a-z).")
        if not re.search(r"\d", value):  # 🟢 Fixed: \d matches digits 0-9
            raise ValueError("Password must contain at least one digit (0-9).")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            raise ValueError("Password must contain at least one special character (!@#$%^&*...).")
        return value

class UserLogin(BaseModel): 
    email: EmailStr = Field(..., examples=["jane.doe@example.com"])
    password: str = Field(..., examples=["SecurePass123!"])

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    email: EmailStr
    first_name: str
    last_name: str
    role: UserRole
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., examples=["OldPassword123!"])
    new_password: str = Field(
        ..., 
        min_length=8, 
        max_length=72, 
        description="New password must be between 8 and 72 characters", 
        examples=["NewSecurePass123!"]
    )

    @field_validator("new_password")
    @classmethod
    def validate_new_password_strength(cls, value: str) -> str:
        if not re.search(r"[A-Z]", value):
            raise ValueError("New password must contain at least one uppercase letter (A-Z).")
        if not re.search(r"[a-z]", value):
            raise ValueError("New password must contain at least one lowercase letter (a-z).")
        if not re.search(r"\d", value):
            raise ValueError("New password must contain at least one digit (0-9).")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            raise ValueError("New password must contain at least one special character (!@#$%^&*...).")
        return value

class SendOTPRequest(BaseModel):
    email: EmailStr = Field(..., examples=["jane.doe@example.com"])

class VerifyOTPRequest(BaseModel):
    email: EmailStr = Field(..., examples=["jane.doe@example.com"])
    otp_code: str = Field(..., min_length=6, max_length=6, examples=["123456"])

class ResetPasswordWithOTPRequest(BaseModel):
    email: EmailStr = Field(...,examples=["example@gmail.com"])
    otp_code :str = Field(...,min_length=6,max_length=6,examples=["123456"])
    new_password:str = Field(...,min_length=8,max_length=72,
                             description="new password must be 8 to 72 charchters.",
                             examples=["Newsecure@12343"])

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, value: str):
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must contain at least one uppercase letter (A-Z).")
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must contain at least one lowercase letter (a-z).")
        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one digit (0-9).")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            raise ValueError("Password must contain at least one special character (!@#$%^&*...).")
        return value