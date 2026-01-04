# backend/security.py - Version corrigée
import hashlib
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError  # <-- CHANGEMENT ICI
from passlib.context import CryptContext

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
API_KEY_SALT = os.getenv("API_KEY_SALT", secrets.token_hex(16))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_api_key(api_key: str) -> str:
    """Hash une clé API avec salt"""
    salted = api_key + API_KEY_SALT
    return hashlib.sha256(salted.encode()).hexdigest()

def verify_api_key(hashed_key: str, input_key: str) -> bool:
    """Vérifie si une clé API correspond au hash"""
    return hash_api_key(input_key) == hashed_key

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crée un JWT token (pour usage futur)"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Vérifie un JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:  # <-- CHANGEMENT ICI
        return None

def get_password_hash(password: str) -> str:
    """Hash un mot de passe"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe"""
    return pwd_context.verify(plain_password, hashed_password)

# FastAPI dependency for expert authentication
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def require_expert(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Vérifie qu'un appelant est un expert uniquement."""
    token = credentials.credentials

    if token != os.getenv("EXPERT_API_KEY", "expert-burkina-2024"):
        raise HTTPException(status_code=403, detail="Invalid API token")

    return {"token": token, "role": "expert"}


async def require_admin_or_expert(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Autorise soit la clé expert, soit la clé admin.

    Utilisé pour les routes d'ingestion IA (/ai/ingest, /ai/ingest/photo)
    afin que le panel admin puisse aussi ingérer des connaissances.
    """
    token = credentials.credentials

    expert_key = os.getenv("EXPERT_API_KEY", "expert-burkina-2024")
    admin_key = os.getenv("ADMIN_API_KEY", "admin-souverain-burkina-2024")

    if token == expert_key:
        return {"token": token, "role": "expert"}

    if token == admin_key:
        return {"token": token, "role": "admin"}

    raise HTTPException(status_code=403, detail="Invalid API token")
