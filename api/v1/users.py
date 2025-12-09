from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.deps import permission_required, get_db
from core.security import get_password_hash
from db.models.users import User

from pydantic import BaseModel

router = APIRouter(prefix="/users", tags=["users"]) 


class CreateUserRequest(BaseModel):
    email: str
    full_name: Optional[str] = None
    password: str
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None


@router.post("/", dependencies=[Depends(permission_required("users.manage"))])
def create_user(payload: CreateUserRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")
    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=get_password_hash(payload.password),
        tenant_id=payload.tenant_id,
        organization_id=payload.organization_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": str(user.id), "email": user.email}
