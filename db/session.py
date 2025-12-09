from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

try:
    from core.config import settings
except ImportError:
    from core.config import settings


# Configure engine based on database type
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
else:
    engine = create_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        connect_args={"options": "-c client_min_messages=error"},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def set_session_vars(db, *, user_id: Optional[str], tenant_id: Optional[str], organization_id: Optional[str], roles_csv: str, is_super_admin: bool, enc_key: Optional[str] = None):
    db.execute(text("SET LOCAL app.user_id = :val"), {"val": user_id or ""})
    db.execute(text("SET LOCAL app.tenant_id = :val"), {"val": tenant_id or ""})
    db.execute(text("SET LOCAL app.organization_id = :val"), {"val": organization_id or ""})
    db.execute(text("SET LOCAL app.roles = :val"), {"val": roles_csv})
    db.execute(text("SET LOCAL app.is_super_admin = :val"), {"val": "true" if is_super_admin else "false"})
    if enc_key:
        db.execute(text("SET LOCAL app.enc_key = :val"), {"val": enc_key})
