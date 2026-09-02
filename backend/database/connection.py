from typing import Annotated
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from database.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True, 
    pool_recycle=300, 
    pool_size=10, 
    max_overflow=20
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False, 
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally: 
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

    