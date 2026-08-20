from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'postgresql://ai_service:199550@localhost:5432/enterprise_agent'

engine=create_engine(DATABASE_URI,pool_pre_ping=True)

SessionLocal=sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()