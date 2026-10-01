"""Seed realistic starter users. Codegen extends this with domain data."""
from database import Base, SessionLocal, engine
from models import User
from auth import hash_password

Base.metadata.create_all(bind=engine)


def run():
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            db.add_all([
                User(email="admin@example.com", hashed_password=hash_password("admin123"), role="ADMIN"),
                User(email="quality@example.com", hashed_password=hash_password("quality123"), role="QUALITY_MANAGER"),
                User(email="inspector@example.com", hashed_password=hash_password("inspector123"), role="INSPECTOR"),
                User(email="viewer@example.com", hashed_password=hash_password("viewer123"), role="VIEWER"),
            ])
            db.commit()
            print("Seeded default users")
        else:
            print("Users already present")
    finally:
        db.close()


if __name__ == "__main__":
    run()
