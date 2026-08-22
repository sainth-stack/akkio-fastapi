from sqlalchemy import Column, Integer, String, Boolean
from database import Base


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    completed = Column(Boolean, default=False, nullable=False)
