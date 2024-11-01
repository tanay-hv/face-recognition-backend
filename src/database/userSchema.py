from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Date, String
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column

class Base:
    allow_unmapped = True

Base = declarative_base(cls = Base)

class User(Base):
    __tablename__ = 'user'
    id = mapped_column(String, primary_key=True, unique=True)
    name = mapped_column(String)
    birthdate = mapped_column(Date)
    faceVectors = mapped_column(Vector(512))