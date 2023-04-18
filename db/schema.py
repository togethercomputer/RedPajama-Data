from sqlalchemy import Column, Integer, String, Text, BigInteger, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Code(Base):
    __tablename__ = 'code'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer)
    language = Column(Text)
    ref = Column(Text)
    repo_name = Column(Text)
    source = Column(Text)
    size = Column(Text)
    content_hash = Column(Text)
    path = Column(Text)
    license = Column(Text)
    line_count = Column(Integer)
    max_line_length = Column(Integer)
    avg_line_length = Column(Float)


class Document(Base): 
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    num_tokens = Column(Integer)
    meta = Column(Text)


