import uuid

from sqlalchemy import Column, Integer, String, Text, BigInteger, Float, UUID, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column, registry
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Document(Base): 
    __tablename__ = 'documents_sample'
    id = mapped_column(Integer, primary_key=True, autoincrement=True) #Column(UUID(as_uuid=True), primary_key=True)
    text = Column(Text)
    source = Column(Text)
    source_file: Mapped["SourceFile"] = relationship(back_populates="document")
    github_meta: Mapped["GithubMeta"] = relationship(back_populates="document")
    arxiv_meta: Mapped["ArxivMeta"] = relationship(back_populates="document")
    book_meta: Mapped["BookMeta"] = relationship(back_populates="document")
    stackexchange_meta: Mapped["StackexchangeMeta"] = relationship(back_populates="document")
    wikipedia_meta: Mapped["WikipediaMeta"] = relationship(back_populates="document")
    c4_meta: Mapped["C4Meta"] = relationship(back_populates="document")
    cc_meta: Mapped["CCMeta"] = relationship(back_populates="document")
    pca: Mapped["PCA"] = relationship(back_populates="document")

class SourceFile(Base):
    __tablename__ = 'source_files_sample'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="source_file")
    filename = Column(String)
    position = Column(Integer)


class GithubMeta(Base):
    __tablename__ = 'github_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="github_meta")
    ref = Column(Text)
    repo_name = Column(Text)
    source = Column(Text)
    size = Column(Text)
    content_hash = Column(Text)
    path = Column(Text)
    license = Column(Text)
    language = Column(Text)
    line_count = Column(Integer)
    max_line_length = Column(Integer)
    avg_line_length = Column(Float)


class ArxivMeta(Base):
    __tablename__ = 'arxiv_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="arxiv_meta")
    timestamp = Column(Text)
    arxiv_id = Column(Text)
    language = Column(Text)
    url = Column(Text)


class BookMeta(Base):
    __tablename__ = 'book_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="book_meta")
    title = Column(Text)
    short_book_title = Column(Text)
    publication_date = Column(Integer)

class StackexchangeMeta(Base):
    __tablename__ = 'stackexchange_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="stackexchange_meta")
    question_score = Column(Text)
    url = Column(Text)
    timestamp = Column(Text)
    language = Column(Text)

class WikipediaMeta(Base):
    __tablename__ = 'wikipedia_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="wikipedia_meta")
    title = Column(Text)
    url = Column(Text)
    timestamp = Column(Text)
    language = Column(Text)

class C4Meta(Base):
    __tablename__ = 'c4_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="c4_meta")
    url = Column(Text)
    timestamp = Column(Text)
    language = Column(Text)

class CCMeta(Base):
    __tablename__ = 'cc_meta_sample'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="cc_meta")
    cc_source = Column(Text)

   
class PCA(Base):

    __tablename__ = "pca_sample"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = mapped_column(ForeignKey("documents_sample.id"))
    document: Mapped["Document"] = relationship(back_populates="pca")
    pca_1 = Column(Float)
    pca_2 = Column(Float)
    pca_3 = Column(Float)
    pca_4 = Column(Float)
    pca_5 = Column(Float)
    pca_6 = Column(Float)
    pca_7 = Column(Float)
    pca_8 = Column(Float)
    pca_9 = Column(Float)
    pca_10 = Column(Float)


