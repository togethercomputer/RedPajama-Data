import hashlib
import logging
from tqdm import tqdm
from typing import Any, Dict, Union

import pyarrow.feather as feather
import sqlalchemy  # type: ignore
from sqlalchemy.orm import sessionmaker
from google.cloud.sql.connector import Connector  # type: ignore
from sqlalchemy import Column, String  # type: ignore
from sqlalchemy.ext.declarative import declarative_base  # type: ignore

from schema import Code, Document, Base


connector = Connector()
cache_name = "postgres"
cache_connection = "hai-gcp-fine-grained:us-central1:redpajama"
cache_user = "postgres"
cache_password = ""
cache_db = "redpajama"


def getconn() -> Any:
    conn = connector.connect(
        cache_connection,
        "pg8000",
        user=cache_user,
        password=cache_password,
        db=cache_db,
    )
    return conn


engine = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=getconn,
)
engine.dialect.description_encoding = None  # type: ignore

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

df = feather.read_feather(
    "/Users/sabrieyuboglu/data/pyjama/consolidated/consolidated/filtered_3be6a5cc7428401393c23e5516a43537.feather"
)

batch_size = 100
counter = 0
for _, row in tqdm(df[:1000].iterrows(), total=len(df[:1000])):
    document = Document(
        text=row["text_sample"],
    )
    session.add(document)
    counter += 1

    # If the counter is a multiple of the batch size, commit the session and start a new one
    if counter % batch_size == 0:
        print("comitting")
        session.commit()
        print("done")
        session = Session()
        print("recreating session")

session.commit()
