import hashlib
import logging
from tqdm import tqdm
import json
from typing import Any, Dict, Union
import pyarrow as pa
import os
import pyarrow.json
import meerkat as mk  
from tqdm import tqdm
import uuid

import pyarrow.feather as feather
import sqlalchemy  # type: ignore
from sqlalchemy.orm import sessionmaker
from google.cloud.sql.connector import Connector  # type: ignore
from sqlalchemy import Column, String  # type: ignore
from sqlalchemy.ext.declarative import declarative_base  # type: ignore

from schema import C4Meta, CCMeta, GithubMeta, ArxivMeta, WikipediaMeta, BookMeta, StackexchangeMeta, Document, Base

BATCH_SIZE = 1000
BASE_DIR = "/home/karan/data/pyjama/RedPajama-Data-1T-Sample"

source_to_model = {
    "github": GithubMeta,
    "arxiv": ArxivMeta,
    "book": BookMeta,
    "stackexchange": StackexchangeMeta,
    "wikipedia": WikipediaMeta,
    "c4": C4Meta,
    "cc": CCMeta,
}

# establish connection to the database
connector = Connector()
creds = json.load(open("/home/karan/data/pyjama/redpajama-db-creds.json"))

def getconn() -> Any:
    conn = connector.connect(
        creds["connection_name"],
        "pg8000",
        user=creds["user"],
        password=creds["password"],
        db=creds["db"],
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

def sanitize(text: any) -> any:
    if isinstance(text, str):
        return text.replace("\x00", "")
    return text

# iterate over all the jsonl files in the directory
for file in tqdm(os.listdir(BASE_DIR)):
    if not file.endswith(".jsonl"):
        continue
    path = os.path.join(BASE_DIR, file)
    
    # infer the source from the filename
    source = file.split("_")[0]
    if source not in source_to_model:
        continue
    model = source_to_model[source]
        
    # load the jsonl file into a dataframe
    df = mk.from_json(
        path,
        backend="arrow",
        lines=True,
        read_options=pa.json.ReadOptions(**{"block_size": 10<<20}),
    )

    # parse the metadata (since it is nested in the jsonl format)
    if source == "cc":
        df["cc_source"] = df["source"]
    else:
        import pyarrow.compute as pc
        struct_array = df["meta"].data
        result = {}
        for field_index in range(struct_array.type.num_fields):
            field = struct_array.type.field(field_index)
            result[field.name] = mk.ArrowScalarColumn(
                pc.struct_field(struct_array, field.name)
            )
        meta_df = mk.DataFrame(result)
        df = df.drop("meta")
        df = mk.concat([df, meta_df], axis=1)

    # iterate over the dataframe and insert the data into the database
    documents = []
    for posidx in tqdm(range(len(df))):
        row = df[posidx]

        # need to unpack language 
        if source == "github":
            language = row["language"]
            row["language"] = "none" if len(language) == 0 else language[0]["name"]
        document = Document(
            text=sanitize(row["text"]), 
            source=source
        )
        meta = model(
            document_id=document.id,
            **{
                k: sanitize(row[k])
                for k in model.__table__.columns.keys()
                if k != "id" and k != "document_id"
            }
        )
        # document.github_meta = meta
        
        setattr(document, f"{source}_meta", meta)
        documents.append(document)

        # if the counter is a multiple of the batch size, commit the session and start a new one
        if len(documents) % BATCH_SIZE == 0:
            try:
                session.add_all(documents)
                session.commit()
                documents = []
            except Exception as e:
                breakpoint()
                print(e)
                session.rollback()
            session = Session()
    session.add_all(documents)
    session.commit()
 

