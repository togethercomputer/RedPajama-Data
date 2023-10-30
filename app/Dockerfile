FROM python:3.11.5-slim-bookworm

WORKDIR /usr/app

RUN apt-get update && apt-get install -y \
    build-essential \
    isal \
    git

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install cmake cython

# copy requirements.txt to the working directory
COPY requirements.txt requirements.txt

# install python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# install mwparserfromhell (v >= 0.7.0 for spanish)
RUN pip3 install apache_beam
RUN pip3 install git+https://github.com/earwig/mwparserfromhell.git@0f89f44

# set python hash seed
ENV PYTHONHASHSEED 42

# copy the local files to the working directory
COPY . .



