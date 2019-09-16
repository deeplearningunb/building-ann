FROM python:3.6

RUN apt-get update

COPY . /building-ann

WORKDIR /building-ann

RUN pip install --no-cache-dir -r requirements.txt
