FROM amd64/python:3.6-alpine3.12

COPY ./requirements.txt /requirements.txt
COPY ./index.py /index.py
RUN apk update
# RUN apk add make automake gcc g++ subversion python3-dev
RUN pip install -r requirements.txt