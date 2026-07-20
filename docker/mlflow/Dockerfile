FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip \
    && python -m pip install mlflow psycopg2-binary boto3
