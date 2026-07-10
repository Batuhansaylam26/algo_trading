FROM python:3.12.12-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gzip \
        libgomp1 \
        postgresql-client \
        rsync \
    && rm -rf /var/lib/apt/lists/*
