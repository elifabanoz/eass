FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY alembic.ini ./
COPY eass ./eass
COPY migrations ./migrations
COPY rules ./rules
RUN pip install --no-cache-dir '.[ml]' && groupadd -g 10001 eass && useradd -r -g eass -u 10001 eass \
    && mkdir /quarantine /hashes && chown eass:eass /quarantine /hashes
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER eass:eass

