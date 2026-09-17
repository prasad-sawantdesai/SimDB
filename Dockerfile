ARG PYVER=3.12
FROM ghcr.io/astral-sh/uv:python${PYVER}-trixie-slim

ENV UV_NO_DEV=1
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libldap2-dev \
    libsasl2-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml alembic.ini ./
COPY src/ ./src/
RUN uv sync --locked --extra all

ENV SIMDB_SITE_CONFIG_PATH=/app/config/simdb.cfg

CMD ["uv", "run", "simdb_server"]

