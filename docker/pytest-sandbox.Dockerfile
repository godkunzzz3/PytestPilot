FROM python:3.11-slim

RUN python -m pip install --no-cache-dir pytest \
    && groupadd --gid 65532 firstcoder \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin firstcoder

WORKDIR /workspace
USER 65532:65532
