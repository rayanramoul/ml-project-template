ARG PYTHON_VERSION=3.11.5
FROM python:${PYTHON_VERSION}-slim as base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

