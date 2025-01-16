# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.6
FROM python:${PYTHON_VERSION}-slim AS lightning-base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# THis is necessary to keep .venv in the container after build
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Prevent uv from downloading isolated Python builds as Python is already available
ENV UV_PYTHON_DOWNLOADS=false

# While doing runs in docker, we want to avoid having automatic sync if packages are missing
ENV UV_NO_SYNC=false

WORKDIR /app


# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
  --disabled-password \
  --gecos "" \
  --home "/nonexistent" \
  --shell "/sbin/nologin" \
  --no-create-home \
  --uid "${UID}" \
  appuser

# Copy the source code into the container.
COPY . .

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked

# Switch to the non-privileged user to run the application.
USER appuser
