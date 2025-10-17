#!/bin/bash

export FASTAPI_ENV=development
export FASTAPI_PORT=8000
export FASTAPI_HOST=localhost

uv run uvicorn app:app --host $FASTAPI_HOST --port $FASTAPI_PORT --reload