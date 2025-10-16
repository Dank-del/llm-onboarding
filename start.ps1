$env:FASTAPI_ENV = "development"
$env:FASTAPI_PORT = "8000"
$env:FASTAPI_HOST = "localhost"

uvicorn app:app --host $env:FASTAPI_HOST --port $env:FASTAPI_PORT --reload
