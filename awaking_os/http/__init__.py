"""HTTP API — FastAPI surface over the A-Kernel.

Importing this module requires the ``[http]`` extra
(``pip install -e ".[http]"``) which pulls in fastapi, uvicorn, and
sse-starlette.
"""

from awaking_os.http.api import (
    HealthResponse,
    SubmitRequest,
    SubmitResponse,
    create_app,
)

__all__ = ["HealthResponse", "SubmitRequest", "SubmitResponse", "create_app"]
