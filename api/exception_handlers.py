from fastapi import Request
from fastapi.responses import JSONResponse

from clients.spring_client import SpringUnauthorizedError
from core.ApiResponse import ApiResponse
from parsers.exceptions import (
    DocumentNotFoundException,
    DocumentParseException,
    DocumentReadException,
    UnsupportedFileTypeException,
)


async def document_parse_exception_handler(
        _request: Request,
        exc: Exception,
) -> JSONResponse:
    if isinstance(exc, UnsupportedFileTypeException):
        status_code = 400
    elif isinstance(exc, DocumentNotFoundException):
        status_code = 404
    elif isinstance(exc, DocumentReadException):
        status_code = 422
    else:
        status_code = 500

    response = ApiResponse.error(
        code=status_code,
        message=str(exc),
    )
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )


async def spring_unauthorized_exception_handler(
        _request: Request,
        _exc: SpringUnauthorizedError,
) -> JSONResponse:
    status_code = 401
    response = ApiResponse.error(
        code=status_code,
        message="登录状态已失效，请重新登录",
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
        headers={"WWW-Authenticate": "Bearer"},
    )
