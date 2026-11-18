"""일관된 오류 응답 형식: {"error": {"code": ..., "message": ...}}

code 값은 frontend/src/types.ts 의 ErrorCode 와 일치해야 합니다.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def validation_error(message: str) -> AppError:
    return AppError("validation", message, 422)


def not_found(message: str) -> AppError:
    return AppError("not_found", message, 404)


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})
