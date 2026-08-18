from fastapi import FastAPI

from api.exception_handlers import document_parse_exception_handler
from api.routes.document import router as document_router
from parsers.exceptions import DocumentParseException

app = FastAPI(
    title="Enterprise Agent AI Service",
    version="1.0.0"
)

#注册全局异常处理
app.add_exception_handler(
    DocumentParseException,
    document_parse_exception_handler,
)
app.include_router(document_router,tags=["document"])
