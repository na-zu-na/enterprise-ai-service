from contextlib import asynccontextmanager

from fastapi import FastAPI

from agents.graph import build_graph
from api.exception_handlers import (
    document_parse_exception_handler,
    spring_unauthorized_exception_handler,
)
from api.routes.document import router as document_router
from clients.google_calendar_mcp import create_google_calendar_mcp_client
from clients.spring_client import SpringUnauthorizedError
from parsers.exceptions import DocumentParseException
from api.routes.embedding import router as embedding_documents_router
from api.routes.rag import router as rag_router
from api.routes.retrieval import router as retrieval_router
from api.routes.agent import router as agent_router
from db.checkpointer import checkpoint_saver


@asynccontextmanager
async def lifespan(app: FastAPI):
    client=create_google_calendar_mcp_client()
    calendar_tools= await client.get_tools()

    async with checkpoint_saver() as saver:
        app.state.google_calendar_mcp = client

        app.state.agent_graph = build_graph(
            checkpointer=saver,
            external_tools=calendar_tools,
        )
        yield

app = FastAPI(
    title="Enterprise Agent AI Service",
    version="1.0.0",
    lifespan=lifespan,
)

#注册全局异常处理
app.add_exception_handler(
    DocumentParseException,
    document_parse_exception_handler,
)
app.add_exception_handler(
    SpringUnauthorizedError,
    spring_unauthorized_exception_handler,
)
app.include_router(document_router,tags=["document"])
app.include_router(embedding_documents_router,tags=["embedding"])
app.include_router(rag_router,tags=["rag"])
app.include_router(retrieval_router)
app.include_router(agent_router,tags=["agent"])
