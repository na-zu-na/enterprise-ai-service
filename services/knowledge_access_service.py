from typing import Annotated

import requests
from fastapi import Depends, HTTPException
from pydantic import Field, TypeAdapter, ValidationError

from clients.spring_client import SpringUnauthorizedError, get_spring_client
from clients.tools.get_access_token import get_access_token
from core.config import Settings
from schemas.retrieval import DocumentDetail


def spring_data(path: str, access_token: str):
    try:
        response = get_spring_client().get(path, access_token=access_token)
        payload = response.json()
    except SpringUnauthorizedError as exc:
        raise HTTPException(401, "登录状态已失效", headers={"WWW-Authenticate": "Bearer"}) from exc
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 502
        raise HTTPException(code if code in {401, 403, 404} else 502, "知识库服务请求失败") from exc
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(502, "知识库服务不可用或响应无效") from exc
    if not isinstance(payload, dict):
        raise HTTPException(502, "知识库服务响应无效")
    if payload.get("code") != 200:
        code = payload.get("code")
        raise HTTPException(code if code in (401, 403, 404) else 502, "知识库服务拒绝请求")
    return payload.get("data")


def get_accessible_knowledge_base_ids(
    access_token: str = Depends(get_access_token),
) -> list[int]:
    data = spring_data(Settings.SPRING_KNOWLEDGE_BASE_IDS_PATH, access_token)
    try:
        ids = TypeAdapter(list[Annotated[int, Field(strict=True, gt=0)]]).validate_python(data)
    except ValidationError as exc:
        raise HTTPException(502, "知识库权限响应无效") from exc
    return list(dict.fromkeys(ids))


def get_document_detail(document_id: int, access_token: str) -> DocumentDetail:
    data = spring_data(f"/api/documents/{document_id}", access_token)
    try:
        detail = DocumentDetail.model_validate(data)
        if detail.id != document_id:
            raise ValueError("document ID mismatch")
        return detail
    except (ValidationError, ValueError) as exc:
        raise HTTPException(502, "文档详情响应无效") from exc
