"""Small, dependency-light backend for the public course demonstration.

This is intentionally separate from the production service in ``main.py``.
It demonstrates the user-visible RAG, knowledge-boundary, and approval flows
without exposing private Spring, database, calendar, or model credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


KNOWLEDGE = {
    "employee-policy": [
        {
            "document": "Annual Leave Policy",
            "section": "Eligibility and Carry Over",
            "text": "Leave eligibility depends on region, employment type, service period, and current leave balance. Employees should confirm their applicable policy with Human Resources.",
        },
        {
            "document": "Hybrid Work Policy",
            "section": "Team Arrangements",
            "text": "Employees must follow approved team arrangements, availability rules, and office-attendance requirements.",
        },
        {
            "document": "Data Classification Standard",
            "section": "Restricted Data",
            "text": "Restricted data may only be accessed by authorised roles and must not be stored in unapproved locations.",
        },
    ],
    "tech-docs": [
        {
            "document": "SDK Integration Guide",
            "section": "Authorised Credentials",
            "text": "The SDK allows approved partners to integrate AI capabilities using authorised credentials.",
        },
        {
            "document": "AI Model Release Procedure",
            "section": "Production Release Controls",
            "text": "Production releases require evaluation, risk review, approval records, deployment information, and a rollback plan.",
        },
    ],
}


class QueryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1_000)
    knowledge_base_ids: list[Literal["employee-policy", "tech-docs"]] = Field(min_length=1)


class ActionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    deadline: str | None = Field(default=None, max_length=100)


PENDING_ACTIONS: dict[str, ActionRequest] = {}

app = FastAPI(title="PE6203 Enterprise Assistant Demo API", version="1.0.0-demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def keywords(text: str) -> set[str]:
    return {word.strip(".,!?;:()[]{}\"'").lower() for word in text.split() if len(word) > 2}


def select_evidence(message: str, selected_ids: list[str]) -> list[dict[str, str]]:
    query_words = keywords(message)
    scored: list[tuple[int, dict[str, str]]] = []
    for kb_id in selected_ids:
        for item in KNOWLEDGE[kb_id]:
            score = len(query_words & keywords(item["document"] + " " + item["text"]))
            scored.append((score, item))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for score, item in scored if score > 0][:2]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "public-course-demo"}


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, object]:
    evidence = select_evidence(request.message, request.knowledge_base_ids)
    if not evidence:
        return {
            "status": "insufficient_information",
            "answer": "The selected knowledge base does not contain sufficient information to answer this question.",
            "citations": [],
            "selected_knowledge_base_ids": request.knowledge_base_ids,
        }

    cited_names = " and ".join(f"【{item['document']}】" for item in evidence)
    answer = f"Based on the selected enterprise material, {evidence[0]['text']} {cited_names}"
    return {
        "status": "completed",
        "answer": answer,
        "citations": evidence,
        "selected_knowledge_base_ids": request.knowledge_base_ids,
    }


@app.post("/api/actions/prepare")
def prepare_action(request: ActionRequest) -> dict[str, object]:
    action_id = str(uuid4())
    PENDING_ACTIONS[action_id] = request
    return {
        "status": "approval_required",
        "action_id": action_id,
        "action": "create_task",
        "payload": request.model_dump(),
        "message": "Review the task details before approving the write action.",
    }


@app.post("/api/actions/{action_id}/approve")
def approve_action(action_id: str) -> dict[str, object]:
    action = PENDING_ACTIONS.pop(action_id, None)
    if action is None:
        raise HTTPException(status_code=404, detail="The pending action was not found or has expired.")
    return {
        "status": "completed",
        "action": "create_task",
        "task": action.model_dump(),
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
        "message": "The demo task was created after explicit approval.",
    }
