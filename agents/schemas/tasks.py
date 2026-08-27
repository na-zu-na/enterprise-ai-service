import datetime
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_serializer


class tasks(BaseModel):
    title:str = Field(
        min_length=1,
        max_length=255
    )

    description:str = Field(
        min_length=1,
    )

    due_at: datetime= Field(
        alias='dueAt',
    )

    @field_serializer("due_at",when_used="json")
    def serialize_due_at(self,value: datetime)->str:
        if value.tzinfo is None:
            value=value.astimezone(
                ZoneInfo("Asia/Shanghai")
            )
        return value.replace(tzinfo=None).isoformat(timespec="seconds")