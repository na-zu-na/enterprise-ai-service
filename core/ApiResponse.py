from typing import TypeVar, Generic

from pydantic import BaseModel

T= TypeVar("T")

class ApiResponse(BaseModel,Generic[T]):
    code: int=200
    message:str="success"
    data:T | None=None

    @classmethod
    def success(cls,data: T | None=None):
        return cls(
            code=200,
            message="success",
            data=data
        )

    @classmethod
    def error(cls,code:int = 500,message:str="internal server error"):
        return cls(
            code=code,
            message=message,
            data=None
        )
