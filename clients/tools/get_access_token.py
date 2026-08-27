from typing import Annotated

from fastapi import Depends,HTTPException,status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer_scheme=HTTPBearer(auto_error=False)

def get_access_token(
        credentials:Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(bearer_scheme)
        ],
)->str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials
