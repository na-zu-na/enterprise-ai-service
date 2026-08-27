from functools import lru_cache

import requests

from core.config import Settings


class SpringClientError(RuntimeError):
    pass


class SpringUnauthorizedError(SpringClientError):
    pass


class SpringClient:
    def __init__(self):
        self.base_url = Settings.SPRING_REQUEST_URL.rstrip("/")
        self._session = requests.Session()

    def request(
            self,
            method: str,
            path: str,
            *,
            access_token: str,
            **kwargs,
    ) -> requests.Response:

        url = f"{self.base_url}/{path.lstrip('/')}"

        request_headers = dict(
            kwargs.pop("headers", {}) or {}
        )

        request_headers["Authorization"] = (
            f"Bearer {access_token}"
        )

        timeout = kwargs.pop("timeout", 20)

        response = self._session.request(
            method=method,
            url=url,
            headers=request_headers,
            timeout=timeout,
            **kwargs,
        )

        if response.status_code == 401:
            raise SpringUnauthorizedError(
                "Spring 拒绝了访问令牌"
            )

        response.raise_for_status()
        return response

    def get(
            self,
            path: str,
            **kwargs,
    ) -> requests.Response:
        return self.request("GET", path, **kwargs)

    def post(
            self,
            path: str,
            **kwargs,
    ) -> requests.Response:
        return self.request("POST", path, **kwargs)


@lru_cache(maxsize=1)
def get_spring_client() -> SpringClient:
    return SpringClient()