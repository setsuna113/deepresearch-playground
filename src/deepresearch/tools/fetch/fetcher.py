"""PageFetcher — async HTTP fetch w/ size cap and basic content-type filter."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger(__name__)


@dataclass
class FetchResult:
    url: str
    status: int
    html: str | None
    error: str | None = None


@dataclass
class PageFetcher:
    user_agent: str = "deepresearch-playground/0.1 (research)"
    timeout_s: float = 20.0
    max_bytes: int = 2_500_000

    async def fetch(self, url: str) -> FetchResult:
        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s, follow_redirects=True, headers=headers
            ) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        return FetchResult(url=url, status=resp.status_code, html=None,
                                           error=f"HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "")
                    if "html" not in ctype and "text" not in ctype:
                        return FetchResult(url=url, status=resp.status_code, html=None,
                                           error=f"unsupported content-type: {ctype}")
                    buf = b""
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) > self.max_bytes:
                            log.info("fetch_truncated", url=url, bytes=len(buf))
                            break
                    return FetchResult(
                        url=url, status=resp.status_code, html=buf.decode("utf-8", errors="replace")
                    )
        except Exception as e:
            return FetchResult(url=url, status=0, html=None, error=repr(e))
