"""Extract main content + title from raw HTML.

Primary: trafilatura. Fallback: a thin BeautifulSoup pass that grabs <title>
and concatenates <p> tags.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Extracted:
    title: str | None
    text: str | None


def extract_content(html: str, url: str | None = None) -> Extracted:
    # Try trafilatura first.
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True, url=url)
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = getattr(meta, "title", None) if meta else None
        if text:
            return Extracted(title=title, text=text)
    except Exception:
        pass

    # Fallback: BeautifulSoup.
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join(p for p in paragraphs if p)
        return Extracted(title=title, text=text or None)
    except Exception:
        return Extracted(title=None, text=None)
