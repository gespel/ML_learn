#!/usr/bin/env python3
"""Lädt zufällige Wikipedia-Artikel und speichert deren reinen Text als .txt-Dateien.

Beispiel:
    python llm/wiki_random_scraper.py --count 5
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Iterable

import requests

WIKIPEDIA_API = "https://de.wikipedia.org/w/api.php"
INVALID_FILENAME_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1F]")
MAX_RANDOM_BATCH = 1000
MAX_RETRIES = 6
MAX_BACKOFF_SECONDS = 20
DEFAULT_REQUEST_TIMEOUT = 20
APP_NAME = "MLLearnWikiScraper"
APP_VERSION = "1.0"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logger = logging.getLogger(__name__)


def sanitize_filename(title: str) -> str:
    """Erzeugt einen sicheren Dateinamen aus einem Artikeltitel."""
    cleaned = INVALID_FILENAME_CHARS.sub("_", title).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "unbenannter_artikel"


def build_user_agent(contact: str) -> str:
    """Erstellt einen Wikimedia-konformen User-Agent mit Kontaktinfo."""
    contact_clean = contact.strip()
    if not contact_clean:
        raise ValueError("Kontaktinfo für User-Agent fehlt")
    return (
        f"{APP_NAME}/{APP_VERSION} "
        f"({contact_clean}) requests/{requests.__version__}"
    )


def _retry_delay_seconds(response: requests.Response | None, attempt: int) -> float:
    """Berechnet die Wartezeit bei Rate-Limit/temporären Fehlern."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
    return min(2 ** attempt, MAX_BACKOFF_SECONDS)


def api_get_json(
    session: requests.Session,
    params: dict[str, object],
    *,
    request_delay: float,
    request_timeout: float,
) -> dict:
    """Führt einen API-Call mit Retry + Backoff aus."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if request_delay > 0:
            time.sleep(request_delay)

        try:
            response = session.get(
                WIKIPEDIA_API,
                params=params,
                timeout=(10, request_timeout),
            )

            if response.status_code == 429 or response.status_code >= 500:
                wait_for = _retry_delay_seconds(response, attempt)
                logger.warning(
                    "Wikipedia API antwortete mit HTTP %s (Versuch %s/%s). Warte %.1fs vor Retry.",
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES + 1,
                    wait_for,
                )
                time.sleep(wait_for)
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            wait_for = _retry_delay_seconds(None, attempt)
            logger.warning(
                "Netzwerk/API-Fehler (%s) bei Versuch %s/%s. Warte %.1fs vor Retry.",
                exc,
                attempt + 1,
                MAX_RETRIES + 1,
                wait_for,
            )
            time.sleep(wait_for)

    if last_error:
        raise last_error
    raise RuntimeError("Unbekannter API-Fehler")


def get_random_titles(
    session: requests.Session,
    count: int,
    *,
    request_delay: float,
    request_timeout: float,
) -> list[str]:
    """Holt `count` zufällige Artikeltitel von der deutschen Wikipedia."""
    titles: list[str] = []
    seen: set[str] = set()

    while len(titles) < count:
        batch_size = min(MAX_RANDOM_BATCH, count - len(titles))
        logger.info("Hole %s zufällige Titel (bisher %s/%s).", batch_size, len(titles), count)
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,  # nur Artikel (keine Spezialseiten)
            "grnlimit": batch_size,
        }
        data = api_get_json(
            session,
            params,
            request_delay=request_delay,
            request_timeout=request_timeout,
        )

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            if title and title not in seen:
                titles.append(title)
                seen.add(title)

        if not pages:
            logger.warning("Wikipedia lieferte keine weiteren Zufallsseiten.")
            break

    logger.info("%s zufällige Titel geladen.", len(titles))
    return titles


def fetch_plain_text(
    session: requests.Session,
    title: str,
    *,
    request_delay: float,
    request_timeout: float,
) -> tuple[str, str]:
    """Lädt Klartext für genau einen Artikel."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,  # reiner Text
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": title,
    }
    data = api_get_json(
        session,
        params,
        request_delay=request_delay,
        request_timeout=request_timeout,
    )

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        resolved_title = page.get("title", "")
        extract = page.get("extract", "").strip()
        if resolved_title and extract:
            return resolved_title, extract
    return "", ""


def save_article(storage_dir: Path, title: str, text: str) -> Path:
    """Speichert den Text als UTF-8 .txt-Datei."""
    safe_name = sanitize_filename(title)
    file_path = storage_dir / f"{safe_name}.txt"
    if file_path.exists():
        suffix = 2
        while True:
            candidate = storage_dir / f"{safe_name}_{suffix}.txt"
            if not candidate.exists():
                file_path = candidate
                break
            suffix += 1
    file_path.write_text(text, encoding="utf-8")
    logger.info("Artikel gespeichert: %s", file_path)
    return file_path


def scrape_random_articles(
    count: int,
    storage_dir: Path,
    *,
    request_delay: float,
    request_timeout: float,
    user_agent: str,
) -> Iterable[Path]:
    """Lädt zufällige Artikel und speichert sie als Textdateien."""
    storage_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        session.headers.update({"User-Agent": user_agent})
        titles = get_random_titles(
            session,
            count,
            request_delay=request_delay,
            request_timeout=request_timeout,
        )

        for title in titles:
            resolved_title, text = fetch_plain_text(
                session,
                title,
                request_delay=request_delay,
                request_timeout=request_timeout,
            )
            if not text:
                logger.debug("Kein Inhalt für '%s'.", title)
                continue
            yield save_article(storage_dir, resolved_title, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scraped zufällige Wikipedia-Artikel in .storage als .txt"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Anzahl zufälliger Artikel (Standard: 1)",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "../.storage",
        help="Zielordner (Standard: <workspace>/.storage)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Pause (Sekunden) zwischen API-Requests (Standard: 0.2)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Read-Timeout pro Request in Sekunden (Standard: "
            f"{DEFAULT_REQUEST_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log-Level (Standard: INFO)",
    )
    parser.add_argument(
        "--ua-contact",
        required=True,
        help=(
            "Kontaktinfo für Wikimedia-konformen User-Agent, z. B. "
            "'mailto:du@beispiel.de' oder 'https://deine-seite.tld/kontakt'"
        ),
    )
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.count < 1:
        raise SystemExit("--count muss mindestens 1 sein")
    if args.request_delay < 0:
        raise SystemExit("--request-delay darf nicht negativ sein")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout muss > 0 sein")

    try:
        user_agent = build_user_agent(args.ua_contact)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    logger.info(
        (
            "Starte Scraping: count=%s, storage_dir=%s, request_delay=%.2f, "
            "request_timeout=%.1f"
        ),
        args.count,
        args.storage_dir,
        args.request_delay,
        args.request_timeout,
    )
    logger.info("Verwende User-Agent: %s", user_agent)

    saved_count = 0
    for index, _ in enumerate(
        scrape_random_articles(
            args.count,
            args.storage_dir,
            request_delay=args.request_delay,
            request_timeout=args.request_timeout,
            user_agent=user_agent,
        ),
        start=1,
    ):
        
        saved_count += 1
        if index % 100 == 0:
            logger.info("Verarbeitet: %s/%s Titel.", index, args.count)

    if saved_count == 0:
        logger.warning("Keine Artikel gespeichert.")
        return

    logger.info("Fertig. Insgesamt %s Artikel gespeichert.", saved_count)


if __name__ == "__main__":
    main()
