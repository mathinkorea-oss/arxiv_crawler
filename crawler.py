"""
arXiv 논문 검색·수집 (공식 API: https://export.arxiv.org/api/query)

이용 지침 반영:
- arXiv API Terms of Use: 연속 요청·단일 연결, 요청 간격 준수(본 도구는 15초 간격 적용)
- abs 상세페이지는 메타데이터 확인용 HTML만 요청 (PDF 미다운로드)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

API_URL = "https://export.arxiv.org/api/query"
ABS_PAGE_TITLE_SELECTOR = "h1.title.mathjax"
ABS_PAGE_TITLE_FALLBACK_SELECTOR = "h1.title"
REQUEST_INTERVAL_SEC = 15.0
MIN_RETRY_WAIT_SEC = 15.0
MAX_RETRIES = 3
NON_RETRYABLE_HTTP = frozenset({400, 403, 404, 410})
MAX_RESULTS_PER_REQUEST = 2000
PAPERS_CSV_NAME = "papers.csv"
API_SORT_BY = "submittedDate"
API_SORT_ORDER = "descending"
USER_AGENT = "arxiv_crawler/1.0 (+local research tool; arXiv API ToU compliant)"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

RESULT_COLUMNS = [
    "제목",
    "저자",
    "초록",
    "최초_등록일",
    "arxiv_id",
    "arxiv_버전",
    "논문_페이지_URL",
    "PDF_URL",
    "상세페이지_제목",
    "제목_일치여부",
    "상세페이지_수집상태",
]


class ArxivApiError(Exception):
    """arXiv API 검색 등 치명적 실패."""


class HtmlParseError(Exception):
    """상세페이지 HTML 파싱 실패."""


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass
class HttpFetchOutcome:
    success: bool
    text: str = ""
    attempts: int = 0
    error_kind: str = ""
    error_message: str = ""
    retried: bool = False


class RunLogger:
    """실행별 파일 로그 (초록·HTML 본문 미기록)."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._file = log_path.open("a", encoding="utf-8")
        self._write("INFO", "-", "-", "run_start", 0, "-", f"로그 시작: {log_path.name}")

    def close(self) -> None:
        if not self._file.closed:
            self._write("INFO", "-", "-", "run_end", 0, "-", "실행 종료")
            self._file.close()

    def _write(
        self,
        level: str,
        arxiv_id: str,
        url: str,
        outcome: str,
        attempts: int,
        error_kind: str,
        reason: str,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = (
            f"{ts}\t{level}\t{arxiv_id}\t{url}\t{outcome}\t"
            f"attempts={attempts}\t{error_kind}\t{reason}\n"
        )
        self._file.write(line)
        self._file.flush()

    def log_success(self, arxiv_id: str, url: str, attempts: int, note: str = "OK") -> None:
        self._write("INFO", arxiv_id, url, "success", attempts, "-", note)

    def log_duplicate_skip(self, arxiv_id: str, url: str, reason: str) -> None:
        self._write("INFO", arxiv_id, url or "-", "duplicate_skip", 0, "-", reason)

    def log_retry(
        self, arxiv_id: str, url: str, attempt: int, error_kind: str, reason: str
    ) -> None:
        self._write("WARN", arxiv_id, url, "retry", attempt, error_kind, reason)

    def log_final_failure(
        self, arxiv_id: str, url: str, attempts: int, error_kind: str, reason: str
    ) -> None:
        self._write("ERROR", arxiv_id, url, "final_failure", attempts, error_kind, reason)


def create_run_logger(base_dir: Path) -> RunLogger:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return RunLogger(base_dir / "logs" / f"run_{stamp}.log")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}시간 {minutes}분 {sec}초"
    if minutes:
        return f"{minutes}분 {sec}초"
    return f"{sec}초"


def retry_backoff_seconds(failed_attempt: int) -> float:
    """failed_attempt: 방금 실패한 시도 번호(1..MAX_RETRIES-1)."""
    return max(MIN_RETRY_WAIT_SEC, MIN_RETRY_WAIT_SEC * failed_attempt)


def parse_retry_after(response: requests.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    header = header.strip()
    if header.isdigit():
        return float(header)
    try:
        dt = parsedate_to_datetime(header)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def wait_request_interval(clock: Clock, last_request_start: float | None) -> None:
    if last_request_start is None:
        return
    elapsed = clock.monotonic() - last_request_start
    if elapsed < REQUEST_INTERVAL_SEC:
        clock.sleep(REQUEST_INTERVAL_SEC - elapsed)


def build_search_query(user_query: str) -> str:
    """일반 검색어는 all 필드 구문 검색으로 변환. 이미 API 문법이면 그대로 사용."""
    q = user_query.strip()
    if not q:
        raise ValueError("검색어가 비어 있습니다.")
    upper = q.upper()
    if any(m in q for m in (":", "+AND+", "+OR+", "+ANDNOT+")) or " AND " in upper or " OR " in upper:
        return q
    escaped = q.replace('"', "")
    return f'all:"{escaped}"'


def base_arxiv_id(value: str) -> str:
    """중복 판단용 기본 ID (버전 접미사 v1, v2 제거)."""
    text = str(value).strip()
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)


def extract_arxiv_version(abs_url: str) -> str:
    match = re.search(r"arxiv\.org/abs/[^/]+?(v\d+)(?:$|[/?#])", abs_url, re.I)
    return match.group(1).lower() if match else ""


def version_sort_key(version: str) -> int:
    if not version:
        return 0
    m = re.match(r"v(\d+)$", version, re.I)
    return int(m.group(1)) if m else 0


def _parse_arxiv_id(entry_id: str) -> str:
    """http://arxiv.org/abs/2301.12345v2 → 2301.12345"""
    match = re.search(r"arxiv\.org/abs/([^/?#]+)", entry_id, re.I)
    if not match:
        return base_arxiv_id(entry_id)
    return base_arxiv_id(match.group(1))


def _text(elem: ET.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return " ".join(elem.text.split())


def _entry_links(entry: ET.Element) -> tuple[str, str]:
    abs_url = ""
    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        href = link.get("href") or ""
        rel = link.get("rel") or ""
        link_type = link.get("type") or ""
        title = (link.get("title") or "").lower()
        if rel == "alternate" and not abs_url:
            abs_url = href
        if link_type == "application/pdf" or title == "pdf":
            pdf_url = href
    if not pdf_url and abs_url:
        pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"
    return abs_url, pdf_url


def parse_feed(xml_text: str) -> tuple[int, list[dict[str, str]]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivApiError("arXiv API 응답을 XML로 해석할 수 없습니다.") from exc

    total_el = root.find("opensearch:totalResults", ATOM_NS)
    try:
        total = int(_text(total_el)) if total_el is not None else 0
    except ValueError:
        total = 0

    rows: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        entry_id = _text(entry.find("atom:id", ATOM_NS))
        title = _text(entry.find("atom:title", ATOM_NS))
        summary = _text(entry.find("atom:summary", ATOM_NS))
        published = _text(entry.find("atom:published", ATOM_NS))
        if published:
            published = published.replace("T", " ").replace("Z", " UTC")

        authors = [
            _text(author.find("atom:name", ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [a for a in authors if a]
        abs_url, pdf_url = _entry_links(entry)

        rows.append(
            {
                "제목": title,
                "저자": "; ".join(authors),
                "초록": summary,
                "최초_등록일": published,
                "arxiv_id": _parse_arxiv_id(entry_id),
                "arxiv_버전": extract_arxiv_version(abs_url),
                "논문_페이지_URL": abs_url,
                "PDF_URL": pdf_url,
            }
        )

    return total, rows


def normalize_title_for_compare(title: str) -> str:
    """비교용: 줄바꿈·탭·연속 공백을 단일 공백으로 정리."""
    return " ".join(title.split())


def titles_match(api_title: str, detail_title: str) -> bool:
    return normalize_title_for_compare(api_title) == normalize_title_for_compare(detail_title)


def extract_title_from_abs_html(html: str) -> tuple[str, str]:
    """
    arXiv abs 상세페이지 HTML에서 제목 추출.
    선택자: h1.title.mathjax (없으면 h1.title), span.descriptor(Title:) 제거.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        raise HtmlParseError(str(exc)) from exc

    h1 = soup.select_one(ABS_PAGE_TITLE_SELECTOR) or soup.select_one(
        ABS_PAGE_TITLE_FALLBACK_SELECTOR
    )
    if h1 is None:
        return "", "실패(HTML): 제목 영역(h1.title)을 찾을 수 없음"

    for descriptor in h1.select("span.descriptor"):
        descriptor.decompose()

    title = h1.get_text(separator=" ", strip=False)
    title = re.sub(r"^\s*Title:\s*", "", title, flags=re.IGNORECASE)
    title = title.strip()
    if not title:
        return "", "실패(HTML): 제목 텍스트가 비어 있음"
    return title, "성공"


def enrich_rows_with_detail_titles(
    client: "ArxivClient",
    rows: list[dict[str, str]],
    logger: RunLogger | None = None,
    progress: "DetailProgress | None" = None,
) -> int:
    """신규 논문 상세 제목 수집. 실패 건수 반환."""
    failures = 0
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        step_start = client.clock.monotonic()
        arxiv_id = row.get("arxiv_id", "")
        abs_url = (row.get("논문_페이지_URL") or "").strip()
        if progress:
            progress.report(index, total)

        if not abs_url:
            row["상세페이지_제목"] = ""
            row["제목_일치여부"] = "비교불가"
            row["상세페이지_수집상태"] = "실패: 논문_페이지_URL 없음"
            failures += 1
            if logger:
                logger.log_final_failure(
                    arxiv_id, "-", 0, "validation", "논문_페이지_URL 없음"
                )
            if progress:
                progress.record_step(client.clock.monotonic() - step_start)
            continue

        outcome = client.fetch_url(abs_url, arxiv_id=arxiv_id, logger=logger)
        if not outcome.success:
            row["상세페이지_제목"] = ""
            row["제목_일치여부"] = "비교불가"
            row["상세페이지_수집상태"] = (
                f"실패({outcome.error_kind}): {outcome.error_message}"
            )
            failures += 1
            if logger:
                logger.log_final_failure(
                    arxiv_id,
                    abs_url,
                    outcome.attempts,
                    outcome.error_kind,
                    outcome.error_message,
                )
            if progress:
                progress.record_step(client.clock.monotonic() - step_start)
            continue

        try:
            detail_title, status = extract_title_from_abs_html(outcome.text)
        except HtmlParseError as exc:
            row["상세페이지_제목"] = ""
            row["제목_일치여부"] = "비교불가"
            row["상세페이지_수집상태"] = f"실패(HTML 파싱): {exc}"
            failures += 1
            if logger:
                logger.log_final_failure(
                    arxiv_id,
                    abs_url,
                    outcome.attempts,
                    "html_parse",
                    str(exc),
                )
            if progress:
                progress.record_step(client.clock.monotonic() - step_start)
            continue

        if status == "성공":
            row["상세페이지_제목"] = detail_title
            row["상세페이지_수집상태"] = status
            row["제목_일치여부"] = (
                "일치" if titles_match(row["제목"], detail_title) else "불일치"
            )
            if logger:
                logger.log_success(arxiv_id, abs_url, outcome.attempts, "상세 제목 수집")
        else:
            row["상세페이지_제목"] = ""
            row["상세페이지_수집상태"] = status
            row["제목_일치여부"] = "비교불가"
            failures += 1
            if logger:
                logger.log_final_failure(
                    arxiv_id, abs_url, outcome.attempts, "html_extract", status
                )
        if progress:
            progress.record_step(client.clock.monotonic() - step_start)
    return failures


@dataclass
class DetailProgress:
    clock: Clock
    run_start: float
    step_durations: list[float] = field(default_factory=list)

    def report(self, current: int, total: int) -> None:
        elapsed = self.clock.monotonic() - self.run_start
        if self.step_durations:
            avg = sum(self.step_durations) / len(self.step_durations)
        else:
            avg = REQUEST_INTERVAL_SEC
        remaining = max(0, total - current)
        eta = remaining * avg
        print(
            f"  신규 논문 {current}/{total} 처리 중 | "
            f"경과 {format_duration(elapsed)} | "
            f"예상 남은 {format_duration(eta)}"
        )

    def record_step(self, duration: float) -> None:
        self.step_durations.append(duration)


def is_test_result_csv(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "_mock" in parts or "mock_overwrite" in parts:
        return True
    if name.startswith("arxiv_mock_"):
        return True
    return False


def is_legacy_result_csv(path: Path, papers_path: Path) -> bool:
    if path.resolve() == papers_path.resolve():
        return False
    if path.name.lower() == PAPERS_CSV_NAME.lower():
        return False
    if is_test_result_csv(path):
        return False
    return path.suffix.lower() == ".csv" and path.name.lower().startswith("arxiv_")


def normalize_loaded_row(row: dict) -> dict[str, str] | None:
    arxiv_raw = row.get("arxiv_id", "")
    if arxiv_raw is None or (isinstance(arxiv_raw, float) and pd.isna(arxiv_raw)):
        arxiv_id = ""
    else:
        arxiv_id = str(arxiv_raw).strip()
    if not arxiv_id or arxiv_id.lower() in ("nan", "none"):
        return None
    normalized: dict[str, str] = {}
    for key, val in row.items():
        if val is None or (isinstance(val, float) and pd.isna(val)):
            normalized[str(key)] = ""
        else:
            normalized[str(key)] = str(val)
    normalized["arxiv_id"] = base_arxiv_id(arxiv_id)
    if not normalized.get("arxiv_버전"):
        normalized["arxiv_버전"] = extract_arxiv_version(normalized.get("논문_페이지_URL", ""))
    for col in RESULT_COLUMNS:
        normalized.setdefault(col, "")
    return normalized


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"arxiv_id": str})
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="utf-8", dtype={"arxiv_id": str})
        except Exception as exc:
            return [], f"인코딩/파싱 실패: {exc}"
    except Exception as exc:
        return [], f"읽기 실패: {exc}"

    if df.empty:
        return [], "데이터 행 없음"

    rows: list[dict[str, str]] = []
    for record in df.to_dict(orient="records"):
        normalized = normalize_loaded_row(record)
        if normalized:
            rows.append(normalized)
    if not rows:
        return [], "유효한 arxiv_id 없음"
    return rows, None


def merge_rows_by_base_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        base = base_arxiv_id(row["arxiv_id"])
        current = best.get(base)
        if current is None:
            best[base] = row
            continue
        if version_sort_key(row.get("arxiv_버전", "")) >= version_sort_key(
            current.get("arxiv_버전", "")
        ):
            best[base] = row
    return list(best.values())


def bootstrap_papers_from_results(
    results_dir: Path, papers_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    messages: list[str] = []
    legacy_files = sorted(
        p for p in results_dir.rglob("*.csv") if is_legacy_result_csv(p, papers_path)
    )
    skipped_test = sorted(
        p for p in results_dir.rglob("*.csv") if is_test_result_csv(p)
    )
    for csv_path in skipped_test:
        messages.append(f"제외(테스트용): {csv_path.relative_to(results_dir)}")
    combined: list[dict[str, str]] = []
    for csv_path in legacy_files:
        rows, err = read_csv_rows(csv_path)
        if err:
            messages.append(f"판별 보류(미병합): {csv_path.name} — {err}")
            continue
        combined.extend(rows)
        messages.append(f"병합 소스: {csv_path.name} ({len(rows)}행)")

    if not combined:
        return pd.DataFrame(columns=RESULT_COLUMNS), messages

    merged = merge_rows_by_base_id(combined)
    messages.append(f"기존 결과 CSV에서 {len(merged)}편을 papers.csv 초기 데이터로 준비했습니다.")
    return pd.DataFrame(merged, columns=RESULT_COLUMNS), messages


def load_papers_dataframe(papers_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(papers_path, encoding="utf-8-sig", dtype={"arxiv_id": str})
    except UnicodeDecodeError:
        df = pd.read_csv(papers_path, encoding="utf-8", dtype={"arxiv_id": str})
    if df.empty:
        return df
    df["arxiv_id"] = df["arxiv_id"].astype(str).map(base_arxiv_id)
    if "arxiv_버전" not in df.columns:
        df["arxiv_버전"] = ""
    df["arxiv_버전"] = df.apply(
        lambda r: str(r.get("arxiv_버전", "") or "")
        or extract_arxiv_version(str(r.get("논문_페이지_URL", ""))),
        axis=1,
    )
    return df


def load_or_initialize_papers(
    results_dir: Path,
) -> tuple[pd.DataFrame, list[str], bool]:
    """returns (df, messages, bootstrapped_new_file)"""
    papers_path = results_dir / PAPERS_CSV_NAME
    messages: list[str] = []
    if papers_path.exists():
        df = load_papers_dataframe(papers_path)
        extra_cols = [c for c in df.columns if c not in RESULT_COLUMNS]
        if extra_cols:
            messages.append(f"papers.csv 추가 열 유지: {', '.join(extra_cols)}")
        return df, messages, False

    df, boot_msgs = bootstrap_papers_from_results(results_dir, papers_path)
    messages.extend(boot_msgs)
    return df, messages, True


def existing_paper_index(df: pd.DataFrame) -> dict[str, str]:
    index: dict[str, str] = {}
    if df.empty:
        return index
    for record in df.to_dict(orient="records"):
        base = base_arxiv_id(str(record.get("arxiv_id", "")))
        if not base:
            continue
        version = str(record.get("arxiv_버전", "") or "").strip()
        if not version:
            version = extract_arxiv_version(str(record.get("논문_페이지_URL", "")))
        index[base] = version
    return index


def partition_search_results(
    api_rows: list[dict[str, str]],
    existing_index: dict[str, str],
    logger: RunLogger | None = None,
) -> tuple[list[dict[str, str]], dict[str, int | list[str]]]:
    stats: dict[str, int | list[str]] = {
        "queried": len(api_rows),
        "existing_dup": 0,
        "batch_dup": 0,
        "new_saved": 0,
        "new_versions": [],
    }
    new_rows: list[dict[str, str]] = []
    seen_batch: set[str] = set()

    for row in api_rows:
        base = base_arxiv_id(row.get("arxiv_id", ""))
        version = row.get("arxiv_버전") or extract_arxiv_version(
            row.get("논문_페이지_URL", "")
        )
        row["arxiv_id"] = base
        row["arxiv_버전"] = version

        if base in seen_batch:
            stats["batch_dup"] = int(stats["batch_dup"]) + 1
            if logger:
                logger.log_duplicate_skip(
                    base,
                    row.get("논문_페이지_URL", ""),
                    "이번 검색 내부 중복",
                )
            continue
        seen_batch.add(base)

        if base in existing_index:
            stats["existing_dup"] = int(stats["existing_dup"]) + 1
            if logger:
                logger.log_duplicate_skip(
                    base,
                    row.get("논문_페이지_URL", ""),
                    "papers.csv 기존 중복",
                )
            stored_version = existing_index[base]
            if version and stored_version and version != stored_version:
                new_versions = stats["new_versions"]
                assert isinstance(new_versions, list)
                new_versions.append(
                    f"{base}: 저장={stored_version or '(없음)'}, 조회={version}"
                )
            continue

        for col in RESULT_COLUMNS:
            row.setdefault(col, "")
        new_rows.append(row)

    stats["new_saved"] = len(new_rows)
    return new_rows, stats


def dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "arxiv_id" in out.columns:
        out["arxiv_id"] = out["arxiv_id"].astype(str).map(base_arxiv_id)
    out.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
    )


def save_papers_csv_atomic(df: pd.DataFrame, papers_path: Path) -> None:
    papers_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{papers_path.stem}_",
        dir=papers_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        dataframe_to_csv(df, tmp_path)
        os.replace(tmp_path, papers_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def append_new_papers(existing: pd.DataFrame, new_rows: list[dict[str, str]]) -> pd.DataFrame:
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    for col in RESULT_COLUMNS:
        if col not in combined.columns:
            combined[col] = ""
    ordered = [c for c in combined.columns if c in RESULT_COLUMNS] + [
        c for c in combined.columns if c not in RESULT_COLUMNS
    ]
    return combined[ordered]


def print_run_summary(stats: dict[str, int | list[str]]) -> None:
    queried = int(stats["queried"])
    existing_dup = int(stats["existing_dup"])
    batch_dup = int(stats["batch_dup"])
    new_saved = int(stats["new_saved"])
    print(
        f"\n조회 {queried}편 중 기존 중복 {existing_dup}편, "
        f"이번 검색 내부 중복 {batch_dup}편, 신규 저장 {new_saved}편"
    )
    new_versions = stats.get("new_versions", [])
    if isinstance(new_versions, list) and new_versions:
        print("새 arXiv 버전 감지 (기본 ID 동일, 버전만 다름 — papers.csv에는 추가하지 않음):")
        for note in new_versions:
            print(f"  - {note}")


class ArxivClient:
    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        clock: Clock | None = None,
        session: requests.Session | None = None,
        get_impl: Callable[..., requests.Response] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.clock = clock or RealClock()
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", USER_AGENT)
        self._get_impl = get_impl or self._session.get
        self._last_request_start: float | None = None

    def fetch_url(
        self,
        url: str,
        *,
        arxiv_id: str = "-",
        logger: RunLogger | None = None,
        fail_fast: bool = False,
    ) -> HttpFetchOutcome:
        """순차 GET. 첫 요청 즉시, 이후 직전 요청 시작으로부터 REQUEST_INTERVAL_SEC."""
        last_error_kind = ""
        last_error_message = ""
        retried = False

        for attempt in range(1, self.max_retries + 1):
            wait_request_interval(self.clock, self._last_request_start)
            request_start = self.clock.monotonic()
            self._last_request_start = request_start

            try:
                response = self._get_impl(url, timeout=60)
            except requests.Timeout as exc:
                last_error_kind = "timeout"
                last_error_message = str(exc) or "요청 시간 초과"
            except requests.RequestException as exc:
                last_error_kind = "connection"
                last_error_message = str(exc) or "연결 오류"

            else:
                if response.status_code == 429:
                    last_error_kind = "http_429"
                    last_error_message = "Too Many Requests"
                    if attempt < self.max_retries:
                        retried = True
                        wait = parse_retry_after(response) or MIN_RETRY_WAIT_SEC
                        wait = max(wait, MIN_RETRY_WAIT_SEC)
                        if logger:
                            logger.log_retry(
                                arxiv_id, url, attempt, last_error_kind, f"Retry-After {wait:.0f}s"
                            )
                        self.clock.sleep(wait)
                        continue
                    return HttpFetchOutcome(
                        success=False,
                        attempts=attempt,
                        error_kind=last_error_kind,
                        error_message=last_error_message,
                        retried=retried,
                    )

                if response.status_code in NON_RETRYABLE_HTTP:
                    kind = f"http_{response.status_code}"
                    msg = f"HTTP {response.status_code}"
                    if fail_fast:
                        raise ArxivApiError(f"{msg}: {url}")
                    return HttpFetchOutcome(
                        success=False,
                        attempts=attempt,
                        error_kind=kind,
                        error_message=msg,
                        retried=retried,
                    )

                if response.status_code >= 500:
                    last_error_kind = "server"
                    last_error_message = f"HTTP {response.status_code}"
                elif not response.ok:
                    last_error_kind = f"http_{response.status_code}"
                    last_error_message = f"HTTP {response.status_code}"
                    if fail_fast:
                        raise ArxivApiError(f"{last_error_message}: {url}")
                    return HttpFetchOutcome(
                        success=False,
                        attempts=attempt,
                        error_kind=last_error_kind,
                        error_message=last_error_message,
                        retried=retried,
                    )
                else:
                    return HttpFetchOutcome(
                        success=True,
                        text=response.text,
                        attempts=attempt,
                        retried=retried,
                    )

            if attempt < self.max_retries:
                retried = True
                backoff = retry_backoff_seconds(attempt)
                if logger:
                    logger.log_retry(
                        arxiv_id, url, attempt, last_error_kind, last_error_message
                    )
                self.clock.sleep(backoff)
                continue

            outcome = HttpFetchOutcome(
                success=False,
                attempts=attempt,
                error_kind=last_error_kind,
                error_message=last_error_message,
                retried=retried,
            )
            if fail_fast:
                if outcome.error_kind == "http_400":
                    raise ArxivApiError(
                        "요청 형식이 올바르지 않습니다. 검색어를 줄이거나 "
                        "다른 표현으로 바꿔 보세요."
                    )
                raise ArxivApiError(
                    f"arXiv 요청 실패 ({last_error_kind}): {last_error_message} "
                    f"(재시도 {self.max_retries}회)"
                )
            return outcome

        return HttpFetchOutcome(
            success=False,
            attempts=self.max_retries,
            error_kind=last_error_kind or "unknown",
            error_message=last_error_message or "알 수 없는 오류",
            retried=retried,
        )

    def fetch_page(self, search_query: str, start: int, max_results: int) -> str:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": API_SORT_BY,
            "sortOrder": API_SORT_ORDER,
        }
        url = f"{API_URL}?{urlencode(params)}"
        outcome = self.fetch_url(url, arxiv_id="-", fail_fast=False)
        if outcome.success:
            return outcome.text
        if outcome.error_kind == "http_400":
            raise ArxivApiError(
                "요청 형식이 올바르지 않습니다. 검색어를 줄이거나 "
                "다른 표현으로 바꿔 보세요."
            )
        raise ArxivApiError(
            f"arXiv API 요청 실패 ({outcome.error_kind}): {outcome.error_message} "
            f"(재시도 {outcome.attempts}회)"
        )

    def search(self, search_query: str, count: int) -> list[dict[str, str]]:
        if count < 1:
            raise ValueError("수집 개수는 1 이상이어야 합니다.")

        collected: list[dict[str, str]] = []
        start = 0
        total_available: int | None = None

        while len(collected) < count:
            page_max = min(MAX_RESULTS_PER_REQUEST, count - len(collected))
            xml_text = self.fetch_page(search_query, start=start, max_results=page_max)
            total, rows = parse_feed(xml_text)

            if total_available is None:
                total_available = total

            if total_available == 0 or not rows:
                if start == 0:
                    return []
                break

            collected.extend(rows)
            start += len(rows)

            if len(rows) == 0 or start >= total_available:
                break

        return collected[:count]


def sanitize_filename_part(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return (cleaned or "search")[:max_len]


def save_csv(rows: list[dict[str, str]], output_dir: Path, query_label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"arxiv_{sanitize_filename_part(query_label)}_{stamp}.csv"
    path = output_dir / filename
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df["arxiv_id"] = df["arxiv_id"].astype(str)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
    )
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="arXiv 공식 API로 논문 메타데이터를 검색·수집합니다.",
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help='검색어 (예: "mathematics education"). 고급 API 문법도 사용 가능.',
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=10,
        help="수집할 논문 개수 (기본: 10)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="CSV 저장 폴더 (기본: arxiv_crawler/results)",
    )
    parser.add_argument(
        "--skip-detail-titles",
        action="store_true",
        help="상세페이지 HTML 제목 수집을 건너뜁니다 (API만 사용).",
    )
    return parser.parse_args(argv)


def parse_published_datetime(published: str) -> datetime | None:
    text = (published or "").strip()
    if not text:
        return None
    cleaned = text.replace("T", " ").replace(" UTC", "").strip()
    try:
        return datetime.strptime(cleaned[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def query_row_save_status(
    row: dict[str, str], existing_index: dict[str, str], seen_batch: set[str]
) -> str:
    base = base_arxiv_id(row.get("arxiv_id", ""))
    if base in seen_batch:
        return "내부 중복(미저장)"
    seen_batch.add(base)
    if base in existing_index:
        return "papers.csv 중복(미저장)"
    return "신규(저장 대상)"


def print_api_query_verification(
    api_rows: list[dict[str, str]], existing_index: dict[str, str]
) -> bool:
    """API 반환 순서·등록일 검증 표 (저장 여부와 무관하게 전체 조회 행 표시)."""
    print(
        f"\n[API 조회 검증] sortBy={API_SORT_BY}, sortOrder={API_SORT_ORDER} "
        f"(최신 등록순, 서버 반환 순서)"
    )
    print("| # | arxiv_id | 최초_등록일 | 저장 | 제목 |")
    print("| --- | --- | --- | --- | --- |")
    seen: set[str] = set()
    parsed_dates: list[datetime] = []
    for i, row in enumerate(api_rows, start=1):
        status = query_row_save_status(row, existing_index, seen)
        title = (row.get("제목") or "").replace("|", "\\|")
        pub = row.get("최초_등록일", "")
        dt = parse_published_datetime(pub)
        if dt:
            parsed_dates.append(dt)
        short_title = title if len(title) <= 60 else title[:57] + "..."
        print(f"| {i} | {row.get('arxiv_id', '')} | {pub} | {status} | {short_title} |")
    if len(parsed_dates) < 2:
        print("최초_등록일 내림차순(최신순): 조회 건수 부족으로 생략")
        return True
    descending = all(
        parsed_dates[i] >= parsed_dates[i + 1] for i in range(len(parsed_dates) - 1)
    )
    print(f"최초_등록일 내림차순(최신순): {'예' if descending else '아니오'}")
    return descending


def print_detail_title_table(rows: list[dict[str, str]]) -> None:
    print("\n| URL | API 제목 | 상세페이지 제목 | 일치 |")
    print("| --- | --- | --- | --- |")
    for row in rows:
        url = row.get("논문_페이지_URL", "")
        api_t = row.get("제목", "").replace("|", "\\|")
        detail_t = (row.get("상세페이지_제목") or "").replace("|", "\\|")
        match = row.get("제목_일치여부", "")
        print(f"| {url} | {api_t} | {detail_t} | {match} |")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_started = time.monotonic()
    base_dir = Path(__file__).resolve().parent
    logger = create_run_logger(base_dir)
    log_path = logger._path

    try:
        search_query = build_search_query(args.query)
    except ValueError as exc:
        print(f"입력 오류: {exc}", file=sys.stderr)
        logger.close()
        return 1

    if args.count < 1:
        print("입력 오류: 수집 개수(-n/--count)는 1 이상이어야 합니다.", file=sys.stderr)
        logger.close()
        return 1

    print(f"검색어: {args.query}")
    print(f"API search_query: {search_query}")
    print(
        f"API 조회 {args.count}편 | 정렬 {API_SORT_BY}/{API_SORT_ORDER} (최신 등록순) | "
        f"요청 간격 {REQUEST_INTERVAL_SEC}초(직전 요청 시작 기준) | "
        f"최대 재시도 {MAX_RETRIES}회"
    )

    client = ArxivClient()
    papers_path = args.results_dir / PAPERS_CSV_NAME
    papers_existed_before = papers_path.exists()

    try:
        existing_df, init_messages, _bootstrapped = load_or_initialize_papers(args.results_dir)
    except ArxivApiError as exc:
        print(exc, file=sys.stderr)
        logger.close()
        return 1
    for msg in init_messages:
        print(msg)

    existing_index = existing_paper_index(existing_df)

    try:
        api_rows = client.search(search_query, args.count)
    except ArxivApiError as exc:
        print(f"\nAPI 검색 실패: {exc}", file=sys.stderr)
        print(
            "인터넷 연결·검색어·arXiv 서비스 상태를 확인한 뒤 다시 시도해 주세요.",
            file=sys.stderr,
        )
        logger.close()
        return 1

    if not api_rows:
        print(
            "\n검색 결과가 없습니다.\n"
            "- 검색어를 더 일반적인 단어로 바꿔 보세요.\n"
            '- 여러 단어는 따옴표 없이 입력하면 구문 검색(all:"...")으로 처리됩니다.\n'
            "- 고급 검색은 API 문법을 그대로 입력할 수 있습니다 (예: ti:education AND cat:math.HO)."
        )
        if not papers_existed_before and not existing_df.empty:
            try:
                save_papers_csv_atomic(existing_df, papers_path)
            except OSError as exc:
                print(f"\npapers.csv 저장 실패: {exc}", file=sys.stderr)
                logger.close()
                return 1
            print(f"\n신규 검색 0편 — 기존 results CSV 병합본만 papers.csv로 저장: {papers_path}")
        elapsed = time.monotonic() - run_started
        print(f"\n총 소요 시간: {format_duration(elapsed)}")
        print(f"로그: {log_path}")
        logger.close()
        return 0

    if len(api_rows) < args.count:
        print(f"\n요청한 {args.count}건 중 {len(api_rows)}건만 찾았습니다.")

    print_api_query_verification(api_rows, existing_index)

    new_rows, stats = partition_search_results(api_rows, existing_index, logger=logger)
    dup_total = int(stats["existing_dup"]) + int(stats["batch_dup"])
    detail_targets = len(new_rows) if not args.skip_detail_titles else 0

    print(
        f"\n[계획] API 조회 {stats['queried']}편 | "
        f"중복 제외 {dup_total}편(기존 {stats['existing_dup']} + "
        f"내부 {stats['batch_dup']}) | "
        f"상세페이지 처리 대상 {detail_targets}편"
    )
    print_run_summary(stats)

    detail_failures = 0
    if new_rows and not args.skip_detail_titles:
        print(f"\n상세페이지 제목 수집 시작 (간격 {REQUEST_INTERVAL_SEC}초, 순차 처리)")
        progress = DetailProgress(clock=client.clock, run_start=client.clock.monotonic())
        detail_failures = enrich_rows_with_detail_titles(
            client, new_rows, logger=logger, progress=progress
        )
        print_detail_title_table(new_rows)
    elif new_rows:
        for row in new_rows:
            row["상세페이지_제목"] = ""
            row["제목_일치여부"] = ""
            row["상세페이지_수집상태"] = "건너뜀"

    should_save = bool(new_rows) or (not papers_existed_before and not existing_df.empty)
    if should_save:
        try:
            combined = append_new_papers(existing_df, new_rows)
            save_papers_csv_atomic(combined, papers_path)
        except OSError as exc:
            print(f"\npapers.csv 저장 실패: {exc}", file=sys.stderr)
            logger.close()
            return 1
    else:
        print("\n신규 논문이 없어 papers.csv는 변경하지 않았습니다.")

    elapsed = time.monotonic() - run_started
    print(
        f"\n[완료] 신규 저장 {int(stats['new_saved'])}편 | "
        f"중복 제외 {dup_total}편 | "
        f"상세페이지 실패 {detail_failures}편 | "
        f"총 소요 {format_duration(elapsed)}"
    )
    if should_save:
        print(f"CSV: {papers_path}")
    print(f"로그: {log_path}")
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
