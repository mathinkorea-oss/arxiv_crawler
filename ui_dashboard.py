"""대시보드 집계·크롤 출력 파싱·표시용 HTML."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import crawler
import run_summary as rs
import summarize_paper as sp

MENU_ITEMS: tuple[str, ...] = (
    "개요",
    "수집",
    "1편 요약",
    "일괄 요약",
    "데이터",
)

SUMMARY_MATCH_COLUMNS = ("arxiv_id", "arxiv_버전", "input_hash")


@dataclass
class OverviewStats:
    crawled: int = 0
    summarized: int = 0
    pending: int = 0
    ok: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class CrawlRunDisplay:
    queried: int | None = None
    dup_excluded: int | None = None
    new_saved: int | None = None
    detail_failures: int | None = None
    requested_count: int | None = None
    paper_rows: list[dict[str, str]] = field(default_factory=list)
    detail_log: str = ""
    meta_lines: list[str] = field(default_factory=list)


def escape_html(text: str) -> str:
    return html.escape(str(text), quote=True)


def prose_html(text: str, *, css_class: str = "dash-prose") -> str:
    safe = escape_html(text).replace("\n", "<br>")
    return f'<div class="{css_class}">{safe}</div>'


def inject_dashboard_css() -> str:
    return """
<style>
  div.block-container {
    padding-top: 1.25rem;
    max-width: 1200px;
  }
  h1.dash-title {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.15rem !important;
    color: #0f172a;
  }
  p.dash-subtitle {
    color: #475569;
    font-size: 0.95rem;
    margin-top: 0;
    margin-bottom: 1rem;
  }
  section[data-testid="stSidebar"] {
    background-color: #EAF3FF !important;
    border-right: 1px solid #C5D9F0;
  }
  section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stRadio label span {
    color: #1e3a5f !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {
    border-radius: 8px;
    padding: 0.35rem 0.5rem;
    margin-bottom: 0.15rem;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
    background-color: #FFFFFF;
    font-weight: 700;
  }
  .dash-stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    min-height: 5.5rem;
    word-wrap: break-word;
    overflow-wrap: anywhere;
  }
  .dash-stat-card h3 {
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 0.35rem 0;
    color: #334155;
  }
  .dash-stat-card p {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    color: #0f172a;
  }
  .dash-prose {
    max-width: 900px;
    line-height: 1.65;
    color: #1e293b;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
  }
  .dash-one-line {
    max-width: 900px;
    line-height: 1.6;
    font-size: 1.05rem;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    white-space: normal;
  }
  .dash-paper-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0f172a;
    max-width: 900px;
    word-wrap: break-word;
    overflow-wrap: anywhere;
  }
  .dash-meta-small {
    font-size: 0.82rem;
    color: #64748b;
  }
  .dash-section-head {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 1rem;
    margin-bottom: 0.35rem;
    color: #0f172a;
  }
</style>
"""


def _canonical_paper_records(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in df.to_dict(orient="records"):
        normalized = crawler.normalize_loaded_row(record)
        if normalized:
            rows.append(normalized)
    return crawler.merge_rows_by_base_id(rows)


def summary_matches_paper(
    summaries: pd.DataFrame,
    *,
    arxiv_id: str,
    arxiv_version: str,
    input_hash: str,
) -> bool:
    if summaries.empty:
        return False
    missing = [c for c in SUMMARY_MATCH_COLUMNS if c not in summaries.columns]
    if missing:
        raise KeyError(missing)
    mask = (
        (summaries["arxiv_id"].astype(str) == str(arxiv_id))
        & (summaries["arxiv_버전"].astype(str) == str(arxiv_version))
        & (summaries["input_hash"].astype(str) == str(input_hash))
    )
    return bool(mask.any())


def compute_overview_stats(
    papers: pd.DataFrame | None,
    summaries: pd.DataFrame | None,
) -> OverviewStats:
    out = OverviewStats()
    if papers is None or papers.empty:
        return out

    if "arxiv_id" not in papers.columns:
        out.ok = False
        out.notes.append("papers.csv에 arxiv_id 열이 없어 집계할 수 없습니다.")
        return out

    bases = (
        papers["arxiv_id"]
        .astype(str)
        .map(lambda x: crawler.base_arxiv_id(x.strip()))
    )
    bases = bases[bases.str.strip().astype(bool)]
    out.crawled = int(bases.nunique())

    if summaries is None:
        summaries = pd.DataFrame()
    try:
        canonical = _canonical_paper_records(papers)
    except Exception as exc:
        out.ok = False
        out.notes.append(f"논문 데이터를 해석하지 못했습니다: {exc}")
        return out

    missing_sum_cols = [
        c for c in SUMMARY_MATCH_COLUMNS if c not in summaries.columns
    ]
    if summaries.empty and missing_sum_cols:
        missing_sum_cols = list(SUMMARY_MATCH_COLUMNS)

    eval_rows: list[dict[str, str]] = []
    for row in canonical:
        title = rs._cell_text(row.get("제목"))
        abstract = rs._cell_text(row.get("초록"))
        if not title or not abstract:
            continue
        eval_rows.append(row)

    if missing_sum_cols and not summaries.empty:
        out.ok = False
        out.notes.append(
            "요약 CSV에 필요한 열("
            + ", ".join(missing_sum_cols)
            + ")이 없어 완료·대기 집계를 할 수 없습니다. "
            "데이터를 확인해 주세요."
        )
        return out

    if missing_sum_cols and summaries.empty:
        out.pending = len(eval_rows)
        out.summarized = 0
        return out

    summarized = 0
    for row in eval_rows:
        arxiv_id = rs._cell_text(row.get("arxiv_id"))
        arxiv_version = rs._cell_text(row.get("arxiv_버전"))
        input_hash = sp.compute_input_hash(
            rs._cell_text(row.get("제목")),
            rs._cell_text(row.get("초록")),
        )
        if summary_matches_paper(
            summaries,
            arxiv_id=arxiv_id,
            arxiv_version=arxiv_version,
            input_hash=input_hash,
        ):
            summarized += 1

    out.summarized = summarized
    out.pending = max(0, len(eval_rows) - summarized)
    return out


def _parse_pub_date(value: str) -> tuple[Any, str]:
    text = (value or "").strip()
    if not text:
        return (0, "")
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(ts):
            return (text, text)
        return (ts, text)
    except Exception:
        return (text, text)


def recent_papers_table(
    papers: pd.DataFrame | None,
    summaries: pd.DataFrame | None,
    *,
    limit: int = 5,
) -> pd.DataFrame | None:
    if papers is None or papers.empty:
        return None
    needed = ("제목", "최초_등록일", "arxiv_id")
    if any(c not in papers.columns for c in needed):
        return None

    if summaries is None:
        summaries = pd.DataFrame()

    rows_out: list[dict[str, str]] = []
    for row in papers.to_dict(orient="records"):
        title = rs._cell_text(row.get("제목"))
        if not title:
            continue
        arxiv_id = rs._cell_text(row.get("arxiv_id"))
        version = rs._cell_text(row.get("arxiv_버전"))
        if not version:
            version = crawler.extract_arxiv_version(
                rs._cell_text(row.get("논문_페이지_URL"))
            )
        pub = rs._cell_text(row.get("최초_등록일"))
        url = rs._cell_text(row.get("논문_페이지_URL"))
        abstract = rs._cell_text(row.get("초록"))
        sort_key, pub_display = _parse_pub_date(pub)

        status = "확인 필요"
        if abstract and arxiv_id:
            try:
                ih = sp.compute_input_hash(title, abstract)
                if not summaries.empty and all(
                    c in summaries.columns for c in SUMMARY_MATCH_COLUMNS
                ):
                    status = (
                        "완료"
                        if summary_matches_paper(
                            summaries,
                            arxiv_id=crawler.base_arxiv_id(arxiv_id),
                            arxiv_version=version,
                            input_hash=ih,
                        )
                        else "대기"
                    )
                elif summaries.empty:
                    status = "대기"
            except KeyError:
                status = "확인 필요"
        elif not abstract:
            status = "초록 없음"

        rows_out.append(
            {
                "_sort": sort_key,
                "제목": title,
                "등록일": pub_display or pub,
                "요약": status,
                "원문": url,
            }
        )

    if not rows_out:
        return None
    rows_out.sort(key=lambda r: r["_sort"], reverse=True)
    trimmed = rows_out[:limit]
    df = pd.DataFrame(trimmed)
    return df.drop(columns=["_sort"])


def parse_crawl_stdout(stdout: str, stderr: str = "") -> CrawlRunDisplay:
    out = CrawlRunDisplay(detail_log=(stdout + stderr).strip())
    text = stdout or ""

    m_done = re.search(
        r"\[완료\]\s*신규 저장\s*(\d+)편\s*\|\s*중복 제외\s*(\d+)편\s*\|\s*상세페이지 실패\s*(\d+)편",
        text,
    )
    if m_done:
        out.new_saved = int(m_done.group(1))
        out.dup_excluded = int(m_done.group(2))
        out.detail_failures = int(m_done.group(3))

    m_plan = re.search(
        r"\[계획\]\s*API 조회\s*(\d+)편",
        text,
    )
    if m_plan:
        out.queried = int(m_plan.group(1))

    m_run = re.search(
        r"조회\s*(\d+)편 중 기존 중복\s*(\d+)편,\s*이번 검색 내부 중복\s*(\d+)편,\s*신규 저장\s*(\d+)편",
        text,
    )
    if m_run:
        if out.queried is None:
            out.queried = int(m_run.group(1))
        if out.new_saved is None:
            out.new_saved = int(m_run.group(4))

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("검색어:"):
            out.meta_lines.append(line)
        elif line.startswith("API search_query:"):
            out.meta_lines.append(line)
        elif line.startswith("API 조회") and "정렬" in line:
            out.meta_lines.append(line)
        elif line.startswith("|") and "| # |" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5 and parts[0].isdigit():
                out.paper_rows.append(
                    {
                        "번호": parts[0],
                        "arxiv_id": parts[1],
                        "최초_등록일": parts[2],
                        "처리": parts[3],
                        "제목": parts[4],
                    }
                )

    return out


def crawl_results_dataframe(display: CrawlRunDisplay) -> pd.DataFrame | None:
    if not display.paper_rows:
        return None
    rows: list[dict[str, str]] = []
    for r in display.paper_rows:
        aid = r.get("arxiv_id", "")
        url = ""
        if aid:
            url = f"https://arxiv.org/abs/{aid}"
        rows.append(
            {
                "제목": r.get("제목", ""),
                "최초 등록일": r.get("최초_등록일", ""),
                "처리 결과": r.get("처리", ""),
                "원문": url,
            }
        )
    return pd.DataFrame(rows)
