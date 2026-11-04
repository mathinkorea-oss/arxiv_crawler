"""results/papers.csv에서 논문 1편을 골라 제목·초록 기반 요약을 실행합니다."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

import summarize_paper as sp

# 공식 GA 모델 ID (https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite)
RECOMMENDED_MODEL_ID = "gemini-3.1-flash-lite"

REQUIRED_CSV_COLUMNS = ("제목", "초록", "arxiv_id")

SUMMARY_LABELS: dict[str, str] = {
    "summary_one_line": "한 줄 요약 (핵심 한 문장)",
    "summary_easy": "쉬운 설명 (3~4문장)",
    "real_world": "현실 활용 (2~3문장)",
    "limitations": "한계 (1~2문장)",
}


class RunSummaryError(Exception):
    """실행 전·후 사용자에게 안내할 오류."""


def resolve_model_name() -> str:
    """GEMINI_MODEL이 없을 때만 권장 모델 ID를 설정 (기존 환경 변수는 유지)."""
    import os

    existing = (os.environ.get("GEMINI_MODEL") or "").strip()
    if existing:
        return existing
    os.environ["GEMINI_MODEL"] = RECOMMENDED_MODEL_ID
    return RECOMMENDED_MODEL_ID


def _cell_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_papers_dataframe(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path if csv_path is not None else sp.papers_csv_path()
    if not path.is_file():
        raise RunSummaryError(
            f"논문 목록 파일을 찾을 수 없습니다.\n  경로: {path}\n"
            "먼저 crawler.py로 논문을 수집해 results/papers.csv를 만드세요."
        )
    try:
        df = pd.read_csv(path, dtype={"arxiv_id": str}, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise RunSummaryError(
            f"논문 목록 CSV가 비어 있습니다.\n  경로: {path}"
        ) from exc
    except Exception as exc:
        raise RunSummaryError(
            f"논문 목록 CSV를 읽을 수 없습니다.\n  경로: {path}\n  원인: {exc}"
        ) from exc
    return df


def validate_papers_dataframe(df: pd.DataFrame, *, csv_path: Path | None = None) -> None:
    path = csv_path if csv_path is not None else sp.papers_csv_path()
    if df.empty:
        raise RunSummaryError(
            f"논문 목록 CSV에 데이터 행이 없습니다.\n  경로: {path}"
        )
    missing = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise RunSummaryError(
            "논문 목록 CSV에 필요한 열이 없습니다.\n"
            f"  경로: {path}\n"
            f"  누락된 열: {', '.join(missing)}\n"
            f"  필요한 열: {', '.join(REQUIRED_CSV_COLUMNS)}"
        )


def select_first_valid_paper(df: pd.DataFrame) -> pd.Series:
    for row in iter_valid_paper_rows(df, limit=1):
        return row
    raise RunSummaryError(
        "제목과 초록이 모두 채워진 논문을 찾지 못했습니다.\n"
        "CSV의 각 행에 '제목'과 '초록' 값이 있는지 확인하세요."
    )


def iter_valid_paper_rows(df: pd.DataFrame, limit: int | None = None):
    """제목·초록이 유효한 행을 CSV 순서대로 최대 limit건 yield."""
    count = 0
    for _, row in df.iterrows():
        title = _cell_text(row.get("제목"))
        abstract = _cell_text(row.get("초록"))
        if not title or not abstract:
            continue
        yield row
        count += 1
        if limit is not None and count >= limit:
            break


def format_summary_report(
    *,
    arxiv_id: str,
    title: str,
    model: str,
    summary: dict[str, str],
) -> str:
    lines = [
        "=== 논문 요약 (제목·초록만 사용한 초록 기반 요약) ===",
        "",
        f"논문 ID (arxiv_id): {arxiv_id}",
        f"제목: {title}",
        f"사용 모델: {model}",
        "",
        "--- 요약 ---",
    ]
    for key in sp.SUMMARY_KEYS:
        label = SUMMARY_LABELS.get(key, key)
        lines.append(f"[{label}]")
        lines.append(summary[key])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_summary(*, csv_path: Path | None = None) -> str:
    """
    CSV에서 논문 1편을 요약하고 출력용 문자열을 반환합니다.
    API·CSV 오류는 RunSummaryError 또는 summarize_paper.SummarizeError로 전달됩니다.
    """
    path = csv_path if csv_path is not None else sp.papers_csv_path()
    df = load_papers_dataframe(path)
    validate_papers_dataframe(df, csv_path=path)
    row = select_first_valid_paper(df)

    model = resolve_model_name()
    title = _cell_text(row["제목"])
    abstract = _cell_text(row["초록"])
    arxiv_id = _cell_text(row.get("arxiv_id"))

    summary = sp.summarize_paper(title, abstract)
    return format_summary_report(
        arxiv_id=arxiv_id,
        title=title,
        model=model,
        summary=summary,
    )


def main() -> int:
    try:
        report = run_summary()
        print(report, end="")
        return 0
    except RunSummaryError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except sp.SummarizeError as exc:
        print(f"요약 API 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
