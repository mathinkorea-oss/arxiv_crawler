"""papers.csv에서 최대 N편 요약 후 paper_summaries.csv에 누적 저장."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

import run_summary as rs
import summarize_paper as sp

DEFAULT_BATCH_LIMIT = 3
RECOVERY_DIR_NAME = "_pending_summaries"

SUMMARY_CSV_COLUMNS = [
    "arxiv_id",
    "arxiv_버전",
    "제목",
    "summary_one_line",
    "summary_easy",
    "real_world",
    "limitations",
    "model",
    "summarized_at_utc",
    "prompt_version",
    "input_hash",
]

DEDUP_KEY_COLUMNS = (
    "arxiv_id",
    "arxiv_버전",
    "input_hash",
    "model",
    "prompt_version",
)


@dataclass
class BatchStats:
    attempts: int = 0
    api_requests: int = 0
    recovery_restored: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0


class SummarySaveError(Exception):
    """요약 CSV 저장 실패 (복구 파일은 남을 수 있음)."""


class BatchAbortError(Exception):
    """복구 파일 저장 불가 등으로 배치를 중단해야 할 때."""


def failure_log_path() -> Path:
    path = sp.project_root() / "logs" / "summary_failures.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_log_message(message: str) -> str:
    sp._ensure_env_loaded()
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    text = message
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED]", text)
    return text


def log_summary_failure(
    *,
    arxiv_id: str,
    arxiv_version: str,
    title: str,
    error_type: str,
    message: str,
    log_path: Path | None = None,
) -> None:
    path = log_path or failure_log_path()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    safe_msg = _sanitize_log_message(message)
    line = (
        f"{ts}\tarxiv_id={arxiv_id}\tarxiv_version={arxiv_version}\t"
        f"error_type={error_type}\tmessage={safe_msg}\t제목={title[:80]}\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def load_summaries_dataframe(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or sp.paper_summaries_csv_path()
    if not csv_path.is_file():
        return pd.DataFrame(columns=SUMMARY_CSV_COLUMNS)
    return pd.read_csv(
        csv_path,
        dtype={"arxiv_id": str, "arxiv_버전": str, "input_hash": str},
        encoding="utf-8-sig",
    )


def save_summaries_csv_atomic(df: pd.DataFrame, path: Path | None = None) -> None:
    csv_path = path or sp.paper_summaries_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f"{csv_path.stem}_",
        dir=csv_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        out = df.copy()
        for col in SUMMARY_CSV_COLUMNS:
            if col not in out.columns:
                out[col] = ""
        out = out[SUMMARY_CSV_COLUMNS]
        out.to_csv(
            tmp_path,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_NONNUMERIC,
        )
        os.replace(tmp_path, csv_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def has_successful_summary(
    existing: pd.DataFrame,
    *,
    arxiv_id: str,
    arxiv_version: str,
    input_hash: str,
    model: str,
    prompt_version: str,
) -> bool:
    """저장 dedup 키 전체(모델·프롬프트 포함)가 일치하는 행이 있는지."""
    if existing.empty:
        return False
    mask = (
        (existing["arxiv_id"].astype(str) == str(arxiv_id))
        & (existing["arxiv_버전"].astype(str) == str(arxiv_version))
        & (existing["input_hash"].astype(str) == str(input_hash))
        & (existing["model"].astype(str) == str(model))
        & (existing["prompt_version"].astype(str) == str(prompt_version))
    )
    if not mask.any():
        return False
    return _any_valid_summary_rows(existing.loc[mask])


def _row_has_valid_summary_fields(row: pd.Series) -> bool:
    for key in (
        "summary_one_line",
        "summary_easy",
        "real_world",
        "limitations",
    ):
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if not str(value).strip():
            return False
    return True


def _any_valid_summary_rows(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return False
    for _, row in rows.iterrows():
        if _row_has_valid_summary_fields(row):
            return True
    return False


def has_summary_for_paper_input(
    existing: pd.DataFrame,
    *,
    arxiv_id: str,
    arxiv_version: str,
    input_hash: str,
) -> bool:
    """
    일괄 요약 기본 건너뛰기: arxiv ID·논문 버전·입력 해시가 같고
    요약 본문이 유효한 행이 하나라도 있으면 True (model/prompt_version 무관).
    """
    if existing.empty:
        return False
    mask = (
        (existing["arxiv_id"].astype(str) == str(arxiv_id))
        & (existing["arxiv_버전"].astype(str) == str(arxiv_version))
        & (existing["input_hash"].astype(str) == str(input_hash))
    )
    if not mask.any():
        return False
    return _any_valid_summary_rows(existing.loc[mask])


def summary_dedup_key_hash(
    *,
    arxiv_id: str,
    arxiv_version: str,
    input_hash: str,
    model: str,
    prompt_version: str,
) -> str:
    """요약 중복·복구 파일명용 전체 키 해시 (arxiv ID에 '/'가 있어도 안전)."""
    payload = "\n".join(
        [arxiv_id, arxiv_version, input_hash, model, prompt_version]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pending_recovery_dir(recovery_root: Path | None = None) -> Path:
    if recovery_root is not None:
        return recovery_root
    return sp.project_root() / "results" / RECOVERY_DIR_NAME


def recovery_artifact_path_for_key(
    key_hash: str,
    *,
    recovery_root: Path | None = None,
) -> Path:
    return pending_recovery_dir(recovery_root) / f"{key_hash}.json"


def recovery_artifact_path_for_row(
    row: dict[str, Any],
    *,
    recovery_root: Path | None = None,
) -> Path:
    key_hash = summary_dedup_key_hash(
        arxiv_id=str(row["arxiv_id"]),
        arxiv_version=str(row["arxiv_버전"]),
        input_hash=str(row["input_hash"]),
        model=str(row["model"]),
        prompt_version=str(row["prompt_version"]),
    )
    return recovery_artifact_path_for_key(key_hash, recovery_root=recovery_root)


def iter_recovery_json_files(recovery_root: Path | None = None) -> Iterator[Path]:
    root = pending_recovery_dir(recovery_root)
    if not root.is_dir():
        return
    yield from sorted(root.glob("*.json"))


def read_recovery_row(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BatchAbortError(
            "손상된 복구 JSON입니다. 파일을 삭제하지 않고 배치를 중단합니다.\n"
            f"  경로: {path}\n"
            f"  원인: {exc}"
        ) from exc
    except OSError as exc:
        raise BatchAbortError(
            "복구 JSON을 읽을 수 없어 배치를 중단합니다.\n"
            f"  경로: {path}\n"
            f"  원인: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise BatchAbortError(
            f"복구 JSON 형식이 올바르지 않습니다. 파일을 삭제하지 않고 중단합니다.\n"
            f"  경로: {path}"
        )
    missing = [col for col in SUMMARY_CSV_COLUMNS if col not in data]
    if missing:
        raise BatchAbortError(
            "복구 JSON에 필수 필드가 없습니다. 파일을 삭제하지 않고 중단합니다.\n"
            f"  경로: {path}\n"
            f"  누락: {', '.join(missing)}"
        )
    return data


def remove_recovery_file(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)


def write_recovery_artifact(
    row: dict[str, Any],
    *,
    recovery_root: Path | None = None,
) -> Path:
    """API 응답을 CSV 반영 전 JSON으로 보존 (API 키 미포함)."""
    path = recovery_artifact_path_for_row(row, recovery_root=recovery_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".json",
        prefix="recovery_",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(row, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return path
    except OSError as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise BatchAbortError(
            "요약 복구 파일을 디스크에 저장할 수 없어 실행을 중단합니다. "
            f"경로: {path.parent}\n"
            f"원인: {exc}"
        ) from exc
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise BatchAbortError(
            "요약 복구 파일을 저장하는 중 오류가 발생해 실행을 중단합니다. "
            f"원인: {exc}"
        ) from exc


def append_row_if_not_duplicate(
    existing: pd.DataFrame,
    row: dict[str, Any],
    *,
    out_csv: Path,
) -> tuple[pd.DataFrame, bool]:
    """CSV에 동일 키가 없을 때만 추가·저장. (appended 여부 반환)"""
    if has_successful_summary(
        existing,
        arxiv_id=str(row["arxiv_id"]),
        arxiv_version=str(row["arxiv_버전"]),
        input_hash=str(row["input_hash"]),
        model=str(row["model"]),
        prompt_version=str(row["prompt_version"]),
    ):
        return existing, False
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    save_summaries_csv_atomic(updated, out_csv)
    return updated, True


def restore_row_from_recovery_path(
    existing: pd.DataFrame,
    path: Path,
    *,
    out_csv: Path,
) -> tuple[pd.DataFrame, bool]:
    """복구 JSON 1건을 CSV에 반영. 실패 시 BatchAbortError."""
    row = read_recovery_row(path)
    try:
        updated, appended = append_row_if_not_duplicate(
            existing, row, out_csv=out_csv
        )
    except Exception as exc:
        raise BatchAbortError(
            "복구 JSON을 paper_summaries.csv에 반영하지 못해 배치를 중단합니다.\n"
            f"  복구 파일: {path}\n"
            f"  원인: {exc}"
        ) from exc
    if appended:
        remove_recovery_file(path)
    else:
        remove_recovery_file(path)
    return updated, appended


def restore_all_pending_summaries(
    existing: pd.DataFrame,
    *,
    out_csv: Path,
    recovery_root: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    restored = 0
    for path in list(iter_recovery_json_files(recovery_root)):
        existing, appended = restore_row_from_recovery_path(
            existing, path, out_csv=out_csv
        )
        if appended:
            restored += 1
    return existing, restored


def persist_summary_row(
    existing: pd.DataFrame,
    new_row: dict[str, Any],
    *,
    out_csv: Path,
    recovery_root: Path | None = None,
) -> pd.DataFrame:
    """복구 JSON 저장 → CSV 저장 확인 → 복구 파일 삭제."""
    recovery_path = write_recovery_artifact(new_row, recovery_root=recovery_root)
    try:
        updated, appended = append_row_if_not_duplicate(
            existing, new_row, out_csv=out_csv
        )
    except Exception as exc:
        raise SummarySaveError(
            f"paper_summaries.csv 저장 실패 (복구 파일: {recovery_path}): {exc}"
        ) from exc
    if not appended:
        remove_recovery_file(recovery_path)
        return updated
    remove_recovery_file(recovery_path)
    return updated


def try_restore_pending_for_paper(
    existing: pd.DataFrame,
    *,
    arxiv_id: str,
    arxiv_version: str,
    input_hash: str,
    model: str,
    prompt_version: str,
    out_csv: Path,
    recovery_root: Path | None = None,
) -> tuple[pd.DataFrame, bool]:
    key_hash = summary_dedup_key_hash(
        arxiv_id=arxiv_id,
        arxiv_version=arxiv_version,
        input_hash=input_hash,
        model=model,
        prompt_version=prompt_version,
    )
    path = recovery_artifact_path_for_key(key_hash, recovery_root=recovery_root)
    if not path.is_file():
        return existing, False
    return restore_row_from_recovery_path(existing, path, out_csv=out_csv)


def build_summary_row(
    *,
    row: pd.Series,
    summary: dict[str, str],
    model: str,
    input_hash: str,
    summarized_at: datetime | None = None,
) -> dict[str, Any]:
    ts = summarized_at or datetime.now(timezone.utc)
    return {
        "arxiv_id": rs._cell_text(row.get("arxiv_id")),
        "arxiv_버전": rs._cell_text(row.get("arxiv_버전")),
        "제목": rs._cell_text(row.get("제목")),
        "summary_one_line": summary["summary_one_line"],
        "summary_easy": summary["summary_easy"],
        "real_world": summary["real_world"],
        "limitations": summary["limitations"],
        "model": model,
        "summarized_at_utc": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "prompt_version": sp.PROMPT_VERSION,
        "input_hash": input_hash,
    }


def run_batch_summaries(
    *,
    limit: int = DEFAULT_BATCH_LIMIT,
    papers_path: Path | None = None,
    summaries_path: Path | None = None,
    failure_log: Path | None = None,
    summarize_fn=None,
    recovery_root: Path | None = None,
    resummarize_with_current_settings: bool = False,
) -> BatchStats:
    """
    이미 저장된 요약은 건너뛰고, API 요약 시도를 최대 limit회 수행합니다.
    실패한 시도도 limit에 포함되며, limit만큼 시도하면 종료합니다.

    resummarize_with_current_settings=False(기본): ID·버전·input_hash가 같은
    유효 요약이 있으면 model/prompt_version과 무관하게 건너뜁니다.
    True: 현재 model·PROMPT_VERSION dedup 키가 CSV에 있을 때만 건너뜁니다
    (새 모델/프롬프트로 추가 행 저장 가능, 기존 행은 유지).
    """
    if limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다.")

    papers_csv = papers_path or sp.papers_csv_path()
    out_csv = summaries_path or sp.paper_summaries_csv_path()

    df = rs.load_papers_dataframe(papers_csv)
    rs.validate_papers_dataframe(df, csv_path=papers_csv)

    model = rs.resolve_model_name()
    prompt_version = sp.PROMPT_VERSION
    existing = load_summaries_dataframe(out_csv)
    stats = BatchStats()
    api_request_counter: list[int] = [0]

    existing, restored = restore_all_pending_summaries(
        existing, out_csv=out_csv, recovery_root=recovery_root
    )
    stats.recovery_restored += restored

    def _summarize(title: str, abstract: str) -> dict[str, str]:
        if summarize_fn is not None:
            return summarize_fn(title, abstract)
        return sp.summarize_paper(
            title,
            abstract,
            api_request_counter=api_request_counter,
        )

    for row in rs.iter_valid_paper_rows(df, limit=None):
        if stats.attempts >= limit:
            break

        title = rs._cell_text(row.get("제목"))
        abstract = rs._cell_text(row.get("초록"))
        arxiv_id = rs._cell_text(row.get("arxiv_id"))
        arxiv_version = rs._cell_text(row.get("arxiv_버전"))
        input_hash = sp.compute_input_hash(title, abstract)

        if resummarize_with_current_settings:
            already_done = has_successful_summary(
                existing,
                arxiv_id=arxiv_id,
                arxiv_version=arxiv_version,
                input_hash=input_hash,
                model=model,
                prompt_version=prompt_version,
            )
        else:
            already_done = has_summary_for_paper_input(
                existing,
                arxiv_id=arxiv_id,
                arxiv_version=arxiv_version,
                input_hash=input_hash,
            )
        if already_done:
            stats.skipped += 1
            continue

        existing, restored_one = try_restore_pending_for_paper(
            existing,
            arxiv_id=arxiv_id,
            arxiv_version=arxiv_version,
            input_hash=input_hash,
            model=model,
            prompt_version=prompt_version,
            out_csv=out_csv,
            recovery_root=recovery_root,
        )
        if restored_one:
            stats.recovery_restored += 1
            continue

        stats.attempts += 1
        try:
            summary = _summarize(title, abstract)
        except sp.SummarizeError as exc:
            stats.failed += 1
            log_summary_failure(
                arxiv_id=arxiv_id,
                arxiv_version=arxiv_version,
                title=title,
                error_type=type(exc).__name__,
                message=str(exc),
                log_path=failure_log,
            )
            continue

        new_row = build_summary_row(
            row=row,
            summary=summary,
            model=model,
            input_hash=input_hash,
        )
        try:
            existing = persist_summary_row(
                existing,
                new_row,
                out_csv=out_csv,
                recovery_root=recovery_root,
            )
        except BatchAbortError:
            raise
        except SummarySaveError as exc:
            stats.failed += 1
            log_summary_failure(
                arxiv_id=arxiv_id,
                arxiv_version=arxiv_version,
                title=title,
                error_type="SummarySaveError",
                message=str(exc),
                log_path=failure_log,
            )
            continue

        stats.success += 1

    stats.api_requests = api_request_counter[0]
    return stats


def format_stats_report(stats: BatchStats, *, attempt_limit: int) -> str:
    return (
        f"논문 시도 {stats.attempts}/{attempt_limit} | "
        f"API 요청 {stats.api_requests} | "
        f"복구 반영 {stats.recovery_restored} | "
        f"신규 성공 {stats.success} | "
        f"기존 제외 {stats.skipped} | "
        f"실패 {stats.failed}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="papers.csv에서 최대 N편 요약해 paper_summaries.csv에 저장",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_BATCH_LIMIT,
        help=f"API 요약 시도 횟수 상한 (기존 저장 제외, 실패 포함, 기본 {DEFAULT_BATCH_LIMIT})",
    )
    parser.add_argument(
        "--resummarize",
        action="store_true",
        help=(
            "현재 GEMINI_MODEL·프롬프트 버전으로 다시 요약 "
            "(동일 dedup 키가 없을 때만 API 호출, 기존 CSV 행 유지)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        stats = run_batch_summaries(
            limit=args.limit,
            resummarize_with_current_settings=args.resummarize,
        )
        print(format_stats_report(stats, attempt_limit=args.limit))
        return 0 if stats.failed == 0 else 3
    except rs.RunSummaryError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except BatchAbortError as exc:
        print(f"중단: {exc}", file=sys.stderr)
        return 4
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
