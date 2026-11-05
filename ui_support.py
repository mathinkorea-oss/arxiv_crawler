"""Streamlit UI 공통: 잠금, 모델 env, 오류 정리."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout

import run_summary as rs
import summarize_paper as sp

LOCK_DIR_NAME = ".ui_locks"


class JobBusyError(Exception):
    """다른 세션/작업이 같은 잠금을 보유 중."""


def lock_dir() -> Path:
    path = sp.project_root() / "results" / LOCK_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def lock_path(job_name: str) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", job_name)
    return lock_dir() / f"{safe}.lock"


@contextmanager
def cross_process_job_lock(job_name: str, *, timeout: float = 0) -> Iterator[None]:
    """여러 Streamlit 세션/프로세스 간 CSV·배치 작업 직렬화."""
    fl = FileLock(str(lock_path(job_name)))
    try:
        fl.acquire(timeout=timeout)
    except Timeout as exc:
        raise JobBusyError(
            f"다른 실행이 '{job_name}' 작업을 사용 중입니다. "
            "완료 후 다시 시도하세요."
        ) from exc
    try:
        yield
    finally:
        fl.release()


@contextmanager
def temporary_gemini_model(selected: str | None) -> Iterator[tuple[str, str]]:
    """
    UI에서 고른 모델을 일시 적용하고 종료 시 복원.
    Yields (선택값 표시용, 실제 resolve_model_name() 결과).
    """
    prev = os.environ.get("GEMINI_MODEL")
    display_selected = (selected or "").strip()
    try:
        if display_selected:
            os.environ["GEMINI_MODEL"] = display_selected
        sp._reset_client_for_tests()
        resolved = rs.resolve_model_name()
        yield display_selected or "(환경/기본값)", resolved
    finally:
        if prev is None:
            os.environ.pop("GEMINI_MODEL", None)
        else:
            os.environ["GEMINI_MODEL"] = prev
        sp._reset_client_for_tests()


def sanitize_user_error(message: str) -> str:
    """API 키 등 민감 문자열을 UI에 노출하지 않음."""
    sp._ensure_env_loaded()
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    text = str(message)
    if key:
        text = text.replace(key, "[REDACTED]")
    text = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED]", text)
    if "GEMINI_API_KEY" in text and "=" in text:
        text = re.sub(
            r"GEMINI_API_KEY\s*=\s*\S+",
            "GEMINI_API_KEY=[REDACTED]",
            text,
            flags=re.IGNORECASE,
        )
    return text


def friendly_summarize_error(exc: BaseException) -> str:
    if isinstance(exc, sp.SummarizeConfigError):
        return (
            "Gemini API 키가 설정되어 있지 않습니다. "
            "로컬은 `.env.local`, Streamlit Cloud는 앱 Secrets에 GEMINI_API_KEY를 설정하세요."
        )
    if isinstance(exc, sp.SummarizeAuthError):
        return "Gemini API 인증에 실패했습니다. API 키를 확인하세요."
    if isinstance(exc, sp.SummarizeRateLimitError):
        return "Gemini API 사용량 제한에 걸렸습니다. 잠시 후 다시 시도하세요."
    if isinstance(exc, sp.SummarizeModelError):
        return sanitize_user_error(str(exc))
    if isinstance(exc, sp.SummarizeError):
        return sanitize_user_error(str(exc))
    return sanitize_user_error(str(exc))
