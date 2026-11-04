"""batch_summaries.py 모의 테스트 (API 호출 없음)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import batch_summaries as bs
import summarize_paper as sp


def _paper_row(aid: str, title: str, abstract: str, ver: str = "v1") -> dict:
    return {
        "arxiv_id": aid,
        "arxiv_버전": ver,
        "제목": title,
        "초록": abstract,
    }


MOCK_SUMMARY = {
    "summary_one_line": "테스트 한 줄 요약",
    "summary_easy": "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
    "real_world": "설계 참고 가능성이 있습니다. 검증된 학습 효과는 초록에 없습니다.",
    "limitations": "초록에 연구의 한계가 명시되어 있지 않습니다.",
}


def _saved_row(
    aid: str,
    title: str,
    abstract: str,
    model: str = "test-model",
    *,
    prompt_version: str | None = None,
) -> dict:
    ih = sp.compute_input_hash(title, abstract)
    return {
        "arxiv_id": aid,
        "arxiv_버전": "v1",
        "제목": title,
        **MOCK_SUMMARY,
        "model": model,
        "summarized_at_utc": "2026-11-04 00:00:00 UTC",
        "prompt_version": prompt_version or sp.PROMPT_VERSION,
        "input_hash": ih,
    }


class BatchSummariesTests(unittest.TestCase):
    def test_recovery_filename_uses_full_key_hash_not_arxiv_id(self):
        row = _saved_row("math-ph/991001", "T", "A")
        key_hash = bs.summary_dedup_key_hash(
            arxiv_id=row["arxiv_id"],
            arxiv_version=row["arxiv_버전"],
            input_hash=row["input_hash"],
            model=row["model"],
            prompt_version=row["prompt_version"],
        )
        path = bs.recovery_artifact_path_for_row(row)
        self.assertEqual(path.name, f"{key_hash}.json")
        self.assertNotIn("/", path.name)

    def test_restore_on_start_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            pd.DataFrame([_paper_row("math-ph/991001", "T1", "A1")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            row = _saved_row("math-ph/991001", "T1", "A1")
            bs.write_recovery_artifact(row, recovery_root=recovery)

            def fail_api(title: str, abstract: str) -> dict:
                raise AssertionError("API should not be called")

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats = bs.run_batch_summaries(
                    limit=3,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=fail_api,
                )

            self.assertEqual(stats.recovery_restored, 1)
            self.assertEqual(stats.success, 0)
            self.assertEqual(stats.attempts, 0)
            self.assertTrue(summaries.is_file())
            self.assertEqual(len(list(recovery.glob("*.json"))), 0)

    def test_restore_after_csv_save_failed_leaves_json_then_rerun_no_dup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            pd.DataFrame([_paper_row("9", "T9", "A9")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            row = _saved_row("9", "T9", "A9")

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                with patch.object(
                    bs,
                    "save_summaries_csv_atomic",
                    side_effect=OSError("disk full"),
                ):
                    stats = bs.run_batch_summaries(
                        limit=1,
                        papers_path=papers,
                        summaries_path=summaries,
                        recovery_root=recovery,
                        summarize_fn=lambda t, a: dict(MOCK_SUMMARY),
                    )
            self.assertEqual(stats.failed, 1)
            self.assertEqual(len(list(recovery.glob("*.json"))), 1)

            def fail_api(title: str, abstract: str) -> dict:
                raise AssertionError("API should not be called on restore rerun")

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats2 = bs.run_batch_summaries(
                    limit=3,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=fail_api,
                )
            self.assertEqual(stats2.recovery_restored, 1)
            self.assertEqual(stats2.attempts, 0)
            df = pd.read_csv(summaries, dtype={"arxiv_id": str}, encoding="utf-8-sig")
            self.assertEqual(len(df), 1)

    def test_csv_saved_recovery_left_simulates_interrupt_then_no_dup(self):
        """CSV 저장 후 복구 파일 삭제 전 중단: CSV+JSON 공존 시 재실행."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            pd.DataFrame([_paper_row("9", "T9", "A9")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            row = _saved_row("9", "T9", "A9")
            existing = pd.DataFrame([row])
            bs.save_summaries_csv_atomic(existing, summaries)
            bs.write_recovery_artifact(row, recovery_root=recovery)

            def fail_api(title: str, abstract: str) -> dict:
                raise AssertionError("API should not be called")

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats = bs.run_batch_summaries(
                    limit=3,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=fail_api,
                )
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.recovery_restored, 0)
            df = pd.read_csv(summaries, dtype={"arxiv_id": str}, encoding="utf-8-sig")
            self.assertEqual(len(df), 1)
            self.assertEqual(len(list(recovery.glob("*.json"))), 0)

    def test_corrupt_recovery_json_aborts_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            recovery.mkdir(parents=True)
            bad = recovery / "deadbeef.json"
            bad.write_text("{not json", encoding="utf-8")
            pd.DataFrame([_paper_row("1", "T", "A")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                with self.assertRaises(bs.BatchAbortError):
                    bs.run_batch_summaries(
                        limit=1,
                        papers_path=papers,
                        summaries_path=summaries,
                        recovery_root=recovery,
                        summarize_fn=lambda t, a: dict(MOCK_SUMMARY),
                    )
            self.assertTrue(bad.is_file())

    def test_attempt_limit_after_skips_not_first_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            rows = [
                _paper_row("1", "T1", "A1"),
                _paper_row("2", "T2", "A2"),
                _paper_row("3", "T3", "A3"),
                _paper_row("4", "T4", "A4"),
            ]
            pd.DataFrame(rows).to_csv(papers, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    _saved_row("1", "T1", "A1"),
                    _saved_row("2", "T2", "A2"),
                    _saved_row("3", "T3", "A3"),
                ]
            ).to_csv(summaries, index=False, encoding="utf-8-sig")

            calls: list[str] = []

            def fake_summarize(title: str, abstract: str) -> dict:
                calls.append(title)
                return dict(MOCK_SUMMARY)

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats = bs.run_batch_summaries(
                    limit=3,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=fake_summarize,
                )

            self.assertEqual(stats.skipped, 3)
            self.assertEqual(stats.attempts, 1)
            self.assertEqual(stats.success, 1)
            self.assertEqual(calls, ["T4"])

    def test_api_request_counter_from_summarize_paper(self):
        counter: list[int] = []

        def counting_summarize(title: str, abstract: str) -> dict:
            counter.append(1)
            return dict(MOCK_SUMMARY)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            pd.DataFrame([_paper_row("1", "T1", "A1")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats = bs.run_batch_summaries(
                    limit=1,
                    papers_path=papers,
                    summaries_path=summaries,
                    summarize_fn=counting_summarize,
                )
            self.assertEqual(stats.success, 1)
            self.assertEqual(stats.api_requests, 0)

    def test_sanitize_log_redacts_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "SECRET_KEY_XYZ"}, clear=False):
            out = bs._sanitize_log_message("failed SECRET_KEY_XYZ here")
        self.assertNotIn("SECRET_KEY_XYZ", out)

    def test_skips_when_old_prompt_version_default_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            pd.DataFrame([_paper_row("1", "T1", "A1")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            old_row = _saved_row(
                "1",
                "T1",
                "A1",
                prompt_version="2026-11-04-guardrails-v2",
            )
            pd.DataFrame([old_row]).to_csv(
                summaries, index=False, encoding="utf-8-sig"
            )

            def fail_api(title: str, abstract: str) -> dict:
                raise AssertionError("API should not be called")

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats = bs.run_batch_summaries(
                    limit=1,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=fail_api,
                )
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.attempts, 0)
            df = pd.read_csv(summaries, dtype={"arxiv_id": str}, encoding="utf-8-sig")
            self.assertEqual(len(df), 1)
            self.assertEqual(
                df.iloc[0]["prompt_version"], "2026-11-04-guardrails-v2"
            )

    def test_resummarize_adds_row_preserves_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            pd.DataFrame([_paper_row("1", "T1", "A1")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            old_row = _saved_row(
                "1",
                "T1",
                "A1",
                model="old-model",
                prompt_version="2026-11-04-guardrails-v2",
            )
            pd.DataFrame([old_row]).to_csv(
                summaries, index=False, encoding="utf-8-sig"
            )

            with patch.object(bs.rs, "resolve_model_name", return_value="new-model"):
                stats = bs.run_batch_summaries(
                    limit=1,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=lambda t, a: dict(MOCK_SUMMARY),
                    resummarize_with_current_settings=True,
                )
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(stats.success, 1)
            df = pd.read_csv(
                summaries,
                dtype={"arxiv_id": str, "input_hash": str},
                encoding="utf-8-sig",
            )
            self.assertEqual(len(df), 2)
            versions = set(df["prompt_version"].astype(str))
            self.assertIn("2026-11-04-guardrails-v2", versions)
            self.assertIn(sp.PROMPT_VERSION, versions)

    def test_resummarize_skips_when_current_dedup_key_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            papers = tmp_path / "papers.csv"
            summaries = tmp_path / "paper_summaries.csv"
            recovery = tmp_path / "recovery"
            pd.DataFrame([_paper_row("1", "T1", "A1")]).to_csv(
                papers, index=False, encoding="utf-8-sig"
            )
            row = _saved_row("1", "T1", "A1", model="test-model")
            pd.DataFrame([row]).to_csv(
                summaries, index=False, encoding="utf-8-sig"
            )

            def fail_api(title: str, abstract: str) -> dict:
                raise AssertionError("API should not be called")

            with patch.object(bs.rs, "resolve_model_name", return_value="test-model"):
                stats = bs.run_batch_summaries(
                    limit=1,
                    papers_path=papers,
                    summaries_path=summaries,
                    recovery_root=recovery,
                    summarize_fn=fail_api,
                    resummarize_with_current_settings=True,
                )
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.attempts, 0)


if __name__ == "__main__":
    unittest.main()
