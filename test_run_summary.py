"""run_summary.py 실행 스크립트 테스트 (API 호출 없음)."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import run_summary as rs
import summarize_paper as sp


VALID_ROW = {
    "제목": "Sample Title",
    "초록": "This is a sample abstract for testing.",
    "arxiv_id": "2610.38990",
}

MOCK_SUMMARY = {
    "summary_one_line": "테스트 한 줄",
    "summary_easy": "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다.",
    "real_world": "활용 첫 문장입니다. 활용 둘째 문장입니다.",
    "limitations": "초록에 연구의 한계가 명시되어 있지 않습니다.",
}


class SelectPaperTests(unittest.TestCase):
    def test_select_first_valid_skips_empty_title(self):
        df = pd.DataFrame(
            [
                {"제목": "", "초록": "abc", "arxiv_id": "1"},
                VALID_ROW,
            ]
        )
        row = rs.select_first_valid_paper(df)
        self.assertEqual(row["arxiv_id"], "2610.38990")

    def test_select_none_raises(self):
        df = pd.DataFrame([{"제목": "", "초록": "x", "arxiv_id": "1"}])
        with self.assertRaises(rs.RunSummaryError) as ctx:
            rs.select_first_valid_paper(df)
        self.assertIn("찾지 못했습니다", str(ctx.exception))


class ValidateCsvTests(unittest.TestCase):
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=list(rs.REQUIRED_CSV_COLUMNS))
        with self.assertRaises(rs.RunSummaryError) as ctx:
            rs.validate_papers_dataframe(df, csv_path=Path("dummy.csv"))
        self.assertIn("데이터 행이 없습니다", str(ctx.exception))

    def test_missing_columns(self):
        df = pd.DataFrame([{"제목": "T", "arxiv_id": "1"}])
        with self.assertRaises(rs.RunSummaryError) as ctx:
            rs.validate_papers_dataframe(df)
        self.assertIn("초록", str(ctx.exception))


class RunSummaryIntegrationTests(unittest.TestCase):
    def test_run_summary_success_mocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "papers.csv"
            pd.DataFrame([VALID_ROW]).to_csv(csv_path, index=False, encoding="utf-8-sig")
            with patch.object(rs, "resolve_model_name", return_value=rs.RECOMMENDED_MODEL_ID):
                with patch.object(sp, "summarize_paper", return_value=MOCK_SUMMARY) as mock_sum:
                    report = rs.run_summary(csv_path=csv_path)
            mock_sum.assert_called_once_with(VALID_ROW["제목"], VALID_ROW["초록"])
            self.assertIn("2610.38990", report)
            self.assertIn("Sample Title", report)
            self.assertIn(rs.RECOMMENDED_MODEL_ID, report)
            self.assertIn("제목·초록만 사용", report)
            self.assertIn("한 줄 요약", report)

    def test_main_csv_missing_returns_1(self):
        with patch.object(sp, "papers_csv_path", return_value=Path("/nonexistent/papers.csv")):
            code = rs.main()
        self.assertEqual(code, 1)

    def test_main_api_error_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "papers.csv"
            pd.DataFrame([VALID_ROW]).to_csv(csv_path, index=False, encoding="utf-8-sig")
            with patch.object(sp, "papers_csv_path", return_value=csv_path):
                with patch.object(
                    sp,
                    "summarize_paper",
                    side_effect=sp.SummarizeModelError("모델 사용 불가"),
                ):
                    buf = io.StringIO()
                    with patch("sys.stderr", buf):
                        code = rs.main()
        self.assertEqual(code, 2)
        self.assertIn("요약 API 오류", buf.getvalue())

    def test_resolve_model_respects_existing_env(self):
        import os

        with patch.dict(os.environ, {"GEMINI_MODEL": "custom-model"}, clear=False):
            name = rs.resolve_model_name()
        self.assertEqual(name, "custom-model")


if __name__ == "__main__":
    unittest.main()
