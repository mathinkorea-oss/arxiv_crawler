"""papers.csv 중복·저장 로직 테스트 (네트워크 불필요)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import crawler


class PapersDedupTests(unittest.TestCase):
    def test_partition_existing_and_batch_duplicates(self):
        existing = {"1111.00001": "v1", "2222.00002": "v1"}
        api_rows = [
            {"arxiv_id": "1111.00001", "arxiv_버전": "v1", "논문_페이지_URL": "https://arxiv.org/abs/1111.00001v1"},
            {"arxiv_id": "1111.00001", "arxiv_버전": "v1", "논문_페이지_URL": "https://arxiv.org/abs/1111.00001v1"},
            {"arxiv_id": "3333.00003", "arxiv_버전": "v1", "논문_페이지_URL": "https://arxiv.org/abs/3333.00003v1"},
        ]
        new_rows, stats = crawler.partition_search_results(api_rows, existing)
        self.assertEqual(stats["queried"], 3)
        self.assertEqual(stats["existing_dup"], 1)
        self.assertEqual(stats["batch_dup"], 1)
        self.assertEqual(stats["new_saved"], 1)
        self.assertEqual(len(new_rows), 1)
        self.assertEqual(new_rows[0]["arxiv_id"], "3333.00003")

    def test_new_version_notice(self):
        existing = {"2610.38990": "v1"}
        api_rows = [
            {
                "arxiv_id": "2610.38990",
                "arxiv_버전": "v2",
                "논문_페이지_URL": "https://arxiv.org/abs/2610.38990v2",
            }
        ]
        new_rows, stats = crawler.partition_search_results(api_rows, existing)
        self.assertEqual(new_rows, [])
        self.assertEqual(stats["existing_dup"], 1)
        self.assertTrue(stats["new_versions"])

    def test_zero_new_does_not_rewrite_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_path = Path(tmp) / "papers.csv"
            df = pd.DataFrame(
                [{"arxiv_id": "1000.0001", "arxiv_버전": "v1", "제목": "T"}]
            )
            crawler.save_papers_csv_atomic(df, papers_path)
            mtime_before = papers_path.stat().st_mtime
            existing = crawler.load_papers_dataframe(papers_path)
            new_rows, stats = crawler.partition_search_results(
                [
                    {
                        "arxiv_id": "1000.0001",
                        "arxiv_버전": "v1",
                        "논문_페이지_URL": "https://arxiv.org/abs/1000.0001v1",
                    }
                ],
                crawler.existing_paper_index(existing),
            )
            self.assertEqual(stats["new_saved"], 0)
            should_save = bool(new_rows)
            self.assertFalse(should_save)
            mtime_after = papers_path.stat().st_mtime
            self.assertEqual(mtime_before, mtime_after)

    def test_save_failure_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            papers_path = Path(tmp) / "papers.csv"
            original = pd.DataFrame([{"arxiv_id": "1000.0001", "제목": "keep"}])
            crawler.save_papers_csv_atomic(original, papers_path)
            before = papers_path.read_text(encoding="utf-8-sig")

            combined = crawler.append_new_papers(
                original,
                [{"arxiv_id": "2000.0002", "arxiv_버전": "v1", "제목": "new"}],
            )
            with patch("crawler.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    crawler.save_papers_csv_atomic(combined, papers_path)
            after = papers_path.read_text(encoding="utf-8-sig")
            self.assertEqual(before, after)

    def test_bootstrap_skips_test_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            (results / "_mock_overwrite_test").mkdir()
            mock_file = results / "_mock_overwrite_test" / "arxiv_mock_x.csv"
            mock_file.write_text("arxiv_id\n9999.9999\n", encoding="utf-8")
            real_file = results / "arxiv_sample_20260101_000000.csv"
            real_file.write_text(
                '"제목","저자","초록","최초_등록일","arxiv_id","논문_페이지_URL","PDF_URL","상세페이지_제목","제목_일치여부","상세페이지_수집상태"\n'
                '"T","A","S","D","8888.8888","https://arxiv.org/abs/8888.8888v1","https://arxiv.org/pdf/8888.8888v1","","",""\n',
                encoding="utf-8-sig",
            )
            df, messages = crawler.bootstrap_papers_from_results(results, results / "papers.csv")
            self.assertEqual(len(df), 1)
            self.assertEqual(str(df.iloc[0]["arxiv_id"]), "8888.8888")
            joined = "\n".join(messages)
            self.assertIn("arxiv_sample", joined)
            self.assertNotIn("병합 소스: arxiv_mock", joined)


if __name__ == "__main__":
    unittest.main()
