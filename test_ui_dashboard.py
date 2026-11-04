"""ui_dashboard 집계·파싱 테스트 (외부 API 없음)."""

from __future__ import annotations

import unittest

import pandas as pd

import summarize_paper as sp
import ui_dashboard as dash


class OverviewStatsTests(unittest.TestCase):
    def test_unique_base_id_and_pending(self):
        papers = pd.DataFrame(
            [
                {
                    "제목": "A",
                    "초록": "abstract one",
                    "arxiv_id": "2301.00001",
                    "arxiv_버전": "v1",
                    "최초_등록일": "2023-01-02",
                    "논문_페이지_URL": "https://arxiv.org/abs/2301.00001v1",
                },
                {
                    "제목": "B",
                    "초록": "abstract two",
                    "arxiv_id": "2301.00002",
                    "arxiv_버전": "v1",
                    "최초_등록일": "2023-01-03",
                    "논문_페이지_URL": "https://arxiv.org/abs/2301.00002",
                },
                {
                    "제목": "A v2",
                    "초록": "abstract one v2",
                    "arxiv_id": "2301.00001",
                    "arxiv_버전": "v2",
                    "최초_등록일": "2023-02-01",
                    "논문_페이지_URL": "https://arxiv.org/abs/2301.00001v2",
                },
            ]
        )
        stats = dash.compute_overview_stats(papers, pd.DataFrame())
        self.assertEqual(stats.crawled, 2)
        self.assertEqual(stats.summarized, 0)
        self.assertEqual(stats.pending, 2)
        self.assertTrue(stats.ok)

    def test_summarized_once_despite_multiple_models(self):
        title = "Paper"
        abstract = "Some abstract text here."
        arxiv_id = "2401.10000"
        version = "v1"
        ih = sp.compute_input_hash(title, abstract)
        papers = pd.DataFrame(
            [
                {
                    "제목": title,
                    "초록": abstract,
                    "arxiv_id": arxiv_id,
                    "arxiv_버전": version,
                    "최초_등록일": "2024-01-01",
                    "논문_페이지_URL": f"https://arxiv.org/abs/{arxiv_id}{version}",
                }
            ]
        )
        summaries = pd.DataFrame(
            [
                {
                    "arxiv_id": arxiv_id,
                    "arxiv_버전": version,
                    "input_hash": ih,
                    "model": "model-a",
                    "prompt_version": "p1",
                },
                {
                    "arxiv_id": arxiv_id,
                    "arxiv_버전": version,
                    "input_hash": ih,
                    "model": "model-b",
                    "prompt_version": "p2",
                },
            ]
        )
        stats = dash.compute_overview_stats(papers, summaries)
        self.assertEqual(stats.crawled, 1)
        self.assertEqual(stats.summarized, 1)
        self.assertEqual(stats.pending, 0)

    def test_missing_summary_columns_not_faked(self):
        papers = pd.DataFrame(
            [
                {
                    "제목": "T",
                    "초록": "ab",
                    "arxiv_id": "1",
                    "arxiv_버전": "v1",
                }
            ]
        )
        summaries = pd.DataFrame([{"arxiv_id": "1"}])
        stats = dash.compute_overview_stats(papers, summaries)
        self.assertFalse(stats.ok)
        self.assertTrue(any("열" in n for n in stats.notes))


class CrawlParseTests(unittest.TestCase):
    def test_parse_completion_line_and_table(self):
        stdout = """
검색어: math
API search_query: all:math
| # | arxiv_id | 최초_등록일 | 저장 | 제목 |
| --- | --- | --- | --- | --- |
| 1 | 2401.1 | 2024-01-01 | 신규 | Hello |
[완료] 신규 저장 1편 | 중복 제외 2편 | 상세페이지 실패 0편
"""
        parsed = dash.parse_crawl_stdout(stdout)
        self.assertEqual(parsed.new_saved, 1)
        self.assertEqual(parsed.dup_excluded, 2)
        self.assertEqual(len(parsed.paper_rows), 1)
        df = dash.crawl_results_dataframe(parsed)
        assert df is not None
        self.assertEqual(df.iloc[0]["제목"], "Hello")


if __name__ == "__main__":
    unittest.main()
