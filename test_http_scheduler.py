"""HTTP 간격·오류 처리 테스트 (모의 응답·가상 시계)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import requests

import crawler


class MockClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class HttpSchedulerTests(unittest.TestCase):
    def test_request_interval_from_previous_start(self):
        clock = MockClock()
        responses = [
            Mock(status_code=200, ok=True, text="a", headers={}),
            Mock(status_code=200, ok=True, text="b", headers={}),
        ]

        def get_impl(url, timeout=60):
            return responses.pop(0)

        client = crawler.ArxivClient(clock=clock, get_impl=get_impl)
        r1 = client.fetch_url("https://example/1")
        r2 = client.fetch_url("https://example/2")
        self.assertTrue(r1.success and r2.success)
        self.assertGreaterEqual(clock.sleeps[0], crawler.REQUEST_INTERVAL_SEC)

    def test_404_no_retry(self):
        clock = MockClock()
        resp = Mock(status_code=404, ok=False, text="nf", headers={})
        client = crawler.ArxivClient(clock=clock, get_impl=lambda *a, **k: resp)
        out = client.fetch_url("https://arxiv.org/abs/missing")
        self.assertFalse(out.success)
        self.assertEqual(out.attempts, 1)
        self.assertEqual(out.error_kind, "http_404")

    def test_429_respects_retry_after(self):
        clock = MockClock()
        calls = {"n": 0}

        def get_impl(url, timeout=60):
            calls["n"] += 1
            if calls["n"] == 1:
                return Mock(status_code=429, ok=False, text="", headers={"Retry-After": "20"})
            return Mock(status_code=200, ok=True, text="ok", headers={})

        client = crawler.ArxivClient(clock=clock, get_impl=get_impl)
        out = client.fetch_url("https://example/x")
        self.assertTrue(out.success)
        self.assertGreaterEqual(max(clock.sleeps), 20.0)

    def test_server_error_retries_three_times(self):
        clock = MockClock()
        calls = {"n": 0}

        def get_impl(url, timeout=60):
            calls["n"] += 1
            return Mock(status_code=503, ok=False, text="", headers={})

        client = crawler.ArxivClient(clock=clock, get_impl=get_impl)
        out = client.fetch_url("https://example/x")
        self.assertFalse(out.success)
        self.assertEqual(out.attempts, 3)
        self.assertEqual(calls["n"], 3)

    def test_html_parse_error_distinct(self):
        with self.assertRaises(crawler.HtmlParseError):
            with unittest.mock.patch(
                "crawler.BeautifulSoup", side_effect=ValueError("bad html")
            ):
                crawler.extract_title_from_abs_html("<html>")

    def test_detail_failure_does_not_stop_batch(self):
        clock = MockClock()

        def get_impl(url, timeout=60):
            if "404" in url:
                return Mock(status_code=404, ok=False, text="", headers={})
            return Mock(status_code=200, ok=True, text="<html><h1 class='title mathjax'><span class='descriptor'>Title:</span>OK</h1></html>", headers={})

        client = crawler.ArxivClient(clock=clock, get_impl=get_impl)
        with tempfile.TemporaryDirectory() as tmp:
            logger = crawler.RunLogger(Path(tmp) / "t.log")
            rows = [
                {
                    "arxiv_id": "1.1",
                    "제목": "A",
                    "논문_페이지_URL": "https://arxiv.org/abs/404",
                },
                {
                    "arxiv_id": "2.2",
                    "제목": "B",
                    "논문_페이지_URL": "https://arxiv.org/abs/2.2v1",
                },
            ]
            fails = crawler.enrich_rows_with_detail_titles(client, rows, logger=logger)
            logger.close()
            self.assertEqual(fails, 1)
            self.assertTrue(rows[1]["상세페이지_제목"])
            self.assertIn("실패(http_404)", rows[0]["상세페이지_수집상태"])


if __name__ == "__main__":
    unittest.main()
