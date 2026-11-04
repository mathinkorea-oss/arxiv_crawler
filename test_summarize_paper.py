"""summarize_paper 단위·통합 테스트."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import summarize_paper as sp


def _mock_response(payload: dict | str, *, empty: bool = False) -> MagicMock:
    resp = MagicMock()
    if empty:
        resp.text = ""
    elif isinstance(payload, dict):
        resp.text = json.dumps(payload, ensure_ascii=False)
    else:
        resp.text = payload
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    resp.usage_metadata = usage
    return resp


VALID_PAYLOAD = {
    "summary_one_line": "AI 동료와 수학 공부",
    "summary_easy": "전문가들이 AI와 수학 문제를 함께 풀었습니다. "
    "사람들은 AI가 친구처럼 느껴지려면 부드럽게 대화해야 한다고 말했습니다. "
    "무조건 동의하거나 가르치기만 하면 친구가 아니라 선생님 같다고 느꼈습니다.",
    "real_world": "초록의 설계 방향은 향후 동료형 AI를 만드는 데 참고할 수 있는 가능성이 있습니다. "
    "실제 수업에서 학습 성과가 오르는지는 이 초록만으로는 알 수 없습니다.",
    "limitations": "초록에 연구의 한계가 명시되어 있지 않습니다.",
}


class PromptGroundingTests(unittest.TestCase):
    def test_system_instruction_includes_guardrails(self):
        instruction = sp._build_system_instruction()
        self.assertIn("더 중요하다", instruction)
        self.assertIn("더 효과적이다", instruction)
        self.assertIn("의도적으로 실수", instruction)
        self.assertIn("초록에 연구의 한계가 명시되어 있지 않습니다", instruction)
        self.assertIn("전문가 인터뷰", instruction)
        self.assertIn("학생 실험이 없다고 단정", instruction)
        self.assertIn("평가받을 걱정 없이", instruction)
        self.assertIn("영향의 크기", instruction)
        self.assertIn("40자", instruction)

    def test_prompt_guardrail_markers_complete(self):
        markers = sp.prompt_grounding_guardrails()
        self.assertGreaterEqual(len(markers), 4)

    def test_user_prompt_reminds_abstract_only(self):
        prompt = sp._build_user_prompt("T", "A")
        self.assertIn("초록에 없는 내용", prompt)


class ProjectPathTests(unittest.TestCase):
    def test_project_root_has_crawler_and_results(self):
        root = sp.project_root()
        self.assertTrue((root / "crawler.py").is_file())
        self.assertTrue((root / "results").is_dir())

    def test_papers_csv_path(self):
        path = sp.papers_csv_path()
        self.assertEqual(path.name, "papers.csv")
        self.assertTrue(path.is_file())

    def test_env_local_path_found_without_printing_secrets(self):
        env_path = sp._find_env_local_path()
        self.assertIsNotNone(env_path)
        self.assertTrue(env_path.is_file())
        self.assertEqual(env_path.name, ".env.local")


class ValidationHelperTests(unittest.TestCase):
    def test_validate_ok(self):
        errors = sp._validate_summary_payload(VALID_PAYLOAD)
        self.assertEqual(errors, [])

    def test_validate_missing_key(self):
        bad = dict(VALID_PAYLOAD)
        del bad["limitations"]
        errors = sp._validate_summary_payload(bad)
        self.assertTrue(any("누락" in e for e in errors))

    def test_validate_one_line_longer_than_legacy_20_ok(self):
        payload = dict(VALID_PAYLOAD)
        payload["summary_one_line"] = (
            "수학 수업에서 AI를 친구처럼 쓰려면 부드럽게 대화하는 설계가 필요하다는 연구입니다."
        )
        errors = sp._validate_summary_payload(payload)
        self.assertEqual(errors, [])

    def test_validate_one_line_empty_string_fails(self):
        bad = dict(VALID_PAYLOAD)
        bad["summary_one_line"] = "   "
        errors = sp._validate_summary_payload(bad)
        self.assertTrue(any("summary_one_line" in e and "비어" in e for e in errors))

    def test_validate_sentence_counts(self):
        bad = dict(VALID_PAYLOAD)
        bad["summary_easy"] = "한 문장만."
        errors = sp._validate_summary_payload(bad)
        self.assertTrue(any("summary_easy" in e for e in errors))


class SummarizePaperInputTests(unittest.TestCase):
    def test_empty_title(self):
        with self.assertRaises(sp.SummarizeInputError):
            sp.summarize_paper("", "초록")

    def test_empty_abstract(self):
        with self.assertRaises(sp.SummarizeInputError):
            sp.summarize_paper("제목", "   ")


class SummarizePaperMockFlowTests(unittest.TestCase):
    def setUp(self):
        sp._reset_client_for_tests()

    def test_successful_mock_response(self):
        calls = {"n": 0}

        def fake_generate(*, correction_hint=None):
            calls["n"] += 1
            return _mock_response(VALID_PAYLOAD)

        with patch.object(sp, "_get_client", return_value=MagicMock()):
            result = sp._run_with_attempts("T", "A", generate_fn=fake_generate)
        self.assertEqual(set(result.keys()), set(sp.SUMMARY_KEYS))
        self.assertEqual(calls["n"], 1)

    def test_one_line_over_20_chars_no_retry(self):
        payload = dict(VALID_PAYLOAD)
        payload["summary_one_line"] = "가" * 25
        calls = {"n": 0}

        def fake_generate(*, correction_hint=None):
            calls["n"] += 1
            self.assertIsNone(correction_hint)
            return _mock_response(payload)

        with patch.object(sp, "_get_client", return_value=MagicMock()):
            result = sp._run_with_attempts("T", "A", generate_fn=fake_generate)
        self.assertEqual(len(result["summary_one_line"]), 25)
        self.assertEqual(calls["n"], 1)

    def test_invalid_json_then_success(self):
        seq = [
            _mock_response("{not json"),
            _mock_response(VALID_PAYLOAD),
        ]

        def fake_generate(*, correction_hint=None):
            return seq.pop(0)

        with patch.object(sp, "_get_client", return_value=MagicMock()):
            result = sp._run_with_attempts("T", "A", generate_fn=fake_generate)
        self.assertIn("summary_one_line", result)

    def test_validation_fail_twice_raises(self):
        bad_easy = dict(VALID_PAYLOAD)
        bad_easy["summary_easy"] = "한 문장만."

        def fake_generate(*, correction_hint=None):
            return _mock_response(bad_easy)

        with patch.object(sp, "_get_client", return_value=MagicMock()):
            with self.assertRaises(sp.SummarizeValidationError):
                sp._run_with_attempts("T", "A", generate_fn=fake_generate)

    def test_api_error_auth(self):
        err = sp.genai_errors.ClientError(401, {}, MagicMock())
        err.response.headers = {}

        def fake_generate(*, correction_hint=None):
            raise err

        with patch.object(sp, "_get_client", return_value=MagicMock()):
            with self.assertRaises(sp.SummarizeAuthError):
                sp._run_with_attempts("T", "A", generate_fn=fake_generate)

    def test_empty_response_blocked(self):
        def fake_generate(*, correction_hint=None):
            return _mock_response({}, empty=True)

        with patch.object(sp, "_get_client", return_value=MagicMock()):
            with self.assertRaises(sp.SummarizeBlockedError):
                sp._run_with_attempts("T", "A", generate_fn=fake_generate)


class LiveGeminiFromCsvTest(unittest.TestCase):
    """results/papers.csv 첫 유효 행 1건으로 실제 API 호출 (기본 skip)."""

    @unittest.skipUnless(
        os.environ.get("RUN_GEMINI_LIVE") == "1",
        "실 API 테스트는 RUN_GEMINI_LIVE=1 일 때만 실행",
    )

    @classmethod
    def setUpClass(cls):
        sp._reset_client_for_tests()
        cls.papers_path = Path(__file__).resolve().parent / "results" / "papers.csv"
        if not cls.papers_path.is_file():
            raise unittest.SkipTest("papers.csv 없음")
        try:
            sp._get_api_key()
        except sp.SummarizeConfigError:
            raise unittest.SkipTest("GEMINI_API_KEY 미설정")

    def test_first_paper_live_summary(self):
        df = pd.read_csv(
            self.papers_path,
            dtype={"arxiv_id": str},
            encoding="utf-8-sig",
        )
        row = None
        for _, r in df.iterrows():
            if str(r.get("제목", "")).strip() and str(r.get("초록", "")).strip():
                row = r
                break
        self.assertIsNotNone(row, "제목·초록이 있는 행이 없습니다")

        result = sp.summarize_paper(str(row["제목"]), str(row["초록"]))
        for key in sp.SUMMARY_KEYS:
            self.assertIsInstance(result[key], str)
            self.assertTrue(result[key].strip())

        print("\n--- Live Gemini summary ---")
        print("arxiv_id:", row["arxiv_id"])
        print("제목:", row["제목"])
        for key in sp.SUMMARY_KEYS:
            print(f"{key}: {result[key]}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    unittest.main()
