"""Streamlit UI 단위·AppTest (실 API/arXiv 없음)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ui_support as ui

APP_FILE = Path(__file__).resolve().parent / "app.py"


def _noop_lock(name: str, *, timeout: float = 0):
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        yield

    return _cm()


class UiSupportTests(unittest.TestCase):
    def test_sanitize_strips_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "SECRET123"}, clear=False):
            out = ui.sanitize_user_error("failed SECRET123")
        self.assertNotIn("SECRET123", out)
        self.assertIn("[REDACTED]", out)

    def test_temporary_gemini_model_restores_env(self):
        with patch.dict(os.environ, {}, clear=True):
            with ui.temporary_gemini_model("model-a") as (sel, resolved):
                self.assertEqual(os.environ.get("GEMINI_MODEL"), "model-a")
                self.assertEqual(sel, "model-a")
            self.assertIsNone(os.environ.get("GEMINI_MODEL"))

    def test_cross_process_lock_serializes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_root = Path(tmp)
            with patch.object(ui, "lock_dir", return_value=lock_root):
                with ui.cross_process_job_lock("test_job"):
                    lock_file = lock_root / "test_job.lock"
                    self.assertTrue(lock_file.exists())


class AppLoadTests(unittest.TestCase):
    def test_app_test_runs_without_exception(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_FILE))
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, msg="st.exception 표시 없어야 함")

    def test_app_has_sidebar_nav_and_title(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_FILE))
        at.run(timeout=15)
        md = " ".join(getattr(m, "value", "") or "" for m in at.markdown)
        self.assertIn("arXiv 논문 수집·요약", md)
        self.assertGreaterEqual(len(at.radio), 1)


class AppButtonMockTests(unittest.TestCase):
    def test_single_button_uses_mock_not_real_api(self):
        import os

        from streamlit.testing.v1 import AppTest

        fake = {
            "summary_one_line": "짧은 한 줄",
            "summary_easy": "a. b. c.",
            "real_world": "x. y.",
            "limitations": "초록에 연구의 한계가 명시되어 있지 않습니다.",
        }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch("ui_support.cross_process_job_lock", _noop_lock):
                with patch(
                    "summarize_paper.summarize_paper",
                    return_value=fake,
                ) as mock_sum:
                    at = AppTest.from_file(str(APP_FILE))
                    at.run(timeout=15)
                    at.radio(key="ui_nav").set_value("1편 요약").run()
                    at.radio(key="single_source").set_value("직접 입력").run()
                    at.text_input(key="single_title").input("T").run()
                    at.text_area(key="single_abstract").input("A" * 50).run()
                    at.button(key="btn_single").click().run()
        self.assertEqual(len(at.exception), 0)
        mock_sum.assert_called_once()
        self.assertTrue(
            any(
                "CSV에는 저장되지 않았습니다" in (getattr(s, "value", "") or "")
                for s in at.success
            )
        )


if __name__ == "__main__":
    unittest.main()
