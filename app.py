"""
arXiv 크롤러·Gemini 요약 통합 UI (Streamlit).

실행:
  streamlit run app.py
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

import batch_summaries as bs
import crawler
import run_summary as rs
import summarize_paper as sp
import ui_dashboard as dash
import ui_support as ui

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parent


def _init_session_state() -> None:
    defaults = {
        "ui_job_active": False,
        "ui_nav": "개요",
        "last_crawl": None,
        "last_single": None,
        "last_batch": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _ensure_cwd() -> None:
    import os

    os.chdir(PROJECT_ROOT)


def _api_key_configured() -> bool:
    sp._ensure_env_loaded()
    import os

    return bool((os.environ.get("GEMINI_API_KEY") or "").strip())


def _load_papers_df() -> pd.DataFrame | None:
    path = sp.papers_csv_path()
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, dtype={"arxiv_id": str}, encoding="utf-8-sig")
    except Exception:
        return None


def _load_summaries_df() -> pd.DataFrame | None:
    path = sp.paper_summaries_csv_path()
    if not path.is_file():
        return None
    try:
        return pd.read_csv(
            path,
            dtype={"arxiv_id": str, "arxiv_버전": str, "input_hash": str},
            encoding="utf-8-sig",
        )
    except Exception:
        return None


def _run_crawler_cli(query: str, count: int, skip_detail: bool) -> tuple[int, str, str]:
    argv = ["-q", query, "-n", str(count)]
    if skip_detail:
        argv.append("--skip-detail-titles")
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = crawler.main(argv)
    return code, out.getvalue(), err.getvalue()


def _render_job_banner() -> None:
    if st.session_state.ui_job_active:
        st.info("작업 실행 중… 완료될 때까지 다른 실행 버튼은 비활성화됩니다.")


def _stat_card(label: str, value: str | int) -> None:
    st.markdown(
        f'<div class="dash-stat-card"><h3>{dash.escape_html(label)}</h3>'
        f"<p>{dash.escape_html(str(value))}</p></div>",
        unsafe_allow_html=True,
    )


def page_overview() -> None:
    papers = _load_papers_df()
    summaries = _load_summaries_df()
    stats = dash.compute_overview_stats(papers, summaries)

    c1, c2, c3 = st.columns(3)
    with c1:
        _stat_card("📚 크롤링된 논문", stats.crawled)
    with c2:
        _stat_card("✅ 요약 완료", stats.summarized if stats.ok else "—")
    with c3:
        _stat_card("⏳ 요약 대기", stats.pending if stats.ok else "—")

    for note in stats.notes:
        st.warning(note)

    st.markdown("### 🆕 최근 논문")
    recent = dash.recent_papers_table(papers, summaries, limit=5)
    if recent is None:
        if papers is None or papers.empty:
            st.info("아직 수집된 논문이 없습니다. 「수집」에서 검색해 보세요.")
        else:
            st.warning(
                "최근 논문 목록을 만들 수 없습니다. "
                "papers.csv에 제목·등록일·arxiv_id 열이 있는지 확인해 주세요."
            )
    else:
        st.dataframe(
            recent,
            width="stretch",
            hide_index=True,
            column_config={
                "원문": st.column_config.LinkColumn("원문", display_text="arXiv"),
            },
        )


def page_crawl() -> None:
    st.subheader("arXiv 논문 수집")
    st.caption("공식 API · 15초 간격 · results/papers.csv 누적 (실행 버튼을 눌렀을 때만 동작)")

    query = st.text_input("검색어", value="mathematics education", key="crawl_query")
    count = st.number_input(
        "수집 개수", min_value=1, max_value=100, value=10, step=1, key="crawl_count"
    )
    skip_detail = st.checkbox(
        "상세페이지 제목 수집 건너뛰기 (API만)",
        value=False,
        key="crawl_skip_detail",
    )

    busy = st.session_state.ui_job_active
    if st.button(
        "크롤링 시작",
        type="primary",
        key="btn_crawl",
        disabled=busy,
    ):
        if not query.strip():
            st.session_state.last_crawl = {
                "status": "failed",
                "message": "검색어를 입력하세요.",
            }
        else:
            st.session_state.ui_job_active = True
            requested = int(count)
            try:
                with ui.cross_process_job_lock("crawl"):
                    progress = st.progress(0, text=f"논문을 수집하고 있습니다 (0/{requested}편)")
                    code, stdout, stderr = _run_crawler_cli(
                        query.strip(), requested, skip_detail
                    )
                    progress.progress(100, text="수집이 완료되었습니다.")
                    parsed = dash.parse_crawl_stdout(stdout, stderr)
                    parsed.requested_count = requested
                    if code == 0:
                        st.session_state.last_crawl = {
                            "status": "success",
                            "message": "크롤링이 완료되었습니다.",
                            "parsed": parsed,
                            "log": (stdout + stderr)[-8000:],
                        }
                    else:
                        st.session_state.last_crawl = {
                            "status": "failed",
                            "message": f"크롤링에 실패했습니다 (종료 코드 {code}).",
                            "parsed": parsed,
                            "log": (stdout + stderr)[-8000:],
                            "stderr": stderr,
                        }
            except ui.JobBusyError as exc:
                st.session_state.last_crawl = {
                    "status": "failed",
                    "message": str(exc),
                }
            except Exception as exc:
                st.session_state.last_crawl = {
                    "status": "failed",
                    "message": ui.sanitize_user_error(str(exc)),
                }
            finally:
                st.session_state.ui_job_active = False

    last = st.session_state.last_crawl
    if last:
        if last["status"] == "success":
            st.success(last["message"])
        else:
            st.error(last.get("message", "크롤링 실패"))
            if last.get("stderr"):
                st.caption(ui.sanitize_user_error(str(last["stderr"])[:500]))

        parsed: dash.CrawlRunDisplay | None = last.get("parsed")
        if parsed:
            q = parsed.queried if parsed.queried is not None else "—"
            dup = parsed.dup_excluded if parsed.dup_excluded is not None else "—"
            new = parsed.new_saved if parsed.new_saved is not None else "—"
            fail = parsed.detail_failures if parsed.detail_failures is not None else "—"
            st.markdown(
                f"**조회** {q}편 · **중복 제외** {dup}편 · "
                f"**신규 저장** {new}편 · **실패** {fail}편"
            )
            table_df = dash.crawl_results_dataframe(parsed)
            if table_df is not None and not table_df.empty:
                st.dataframe(
                    table_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "원문": st.column_config.LinkColumn(
                            "원문", display_text="열기"
                        ),
                    },
                )
            elif last["status"] == "success":
                st.caption("이번 실행에서 표시할 논문 표가 없습니다.")

        if last.get("log"):
            with st.expander("실행 상세"):
                if parsed and parsed.meta_lines:
                    st.markdown("**API·검색 설정**")
                    st.code("\n".join(parsed.meta_lines))
                st.text(last["log"])


def page_single_summary() -> None:
    st.subheader("논문 1편 요약")
    st.caption("제목·초록을 바탕으로 한 요약 · 일괄 저장은 「일괄 요약」 탭을 사용하세요.")

    papers = _load_papers_df()
    source = st.radio(
        "입력",
        ["papers.csv에서 선택", "직접 입력"],
        horizontal=True,
        key="single_source",
    )

    title = ""
    abstract = ""
    arxiv_id = ""

    if source == "papers.csv에서 선택":
        if papers is None or papers.empty:
            st.warning("papers.csv가 없거나 비어 있습니다. 먼저 「수집」에서 크롤링하세요.")
        else:
            valid = papers[
                papers["제목"].astype(str).str.strip().ne("")
                & papers["초록"].astype(str).str.strip().ne("")
            ]
            if valid.empty:
                st.warning("제목·초록이 모두 있는 행이 없습니다.")
            else:
                labels = [
                    f"{r.get('arxiv_id', '')} · {str(r.get('제목', ''))[:72]}"
                    + ("…" if len(str(r.get("제목", ""))) > 72 else "")
                    for _, r in valid.iterrows()
                ]
                idx = st.selectbox(
                    "논문 선택",
                    range(len(valid)),
                    format_func=lambda i: labels[i],
                    key="single_paper_idx",
                )
                row = valid.iloc[idx]
                arxiv_id = str(row.get("arxiv_id", ""))
                title = str(row["제목"])
                abstract = str(row["초록"])
                st.markdown(
                    f'<p class="dash-paper-title">{dash.escape_html(title)}</p>',
                    unsafe_allow_html=True,
                )
                if arxiv_id:
                    st.markdown(
                        f'<p class="dash-meta-small">arXiv ID: {dash.escape_html(arxiv_id)}</p>',
                        unsafe_allow_html=True,
                    )
                with st.expander("초록 미리보기"):
                    st.markdown(dash.prose_html(abstract[:2000]), unsafe_allow_html=True)
    else:
        title = st.text_input("제목", key="single_title")
        abstract = st.text_area("초록", height=200, key="single_abstract")

    busy = st.session_state.ui_job_active
    if st.button("요약 생성", type="primary", key="btn_single", disabled=busy):
        if not _api_key_configured():
            st.session_state.last_single = {
                "status": "failed",
                "message": ui.friendly_summarize_error(sp.SummarizeConfigError()),
            }
        elif not str(title).strip() or not str(abstract).strip():
            st.session_state.last_single = {
                "status": "failed",
                "message": "제목과 초록을 입력하세요.",
            }
        else:
            st.session_state.ui_job_active = True
            try:
                with ui.cross_process_job_lock("gemini_single"):
                    with ui.temporary_gemini_model(None) as (
                        selected_label,
                        resolved_model,
                    ):
                        counter: list[int] = [0]
                        with st.spinner("Gemini 요약 중…"):
                            result = sp.summarize_paper(
                                str(title).strip(),
                                str(abstract).strip(),
                                api_request_counter=counter,
                            )
                        st.session_state.last_single = {
                            "status": "success",
                            "arxiv_id": arxiv_id,
                            "paper_title": str(title).strip(),
                            "selected_model": selected_label,
                            "resolved_model": resolved_model,
                            "api_requests": counter[0],
                            "result": result,
                            "saved": False,
                        }
            except ui.JobBusyError as exc:
                st.session_state.last_single = {
                    "status": "failed",
                    "message": str(exc),
                }
            except sp.SummarizeError as exc:
                st.session_state.last_single = {
                    "status": "failed",
                    "message": ui.friendly_summarize_error(exc),
                }
            except Exception as exc:
                st.session_state.last_single = {
                    "status": "failed",
                    "message": ui.sanitize_user_error(str(exc)),
                }
            finally:
                st.session_state.ui_job_active = False

    last = st.session_state.last_single
    if last:
        if last["status"] == "success":
            st.success(
                "화면에서만 확인한 결과입니다. CSV에는 저장되지 않았습니다."
            )
            show_title = last.get("paper_title") or ""
            if show_title:
                st.markdown(
                    f'<p class="dash-paper-title">{dash.escape_html(show_title)}</p>',
                    unsafe_allow_html=True,
                )
            if last.get("arxiv_id"):
                st.markdown(
                    f'<p class="dash-meta-small">arXiv ID: {dash.escape_html(last["arxiv_id"])}</p>',
                    unsafe_allow_html=True,
                )
            r = last["result"]
            st.markdown(
                '<p class="dash-meta-small">제목·초록을 바탕으로 한 요약</p>',
                unsafe_allow_html=True,
            )
            st.markdown('<p class="dash-section-head">💡 한 줄 요약</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="dash-one-line">{dash.escape_html(r["summary_one_line"])}</div>',
                unsafe_allow_html=True,
            )
            sections = [
                ("📖 쉬운 설명", r["summary_easy"]),
                ("🌍 현실 활용", r["real_world"]),
                ("🔎 한계", r["limitations"]),
            ]
            for head, body in sections:
                st.markdown(f'<p class="dash-section-head">{head}</p>', unsafe_allow_html=True)
                st.markdown(dash.prose_html(body), unsafe_allow_html=True)
            with st.expander("실행 상세"):
                st.markdown(
                    f"- 선택 모델: `{dash.escape_html(last.get('selected_model', ''))}`"
                )
                st.markdown(
                    f"- 실제 API 모델: `{dash.escape_html(last.get('resolved_model', ''))}`"
                )
                st.markdown(f"- generate 호출: {last.get('api_requests', 0)}회")
        else:
            st.error(last.get("message", "요약 실패"))


def page_batch_summary() -> None:
    st.subheader("일괄 요약")
    st.caption(
        "paper_summaries.csv에 누적 저장 · 복구 JSON 자동 반영 · 중복 방지 "
        "(실행 버튼을 눌렀을 때만 API/저장)"
    )

    limit = st.slider(
        "논문 시도 상한 (--limit)",
        1,
        10,
        bs.DEFAULT_BATCH_LIMIT,
        key="batch_limit",
    )
    resummarize = st.checkbox(
        "현재 모델·프롬프트로 다시 요약 (기존 요약이 있어도 새 설정으로 추가 저장)",
        value=False,
        key="batch_resummarize",
        help=(
            "끄면: 같은 논문·초록(input_hash)에 대한 저장 요약이 있으면 API를 호출하지 않습니다. "
            "켜면: 현재 GEMINI_MODEL과 프롬프트 버전 조합이 CSV에 없을 때만 다시 요약합니다."
        ),
    )

    busy = st.session_state.ui_job_active
    if st.button("일괄 요약 실행", type="primary", key="btn_batch", disabled=busy):
        if not _api_key_configured():
            st.session_state.last_batch = {
                "status": "failed",
                "message": ui.friendly_summarize_error(sp.SummarizeConfigError()),
            }
        else:
            st.session_state.ui_job_active = True
            try:
                with ui.cross_process_job_lock("gemini_batch"):
                    with ui.temporary_gemini_model(None) as (
                        selected_label,
                        resolved_model,
                    ):
                        with st.spinner("일괄 요약 실행 중…"):
                            stats = bs.run_batch_summaries(
                                limit=int(limit),
                                resummarize_with_current_settings=resummarize,
                            )
                        st.session_state.last_batch = {
                            "status": "success",
                            "selected_model": selected_label,
                            "resolved_model": resolved_model,
                            "stats": stats,
                            "report": bs.format_stats_report(
                                stats, attempt_limit=int(limit)
                            ),
                        }
            except ui.JobBusyError as exc:
                st.session_state.last_batch = {
                    "status": "failed",
                    "message": str(exc),
                }
            except bs.BatchAbortError as exc:
                st.session_state.last_batch = {
                    "status": "failed",
                    "message": _friendly_batch_abort(exc),
                }
            except rs.RunSummaryError as exc:
                st.session_state.last_batch = {
                    "status": "failed",
                    "message": ui.sanitize_user_error(str(exc)),
                }
            except Exception as exc:
                st.session_state.last_batch = {
                    "status": "failed",
                    "message": ui.sanitize_user_error(str(exc)),
                }
            finally:
                st.session_state.ui_job_active = False

    last = st.session_state.last_batch
    if last:
        if last["status"] == "success":
            stats = last["stats"]
            if stats.failed:
                st.warning("일부 논문 요약에 실패했습니다. 아래 통계를 확인하세요.")
            else:
                st.success("일괄 요약이 완료되었습니다.")
            st.code(last.get("report", ""))
            with st.expander("실행 상세"):
                st.markdown(
                    f"선택 모델: `{last.get('selected_model', '')}` · "
                    f"실제 API 모델: `{last.get('resolved_model', '')}`"
                )
            if stats.failed:
                st.warning(
                    "실패 내역은 logs/summary_failures.log 에 기록됩니다. "
                    "파일을 열어 원인을 확인하세요."
                )
        else:
            st.error(last.get("message", "일괄 요약 실패"))


def _friendly_batch_abort(exc: bs.BatchAbortError) -> str:
    msg = ui.sanitize_user_error(str(exc))
    if "복구" in msg or "pending" in msg.lower() or "json" in msg.lower():
        return (
            "저장 복구 데이터를 처리하지 못해 일괄 요약을 중단했습니다. "
            "results/_pending_summaries 폴더의 JSON 파일이 손상되지 않았는지 확인한 뒤 "
            "다시 실행해 주세요."
        )
    return msg


def page_browse() -> None:
    st.subheader("데이터 보기")

    tab_p, tab_s = st.tabs(["papers.csv", "paper_summaries.csv"])

    with tab_p:
        papers = _load_papers_df()
        if papers is None:
            st.info("papers.csv 없음 — 「수집」에서 논문을 수집하세요.")
        else:
            st.dataframe(papers, width="stretch", height=400)

    with tab_s:
        summaries = _load_summaries_df()
        if summaries is None:
            st.info("paper_summaries.csv 없음 — 「일괄 요약」을 실행하세요.")
        else:
            st.dataframe(summaries, width="stretch", height=400)


def _render_sidebar_nav() -> str:
    st.sidebar.markdown("**메뉴**")
    choice = st.sidebar.radio(
        "메뉴",
        dash.MENU_ITEMS,
        key="ui_nav",
        label_visibility="collapsed",
    )
    return choice


def main() -> None:
    _ensure_cwd()
    _init_session_state()
    st.set_page_config(
        page_title="arXiv 논문 수집·요약",
        page_icon="📚",
        layout="wide",
    )
    st.markdown(dash.inject_dashboard_css(), unsafe_allow_html=True)

    page = _render_sidebar_nav()

    st.markdown(
        '<h1 class="dash-title">arXiv 논문 수집·요약</h1>'
        '<p class="dash-subtitle">관심 논문을 모으고, 초록을 쉬운 한국어로 확인하세요.</p>',
        unsafe_allow_html=True,
    )

    _render_job_banner()

    if page == "개요":
        page_overview()
    elif page == "수집":
        page_crawl()
    elif page == "1편 요약":
        page_single_summary()
    elif page == "일괄 요약":
        page_batch_summary()
    elif page == "데이터":
        page_browse()


if __name__ == "__main__":
    main()
