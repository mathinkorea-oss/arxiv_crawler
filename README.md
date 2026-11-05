# arXiv Crawler & 요약

arXiv 논문 수집·Gemini 초록 요약 Streamlit 대시보드.

## 로컬 실행

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
# .env.local 에 GEMINI_API_KEY 설정
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. [share.streamlit.io](https://share.streamlit.io/) → GitHub 연결
2. **Create app** → Repository: `mathinkorea-oss/arxiv_crawler`, Branch: `main`, Main file: `app.py`
3. **Advanced settings → Secrets** (`.streamlit/secrets.toml.example` 참고):

```toml
GEMINI_API_KEY = "..."
```

4. Deploy

배포 URL 예: `https://mathinkorea-oss-arxiv-crawler-app-*.streamlit.app`

> Cloud에서는 디스크가 재시작 시 초기화될 수 있습니다. `results/` CSV는 세션 간 유지가 보장되지 않습니다.
