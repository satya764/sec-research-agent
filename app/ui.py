"""Streamlit front end. Run the API first, then: uv run streamlit run app/ui.py"""

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}

st.set_page_config(page_title="SEC Filings Research Agent", page_icon="📄")
st.title("SEC Filings Research Agent")
st.caption("Answers from 10-K and 10-Q filings. Every number comes from SEC XBRL data, with sources.")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:16]
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])

question = st.chat_input("e.g. How much did Apple's revenue change from fiscal 2023 to 2024?")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"), st.spinner("Reading filings..."):
        try:
            resp = httpx.post(
                f"{API_URL}/ask",
                json={"question": question, "session_id": st.session_state.session_id},
                headers=HEADERS,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
        except httpx.HTTPError as err:
            st.error(f"API error: {err}")
            st.stop()
        st.write(result["answer"])
        if result["citations"]:
            with st.expander(f"Sources ({len(result['citations'])})"):
                for c in result["citations"]:
                    where = c["section"] or "XBRL financial data"
                    st.markdown(f"**[{c['source_id']}]** {c['company']} {c['form']}, filed {c['filing_date']}, {where} — [open]({c['url']})")
                    st.caption(c["snippet"][:400])
        st.caption(
            f"{result['model']} · {result['steps']} steps · ${result['cost_usd']:.4f} · "
            f"{result['latency_ms'] / 1000:.1f} s"
        )
    st.session_state.history.append({"question": question, "answer": result["answer"]})
