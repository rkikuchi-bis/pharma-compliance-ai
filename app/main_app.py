import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.rag_engine import RAGEngine


DEFAULT_TOP_K = 8


def t(lang: str, ja: str, en: str) -> str:
    return ja if lang == "ja" else en


def render_single_response(response: dict, lang: str) -> None:
    status = response.get("status", "")
    risk_level = response.get("risk_level", "")
    citations = response.get("citations", [])

    st.subheader(t(lang, "回答", "Answer"))
    st.write(response.get("answer", ""))

    st.subheader(t(lang, "根拠ステータス", "Evidence Status"))
    if status == "NO_CLEAR_EVIDENCE":
        st.warning(status)
    elif status in {"PROHIBITED", "OUT_OF_SCOPE", "GENERATION_ERROR"}:
        st.error(status)
    elif status == "ALLOWED":
        st.success(status)
    else:
        st.info(status)

    st.subheader(t(lang, "リスクレベル", "Risk Level"))
    if risk_level == "HIGH":
        st.error(risk_level)
    elif risk_level == "MEDIUM":
        st.warning(risk_level)
    elif risk_level == "LOW":
        st.success(risk_level)
    else:
        st.info(risk_level)

    st.subheader(t(lang, "推奨アクション", "Recommended Action"))
    st.write(response.get("recommended_action", ""))

    st.subheader(t(lang, "判断理由", "Reasoning"))
    st.write(response.get("reasoning", ""))

    st.subheader(t(lang, "主要根拠", "Key Evidence"))
    st.write(response.get("key_evidence", ""))

    st.subheader(t(lang, "検索範囲", "Search Scope"))
    st.write(response.get("search_scope", ""))

    if status == "NO_CLEAR_EVIDENCE":
        st.error(
            t(
                lang,
                "明示的な根拠が見つからないことは、許容を意味しません。追加確認が必要です。",
                "Absence of clear evidence does not imply permission. Additional verification is required.",
            )
        )

    st.subheader(t(lang, "引用根拠", "Citations"))
    if citations:
        for citation in citations:
            st.write(f"- {citation}")
    else:
        st.write(t(lang, "引用根拠はありません。", "No citations available."))


def build_comparison_summary(us_response: dict | None, uk_response: dict | None) -> str:
    if not us_response and not uk_response:
        return "No jurisdiction-specific results were available."

    if us_response and not uk_response:
        return f"Only US results were available: {us_response.get('status', 'UNKNOWN')} / Risk={us_response.get('risk_level', 'UNKNOWN')}."

    if uk_response and not us_response:
        return f"Only UK results were available: {uk_response.get('status', 'UNKNOWN')} / Risk={uk_response.get('risk_level', 'UNKNOWN')}."

    us_status = us_response.get("status", "UNKNOWN")
    uk_status = uk_response.get("status", "UNKNOWN")
    us_risk = us_response.get("risk_level", "UNKNOWN")
    uk_risk = uk_response.get("risk_level", "UNKNOWN")

    if us_status == uk_status and us_risk == uk_risk:
        return f"US and UK are aligned: status={us_status}, risk={us_risk}."

    return f"US and UK differ: US={us_status} ({us_risk}), UK={uk_status} ({uk_risk})."


st.set_page_config(
    page_title="Pharma Compliance AI",
    page_icon="💊",
    layout="wide",
)

st.title("Pharma Compliance AI")

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"

lang = st.session_state.ui_language

st.caption(
    t(
        lang,
        "製薬規制を対象としたRAGベースのコンプライアンス支援アシスタント",
        "RAG-based compliance assistant for pharmaceutical regulations",
    )
)


@st.cache_resource
def get_engine() -> RAGEngine:
    engine = RAGEngine()
    engine.initialize_data("./data")
    return engine


engine = get_engine()

col_a, col_b = st.columns(2)

with col_a:
    ui_language = st.selectbox(
        "Language",
        options=["English", "日本語"],
        index=0 if st.session_state.ui_language == "en" else 1,
    )
    lang = "ja" if ui_language == "日本語" else "en"
    st.session_state.ui_language = lang

with col_b:
    region_label = st.selectbox(
        "Region",
        options=["Japan", "US", "UK", "US + UK"],
        index=0 if lang == "ja" else 1,
    )

st.caption("・Questions entered in English can be evaluated against the selected region(s).")
st.caption("・日本語で入力した質問は、選択した地域の規制を基準に回答します。")

query = st.text_area(
    t(lang, "質問", "Question"),
    height=120,
    placeholder=t(
        lang,
        "例）講演を依頼した医師に対して謝礼を支払うことは可能ですか？",
        "Example: Is it permissible to pay a physician speaker an honorarium?",
    ),
)

run_button = st.button(
    t(lang, "実行", "Ask"),
    type="primary",
)

response = None

if run_button:
    with st.spinner(
        t(
            lang,
            "検索・回答生成中...",
            "Searching and generating answer...",
        )
    ):
        response = engine.answer_question(
            query=query,
            lang=lang,
            region=region_label,
            top_k=DEFAULT_TOP_K,
        )

if response:
    if region_label == "US + UK":
        us_response = response.get("US")
        uk_response = response.get("UK")

        st.subheader(t(lang, "比較", "Comparison"))
        st.info(build_comparison_summary(us_response, uk_response))

        if us_response:
            st.markdown("## US (PhRMA Code / OIG)")
            render_single_response(us_response, lang)

        st.markdown("---")

        if uk_response:
            st.markdown("## UK (ABPI Code)")
            render_single_response(uk_response, lang)
    else:
        render_single_response(response, lang)

st.markdown("---")

if lang == "ja":
    st.markdown(
        """
**免責事項**  
本ツールは参考情報提供を目的としています。  
法的助言ではありません。最終判断は必ず公式文書および社内基準で確認してください。
"""
    )
else:
    st.markdown(
        """
**Disclaimer**  
This tool is for informational purposes only and does not constitute legal advice.
Final decisions should be verified against official documents and internal policies.
"""
    )