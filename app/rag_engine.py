import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

from app.audit_logger import write_audit_log
from app.hybrid_search import HybridSearcher


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class EvidenceStatus(str, Enum):
    ALLOWED = "ALLOWED"
    PROHIBITED = "PROHIBITED"
    CONDITIONAL = "CONDITIONAL"
    NO_CLEAR_EVIDENCE = "NO_CLEAR_EVIDENCE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    GENERATION_ERROR = "GENERATION_ERROR"


SYSTEM_PROMPT = """
You are a pharmaceutical compliance assistant.

Language rule:
- If the UI language is Japanese, respond in Japanese.
- If the UI language is English, respond in English.

Rules:
1. Base your answer only on retrieved evidence.
2. Do not infer permission from the absence of prohibition.
3. Do not infer permission from general spending limits alone.
4. If the user asks about a specific recipient type, role, or scenario, and the retrieved evidence does not explicitly cover that recipient type, role, or scenario, return NO_CLEAR_EVIDENCE.
5. Distinguish clearly between:
   - ALLOWED
   - PROHIBITED
   - CONDITIONAL
   - NO_CLEAR_EVIDENCE
   - OUT_OF_SCOPE
   - CONFLICTING_EVIDENCE
6. Always provide concise reasoning tied to the evidence.
7. Return valid JSON only, with no markdown fences.

Output schema:
{
  "status": "ALLOWED | PROHIBITED | CONDITIONAL | NO_CLEAR_EVIDENCE | OUT_OF_SCOPE | CONFLICTING_EVIDENCE",
  "answer": "string",
  "reasoning": "string",
  "key_evidence": "string or list of strings",
  "citations": ["string"]
}
"""


def assess_risk_level(status: str, query: str) -> str:
    q = (query or "").lower()

    high_risk_terms = [
        "first class",
        "first-class",
        "business class",
        "alcohol",
        "luxury",
        "ファーストクラス",
        "ビジネスクラス",
        "アルコール",
        "高級",
    ]

    if status == "PROHIBITED":
        return "HIGH"

    if status == "NO_CLEAR_EVIDENCE":
        return "HIGH"

    if any(term in q for term in high_risk_terms):
        return "HIGH"

    if status == "CONDITIONAL":
        return "MEDIUM"

    if status == "ALLOWED":
        return "LOW"

    return "MEDIUM"


def build_recommended_action(status: str, query: str, lang: str) -> str:
    q = (query or "").lower()

    if lang == "ja":
        if status == "PROHIBITED":
            return "実施は避け、引用条文と社内コンプライアンス部門へ確認してください。"

        if status == "NO_CLEAR_EVIDENCE":
            return "追加の規制文書、FAQ、関連ガイダンス、社内基準を確認し、コンプライアンス部門へ相談してください。"

        if status == "CONDITIONAL":
            if any(k in q for k in ["講演", "謝礼", "講演料", "原稿執筆料", "コンサル", "演者"]):
                return "事前契約、業務内容、適正対価（FMV相当）、透明性開示、記録保存の確認を推奨します。"
            if any(k in q for k in ["旅費", "宿泊", "航空券", "ファーストクラス", "ビジネスクラス", "渡航費"]):
                return "必要性、合理性、旅費クラス設定、社内旅費基準との整合を確認してください。"
            if any(k in q for k in ["弁当", "飲食", "ランチョン", "接待"]):
                return "対象者、金額、提供方法、イベント目的との整合を確認してください。"
            return "条件の適用範囲と社内基準の確認を推奨します。"

        if status == "ALLOWED":
            return "引用条文と社内手順に沿って最終確認のうえ実施してください。"

        return "関連資料と社内基準の確認を推奨します。"

    else:
        if status == "PROHIBITED":
            return "Do not proceed. Review the cited provisions and consult compliance."

        if status == "NO_CLEAR_EVIDENCE":
            return "Review additional regulations, FAQs, guidance, and internal policies before proceeding."

        if status == "CONDITIONAL":
            if any(k in q for k in ["speaker", "honorarium", "consulting", "consultant", "fee", "fmv"]):
                return "Confirm written agreement, legitimate need, fair market value, and documentation requirements."
            if any(k in q for k in ["travel", "airfare", "flight", "first class", "business class", "lodging"]):
                return "Confirm necessity, proportionality, travel class policy, and internal approval requirements."
            if any(k in q for k in ["meal", "lunch", "hospitality", "boxed lunch"]):
                return "Confirm audience eligibility, modesty, event purpose, and hospitality limits."
            return "Confirm the applicable conditions and internal policy requirements."

        if status == "ALLOWED":
            return "Proceed only after verifying the cited provisions and required internal approvals."

        return "Review the relevant materials and internal compliance standards."


class RAGEngine:
    def __init__(self) -> None:
        self.searcher = HybridSearcher()

        api_key = (
            os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        )
        print(f"DEBUG GOOGLE_API_KEY loaded={bool(api_key)} length={len(api_key)}")
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def initialize_data(self, data_dir: str = "./data") -> int:
        file_candidates = [
            os.path.join(data_dir, "Japan.jsonl"),
            os.path.join(data_dir, "US.jsonl"),
            os.path.join(data_dir, "UK.jsonl"),
            os.path.join(data_dir, "jp_code.jsonl"),
            os.path.join(data_dir, "jp_faq.jsonl"),
            os.path.join(data_dir, "jp_guidance.jsonl"),
            os.path.join(data_dir, "jp_transparency.jsonl"),
            os.path.join(data_dir, "us_code.jsonl"),
            os.path.join(data_dir, "us_guidance.jsonl"),
            os.path.join(data_dir, "us_faq.jsonl"),
            os.path.join(data_dir, "uk_code.jsonl"),
            os.path.join(data_dir, "uk_guidance.jsonl"),
        ]
        existing_files = [f for f in file_candidates if os.path.exists(f)]
        print(f"DEBUG existing_files={existing_files}")
        return self.searcher.load_jsonl_files(existing_files)

    def answer_question(
        self,
        query: str,
        lang: str,
        region: str,
        top_k: int = 8,
    ) -> dict[str, Any]:
        query = (query or "").strip()

        if not query:
            return {
                "status": EvidenceStatus.OUT_OF_SCOPE.value,
                "risk_level": "MEDIUM",
                "recommended_action": "質問を入力してください。" if lang == "ja" else "Please enter a question.",
                "answer": "質問を入力してください。" if lang == "ja" else "Please enter a question.",
                "reasoning": "空の質問には回答できません。" if lang == "ja" else "An empty question cannot be answered.",
                "key_evidence": "クエリが入力されていません。" if lang == "ja" else "No query was provided.",
                "citations": [],
                "search_scope": "",
            }

        if region == "Japan":
            return self._answer_for_jurisdiction(query, "JP", lang, top_k)

        if region == "US":
            return self._answer_for_jurisdiction(query, "US", lang, top_k)

        if region == "UK":
            return self._answer_for_jurisdiction(query, "UK", lang, top_k)

        if region == "US + UK":
            return {
                "US": self._answer_for_jurisdiction(query, "US", lang, top_k),
                "UK": self._answer_for_jurisdiction(query, "UK", lang, top_k),
            }

        return {
            "status": EvidenceStatus.OUT_OF_SCOPE.value,
            "risk_level": "MEDIUM",
            "recommended_action": "地域設定を確認してください。" if lang == "ja" else "Please review the region setting.",
            "answer": "地域設定が不正です。" if lang == "ja" else "Invalid region setting.",
            "reasoning": "対応していない地域が指定されました。" if lang == "ja" else "An unsupported region was selected.",
            "key_evidence": "",
            "citations": [],
            "search_scope": "",
        }

    def _answer_for_jurisdiction(
        self,
        query: str,
        jurisdiction: str,
        lang: str,
        top_k: int = 8,
    ) -> dict[str, Any]:
        search_result = self.searcher.fallback_search(
            query=query,
            jurisdiction=jurisdiction,
            top_k=top_k,
        )

        retrieved_docs = search_result.get("results", [])
        search_scope = search_result.get("search_scope", "")
        searched_queries = search_result.get("searched_queries", [])

        if search_result.get("status") == "NO_CLEAR_EVIDENCE":
            response = self.build_no_clear_evidence_response(
                query=query,
                lang=lang,
                results=retrieved_docs,
                search_scope=search_scope,
            )
            self._log(query, jurisdiction, retrieved_docs, response, searched_queries, search_result.get("stage", ""))
            return response

        if not self.client:
            response = {
                "status": EvidenceStatus.GENERATION_ERROR.value,
                "risk_level": "HIGH",
                "recommended_action": "API設定を確認してください。" if lang == "ja" else "Check API configuration.",
                "answer": "生成機能が利用できません。" if lang == "ja" else "Generation is unavailable.",
                "reasoning": "GOOGLE_API_KEY が設定されていないため、Gemini APIを利用できません。" if lang == "ja" else "Gemini API cannot be used because GOOGLE_API_KEY is not set.",
                "key_evidence": "検索結果は取得できましたが、生成処理が実行できませんでした。" if lang == "ja" else "Search results were retrieved, but generation could not be executed.",
                "citations": self.format_citations(retrieved_docs[:3]),
                "search_scope": search_scope,
            }
            self._log(query, jurisdiction, retrieved_docs, response, searched_queries, search_result.get("stage", ""))
            return response

        try:
            llm_response = self._generate_structured_answer(
                query=query,
                retrieved_docs=retrieved_docs,
                search_scope=search_scope,
                lang=lang,
            )
            self._log(query, jurisdiction, retrieved_docs, llm_response, searched_queries, search_result.get("stage", ""))
            return llm_response
        except Exception as e:
            response = {
                "status": EvidenceStatus.GENERATION_ERROR.value,
                "risk_level": "HIGH",
                "recommended_action": "引用根拠を確認し、手動で判断してください。" if lang == "ja" else "Review the citations and assess manually.",
                "answer": "回答生成中にエラーが発生しました。" if lang == "ja" else "An error occurred during answer generation.",
                "reasoning": "検索結果は取得できましたが、LLM生成に失敗しました。追加確認が必要です。" if lang == "ja" else "Search results were retrieved, but LLM generation failed. Additional verification is required.",
                "key_evidence": f"Gemini API failure: {type(e).__name__}",
                "citations": self.format_citations(retrieved_docs[:3]),
                "search_scope": search_scope,
            }
            self._log(query, jurisdiction, retrieved_docs, response, searched_queries, search_result.get("stage", ""))
            return response

    def build_no_clear_evidence_response(
        self,
        query: str,
        lang: str,
        results: list[dict[str, Any]],
        search_scope: str,
    ) -> dict[str, Any]:
        citations = self.format_citations(results[:3]) if results else []
        risk_level = assess_risk_level(EvidenceStatus.NO_CLEAR_EVIDENCE.value, query)
        recommended_action = build_recommended_action(EvidenceStatus.NO_CLEAR_EVIDENCE.value, query, lang)

        if lang == "ja":
            answer = "現時点で参照している規制文書の範囲では、当該ケースに直接対応する明確な根拠は確認できません。"
            reasoning = "本回答は『許容される』ことを示すものではありません。該当事項が明示されていないため、追加の規制文書、関連ガイダンス、社内コンプライアンス基準を確認したうえで慎重に判断すべきです。"
            key_evidence = "検索対象となった関連条項では、質問に直接言及する明確な記載は見当たりませんでした。"
        else:
            answer = "Within the currently referenced regulatory documents, no clear evidence was found that directly addresses this scenario."
            reasoning = "This does not imply that the conduct is permitted. Because the relevant point is not explicitly addressed, additional regulatory materials, related guidance, or internal compliance standards should be reviewed before making a decision."
            key_evidence = "No directly applicable clause addressing this specific scenario was identified in the retrieved materials."

        return {
            "status": EvidenceStatus.NO_CLEAR_EVIDENCE.value,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "answer": answer,
            "reasoning": reasoning,
            "key_evidence": key_evidence,
            "citations": citations,
            "search_scope": search_scope,
        }

    def _generate_structured_answer(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        search_scope: str,
        lang: str,
    ) -> dict[str, Any]:
        context_blocks: list[str] = []

        for idx, doc in enumerate(retrieved_docs[:5], start=1):
            meta = doc.get("metadata", {})
            context_blocks.append(
                "\n".join(
                    [
                        f"[Document {idx}]",
                        f"id: {meta.get('id', '')}",
                        f"regulation: {meta.get('regulation', '')}",
                        f"version: {meta.get('version', '')}",
                        f"section: {meta.get('section', '')}",
                        f"clause: {meta.get('clause', '')}",
                        f"title: {meta.get('title', '')}",
                        f"content: {doc.get('content', '')}",
                    ]
                )
            )

        prompt = f"""
{SYSTEM_PROMPT}

UI language:
{"Japanese" if lang == "ja" else "English"}

User question:
{query}

Search scope:
{search_scope}

Retrieved evidence:
{chr(10).join(context_blocks)}

Important instructions:
- Return JSON only.
- Do not wrap JSON in markdown code fences.
- If no explicit supporting clause exists, use status "NO_CLEAR_EVIDENCE".
- Absence of explicit evidence does not imply allowance.
- Prefer a practical compliance-support answer over a minimal answer.
- When evidence suggests a path forward, classify as CONDITIONAL and explain the conditions.

Return JSON only with keys:
status, answer, reasoning, key_evidence, citations
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        raw_text = (response.text or "").strip()
        cleaned_text = self._extract_json_text(raw_text)

        try:
            parsed = json.loads(cleaned_text)
        except Exception:
            return {
                "status": EvidenceStatus.GENERATION_ERROR.value,
                "risk_level": "HIGH",
                "recommended_action": build_recommended_action(EvidenceStatus.GENERATION_ERROR.value, query, lang),
                "answer": raw_text,
                "reasoning": "LLMのJSON出力を解析できませんでした。生テキストを表示しています。" if lang == "ja" else "The LLM JSON output could not be parsed. Raw text is displayed.",
                "key_evidence": "JSON parse failed",
                "citations": self.format_citations(retrieved_docs[:3]),
                "search_scope": search_scope,
            }

        return self._normalize_llm_output(parsed, retrieved_docs, search_scope, query, lang)

    def _extract_json_text(self, text: str) -> str:
        text = text.strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()

        return text

    def _normalize_llm_output(
        self,
        parsed: dict[str, Any],
        retrieved_docs: list[dict[str, Any]],
        search_scope: str,
        query: str,
        lang: str,
    ) -> dict[str, Any]:
        raw_status = str(parsed.get("status", "")).strip().upper()

        status_map = {
            "ALLOWED": EvidenceStatus.ALLOWED.value,
            "PROHIBITED": EvidenceStatus.PROHIBITED.value,
            "CONDITIONAL": EvidenceStatus.CONDITIONAL.value,
            "CONDITIONAL / CONTEXT-DEPENDENT": EvidenceStatus.CONDITIONAL.value,
            "CONTEXT-DEPENDENT": EvidenceStatus.CONDITIONAL.value,
            "NO_CLEAR_EVIDENCE": EvidenceStatus.NO_CLEAR_EVIDENCE.value,
            "NO CLEAR EVIDENCE FOUND": EvidenceStatus.NO_CLEAR_EVIDENCE.value,
            "NO CLEAR EVIDENCE": EvidenceStatus.NO_CLEAR_EVIDENCE.value,
            "OUT_OF_SCOPE": EvidenceStatus.OUT_OF_SCOPE.value,
            "CONFLICTING_EVIDENCE": EvidenceStatus.CONFLICTING_EVIDENCE.value,
        }

        normalized_status = status_map.get(raw_status, EvidenceStatus.NO_CLEAR_EVIDENCE.value)

        key_evidence = parsed.get("key_evidence", "")
        if isinstance(key_evidence, list):
            key_evidence = json.dumps(key_evidence, ensure_ascii=False)

        citations = parsed.get("citations", [])
        if not isinstance(citations, list) or not citations:
            citations = self.format_citations(retrieved_docs[:3])
        elif all(isinstance(c, str) and len(c) < 30 for c in citations):
            citations = self.format_citations(retrieved_docs[:3])

        risk_level = assess_risk_level(normalized_status, query)
        recommended_action = build_recommended_action(normalized_status, query, lang)

        return {
            "status": normalized_status,
            "risk_level": risk_level,
            "recommended_action": recommended_action,
            "answer": parsed.get("answer", ""),
            "reasoning": parsed.get("reasoning", ""),
            "key_evidence": key_evidence,
            "citations": citations,
            "search_scope": search_scope,
        }

    def format_citations(self, docs: list[dict[str, Any]]) -> list[str]:
        citations: list[str] = []

        for doc in docs:
            meta = doc.get("metadata", {})
            parts = [
                meta.get("regulation", ""),
                meta.get("version", ""),
                meta.get("section", ""),
                meta.get("clause", ""),
                meta.get("title", ""),
            ]
            citation = " / ".join([p for p in parts if p])
            if citation:
                citations.append(citation)

        return citations

    def _log(
        self,
        query: str,
        jurisdiction: str,
        retrieved_docs: list[dict[str, Any]],
        response: dict[str, Any],
        searched_queries: list[str],
        search_stage: str,
    ) -> None:
        record = {
            "query": query,
            "jurisdiction": jurisdiction,
            "searched_queries": searched_queries,
            "search_stage": search_stage,
            "status": response.get("status", ""),
            "risk_level": response.get("risk_level", ""),
            "recommended_action": response.get("recommended_action", ""),
            "search_scope": response.get("search_scope", ""),
            "retrieved_documents": [
                {
                    "id": d.get("metadata", {}).get("id", ""),
                    "regulation": d.get("metadata", {}).get("regulation", ""),
                    "section": d.get("metadata", {}).get("section", ""),
                    "clause": d.get("metadata", {}).get("clause", ""),
                    "title": d.get("metadata", {}).get("title", ""),
                    "hybrid_score": d.get("hybrid_score", 0.0),
                }
                for d in retrieved_docs[:10]
            ],
            "response": response,
        }
        write_audit_log(record)