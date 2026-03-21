import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.query_normalizer import build_search_queries


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pharma_compliance_docs")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class HybridSearcher:
    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        self.docs_cache: list[dict[str, Any]] = []
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []

            cache: list[dict[str, Any]] = []
            for doc_text, meta in zip(documents, metadatas):
                meta = meta or {}
                cache.append(
                    {
                        "content": doc_text,
                        "metadata": meta,
                    }
                )
            self.docs_cache = cache
        except Exception:
            self.docs_cache = []

    def load_jsonl_files(self, file_paths: list[str]) -> int:
        records: list[dict[str, Any]] = []

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                continue

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    regulation_name = record.get("regulation", "")

                    if regulation_name == "公競規":
                        inferred_jurisdiction = "JP"
                    elif regulation_name == "PhRMA Code":
                        inferred_jurisdiction = "US"
                    elif regulation_name == "ABPI Code":
                        inferred_jurisdiction = "UK"
                    else:
                        inferred_jurisdiction = ""

                    metadata = {
                        "id": record.get("id", ""),
                        "jurisdiction": record.get("jurisdiction", inferred_jurisdiction),
                        "regulation": record.get("regulation", ""),
                        "document_type": record.get("document_type", "code"),
                        "source_authority": record.get("source_authority", "official"),
                        "version": record.get("version", ""),
                        "effective_date": record.get("effective_date", ""),
                        "section": record.get("section", ""),
                        "clause": record.get("clause", ""),
                        "title": record.get("title", ""),
                        "language": record.get("language", ""),
                        "superseded": str(record.get("superseded", False)),
                        "scenario_tags": json.dumps(record.get("scenario_tags", []), ensure_ascii=False),
                        "keywords": json.dumps(record.get("keywords", []), ensure_ascii=False),
                    }

                    content = record.get("content", "")
                    if not content:
                        continue

                    records.append(
                        {
                            "id": record.get("id", ""),
                            "content": content,
                            "metadata": metadata,
                        }
                    )

        if not records:
            return 0

        ids = [r["id"] for r in records]
        documents = [r["content"] for r in records]
        metadatas = [r["metadata"] for r in records]
        embeddings = self.embedding_model.encode(documents).tolist()

        existing = self.collection.get(ids=ids)
        existing_ids = set(existing.get("ids", []) or [])

        add_ids: list[str] = []
        add_docs: list[str] = []
        add_metas: list[dict[str, Any]] = []
        add_embs: list[list[float]] = []

        for record, emb in zip(records, embeddings):
            if record["id"] in existing_ids:
                continue
            add_ids.append(record["id"])
            add_docs.append(record["content"])
            add_metas.append(record["metadata"])
            add_embs.append(emb)

        if add_ids:
            self.collection.add(
                ids=add_ids,
                documents=add_docs,
                metadatas=add_metas,
                embeddings=add_embs,
            )

        self._refresh_cache()
        return len(add_ids)

    def _dense_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query).tolist()
        where = self._build_chroma_where(filters)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        found_docs = results.get("documents", [[]])[0]
        found_metas = results.get("metadatas", [[]])[0]
        found_distances = results.get("distances", [[]])[0]

        output: list[dict[str, Any]] = []

        for doc_text, meta, distance in zip(found_docs, found_metas, found_distances):
            dense_score = 1.0 / (1.0 + _safe_float(distance, 9999.0))
            output.append(
                {
                    "content": doc_text,
                    "metadata": meta or {},
                    "dense_score": dense_score,
                }
            )
        return output

    def _bm25_like_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filtered_docs = self._filter_docs_cache(filters)

        if not filtered_docs:
            return []

        texts = [d["content"] for d in filtered_docs]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])

        scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

        ranked = sorted(
            zip(filtered_docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        output: list[dict[str, Any]] = []
        for doc, score in ranked:
            output.append(
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "bm25_score": _safe_float(score),
                }
            )
        return output

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        dense_results = self._dense_search(query, top_k=top_k, filters=filters)
        sparse_results = self._bm25_like_search(query, top_k=top_k, filters=filters)

        merged: dict[str, dict[str, Any]] = {}

        for item in dense_results:
            doc_id = item["metadata"].get("id", "")
            merged.setdefault(
                doc_id,
                {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "hybrid_score": 0.0,
                },
            )
            merged[doc_id]["dense_score"] = item.get("dense_score", 0.0)

        for item in sparse_results:
            doc_id = item["metadata"].get("id", "")
            merged.setdefault(
                doc_id,
                {
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "hybrid_score": 0.0,
                },
            )
            merged[doc_id]["bm25_score"] = item.get("bm25_score", 0.0)

        for doc_id in merged:
            dense = _safe_float(merged[doc_id].get("dense_score", 0.0))
            sparse = _safe_float(merged[doc_id].get("bm25_score", 0.0))
            merged[doc_id]["hybrid_score"] = (0.55 * dense) + (0.45 * sparse)

        ranked = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return ranked[:top_k]

    def fallback_search(
        self,
        query: str,
        jurisdiction: str,
        top_k: int = 8,
    ) -> dict[str, Any]:
        regulation_name = self._jurisdiction_to_regulation_name(jurisdiction)

        stages = [
            {
                "name": "strict_regulation_search",
                "filters": {
                    "regulation": regulation_name,
                },
                "top_k": max(top_k, 5),
            },
            {
                "name": "expanded_regulation_search",
                "filters": {
                    "regulation": regulation_name,
                },
                "top_k": max(top_k, 8),
            },
        ]

        expanded_queries = build_search_queries(query)
        all_results: list[dict[str, Any]] = []

        for stage in stages:
            stage_results: list[dict[str, Any]] = []

            print(f"DEBUG stage={stage['name']} filters={stage['filters']}")
            print(f"DEBUG expanded_queries={expanded_queries}")

            for q in expanded_queries:
                hits = self.hybrid_search(
                    query=q,
                    top_k=stage["top_k"],
                    filters=stage["filters"],
                )
                print(f"DEBUG query={q} hits={len(hits)}")
                stage_results.extend(hits)

            deduped = self._deduplicate_results(stage_results)

            if self.has_sufficient_evidence(deduped):
                return {
                    "status": "FOUND",
                    "stage": stage["name"],
                    "results": deduped[: stage["top_k"]],
                    "searched_queries": expanded_queries,
                    "search_scope": self._describe_search_scope(stage["filters"]),
                }

            all_results.extend(deduped)

        deduped_all = self._deduplicate_results(all_results)

        return {
            "status": "NO_CLEAR_EVIDENCE",
            "stage": "completed_all_fallbacks",
            "results": deduped_all[:top_k],
            "searched_queries": expanded_queries,
            "search_scope": self._describe_search_scope({"regulation": regulation_name}),
        }

    def has_sufficient_evidence(
        self,
        results: list[dict[str, Any]],
        min_score: float = 0.25,
    ) -> bool:
        strong = [
            r for r in results
            if _safe_float(r.get("hybrid_score", 0.0)) >= min_score
        ]
        return len(strong) >= 1

    def _filter_docs_cache(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not filters:
            return self.docs_cache

        output: list[dict[str, Any]] = []

        for doc in self.docs_cache:
            meta = doc.get("metadata", {})
            matched = True

            for key, value in filters.items():
                meta_value = meta.get(key, "")

                if key == "superseded":
                    meta_bool = str(meta_value).lower() == "true"
                    if isinstance(value, bool):
                        if meta_bool != value:
                            matched = False
                            break
                    else:
                        if str(meta_value) != str(value):
                            matched = False
                            break
                elif isinstance(value, list):
                    if str(meta_value) not in [str(v) for v in value]:
                        matched = False
                        break
                else:
                    if str(meta_value) != str(value):
                        matched = False
                        break

            if matched:
                output.append(doc)

        return output

    def _build_chroma_where(
        self,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not filters:
            return None

        clauses: list[dict[str, Any]] = []

        for key, value in filters.items():
            if isinstance(value, list):
                clauses.append({key: {"$in": [str(v) for v in value]}})
            elif isinstance(value, bool):
                clauses.append({key: str(value)})
            else:
                clauses.append({key: str(value)})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _deduplicate_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        for item in results:
            doc_id = item.get("metadata", {}).get("id", "")
            if not doc_id:
                continue

            if doc_id not in merged:
                merged[doc_id] = item
                continue

            existing_score = _safe_float(merged[doc_id].get("hybrid_score", 0.0))
            new_score = _safe_float(item.get("hybrid_score", 0.0))
            if new_score > existing_score:
                merged[doc_id] = item

        ranked = sorted(
            merged.values(),
            key=lambda x: x.get("hybrid_score", 0.0),
            reverse=True,
        )
        return ranked

    def _describe_search_scope(self, filters: dict[str, Any] | None) -> str:
        if not filters:
            return "broad search across available regulation documents"

        if "regulation" in filters:
            regulation = filters["regulation"]
            if regulation == "公競規":
                return "公競規 (current version)"
            if regulation == "PhRMA Code":
                return "PhRMA Code (current version)"
            if regulation == "ABPI Code":
                return "ABPI Code (current version)"
            return f"{regulation} (current version)"

        jurisdiction = filters.get("jurisdiction", "unknown")
        source_authority = filters.get("source_authority", "unknown")
        document_types = filters.get("document_type", [])

        return f"jurisdiction={jurisdiction}, authority={source_authority}, document_types={document_types}"

    def _jurisdiction_to_regulation_name(self, jurisdiction: str) -> str:
        mapping = {
            "JP": "公競規",
            "US": "PhRMA Code",
            "UK": "ABPI Code",
        }
        return mapping.get(jurisdiction, "")