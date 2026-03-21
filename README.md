# Pharma Compliance AI

Pharma Compliance AI is a Retrieval-Augmented Generation (RAG)-based compliance assistant designed for pharmaceutical regulatory review workflows.

It supports practical decision-making by combining:

- regulation-aware retrieval
- citation-based answers
- risk classification
- recommended next actions

---

## 🎯 What This Tool Solves

In real compliance workflows:

- Regulations are fragmented
- Answers are often not explicitly written
- Users need **practical judgment**, not just citations

This tool provides:

👉 Evidence  
👉 Interpretation  
👉 Risk level  
👉 Next action

---

## 🌍 Covered Regulations

- 🇯🇵 Japan: 公競規
- 🇺🇸 United States: PhRMA Code + OIG guidance
- 🇬🇧 United Kingdom: ABPI Code

---

## 🌐 Language Support

- Japanese
- English

---

## ⚙️ Core Features

- Hybrid search (vector + keyword)
- Query normalization (JP/EN)
- Citation-based answers
- Risk classification (LOW / MEDIUM / HIGH)
- Recommended actions
- Multi-region comparison (US vs UK)
- Audit logging

---

## 🧠 Output Example

```json
{
  "status": "CONDITIONAL",
  "risk_level": "MEDIUM",
  "recommended_action": "Confirm FMV and written agreement",
  "answer": "...",
  "reasoning": "...",
  "citations": ["..."]
}
```
