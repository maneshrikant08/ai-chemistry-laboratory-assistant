from typing import List, Dict
import re

_URL_RE = re.compile(r"https?://\S+")


def build_system_prompt(response_mode: str) -> str:
    base = (
        "You are an AI Chemistry Laboratory Assistant. "
        "Prioritize safety and precision. "
        "Only answer from the provided evidence when evidence is present. "
        "If the provided evidence does not contain the answer, say so clearly. "
        "Do not fabricate citations or URLs. "
        "Avoid raw URLs in the response. "
        "When using PDF context, cite claims inline with source labels like [S1]. "
        "When using web results, cite claims inline with labels like [W1]. "
        "Use the actual current date provided in the user message for any date-sensitive reasoning. "
        "Do not infer today's date from web snippets. "
        "Do not mix unsupported assumptions with sourced facts. "
        "If the user is only greeting or chatting casually, respond naturally without asking for document context."
    )

    if response_mode == "concise":
        return (
            base
            + " Respond briefly with clear steps or bullet points. "
            + "Prefer 3 to 5 short bullets or a short paragraph. "
            + "Keep factual answers tight and cite the specific supporting source labels."
        )

    return (
        base
        + " Respond with a clearly detailed, structured explanation. "
        + "When evidence is available, expand the answer with multiple sections such as direct answer, explanation, step-by-step details, and important notes or precautions where relevant. "
        + "Do not give a short summary when the user selected detailed mode unless the evidence itself is very limited. "
        + "For important factual, procedural, safety, legal, financial, or date-sensitive claims, include inline citations."
    )


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("", text).strip()


def format_context(chunks: List[Dict]) -> str:
    lines = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        citation = chunk.get("citation", "S?")
        header = (
            f"[{citation}] Source: {meta.get('doc_name')} | Unit: {meta.get('unit_title')} | "
            f"Section: {meta.get('section')} | Page: {meta.get('page')}"
        )
        lines.append(header)
        lines.append(_strip_urls(chunk.get("text", "")))
    return "\n\n".join(lines)


def format_web_results(results: List[Dict]) -> str:
    lines = []
    for idx, r in enumerate(results, start=1):
        title = _strip_urls(r.get("title", ""))
        content = _strip_urls(r.get("content", ""))
        if title or content:
            citation = r.get("citation", f"W{idx}")
            lines.append(f"[{citation}] Source: {title}")
            lines.append(content)
    return "\n\n".join(lines)


def format_sources_list(chunks: List[Dict]) -> List[str]:
    seen = set()
    items = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        citation = chunk.get("citation", "S?")
        key = (
            meta.get("doc_name"),
            meta.get("unit_title"),
            meta.get("section"),
            meta.get("page"),
        )
        if key in seen:
            continue
        seen.add(key)
        items.append(
            f"[{citation}] {meta.get('doc_name')} | Unit: {meta.get('unit_title')} | "
            f"Section: {meta.get('section')} | Page: {meta.get('page')}"
        )
    return items


def format_web_sources(results: List[Dict]) -> List[str]:
    items = []
    for idx, r in enumerate(results, start=1):
        title = _strip_urls(r.get("title", ""))
        if title:
            citation = r.get("citation", f"W{idx}")
            items.append(f"[{citation}] {title}")
    return items
