from pathlib import Path
from datetime import date
import re
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import config
from models.llm import get_llm
from models.embeddings import get_embeddings
from utils.vector_store import (
    load_or_build_index,
    build_index_from_documents,
    extract_experiments_from_store,
)
from utils.document_loader import load_pdf, load_docx, load_txt, load_image
from utils.chunking import chunk_pdf_pages, chunk_text
from utils.retrieval import (
    extract_experiment_number,
    extract_unit_number,
    retrieve_with_scores,
)
from utils.web_search import search_web
from utils.prompting import (
    build_system_prompt,
    format_context,
    format_web_results,
    format_sources_list,
    format_web_sources,
)
from utils.memory import load_history, save_history

_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_GENERAL_CHAT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*(hi|hello|hey|hii|good morning|good afternoon|good evening)\b",
        r"\bmy name is\b",
        r"\bi am\b",
        r"\bwho am i\b",
        r"\bwhat is my name\b",
        r"\bhow old am i\b",
        r"\bwhat is my age\b",
        r"\bwhen is my birthday\b",
        r"\bthank you\b",
        r"\bthanks\b",
    ]
]


def _infer_unit_filter(query: str, unit_titles: list[str]) -> str | None:
    q = query.lower()
    for unit in unit_titles:
        if unit.lower() in q:
            return unit
    query_unit = extract_unit_number(query)
    if query_unit:
        for unit in unit_titles:
            if extract_unit_number(unit) == query_unit:
                return unit
    return None


def _infer_experiment_filter(query: str, experiments: list[str]) -> str | None:
    q = query.lower()
    for exp in experiments:
        if exp.lower() in q:
            return exp
    query_experiment = extract_experiment_number(query)
    if query_experiment:
        for exp in experiments:
            if extract_experiment_number(exp) == query_experiment:
                return exp
    return None


def _should_use_web(query: str, chunk_count: int) -> bool:
    q = query.lower()
    if any(k in q for k in ["latest", "recent", "current", "today", "standard"]):
        return True
    return chunk_count < config.MIN_CHUNKS


def _has_relevant_overlap(query: str, chunks) -> bool:
    tokens = set(t.lower() for t in _WORD_RE.findall(query))
    if not tokens:
        return False
    for c in chunks:
        text = (c.get("text") or "").lower()
        if any(t in text for t in tokens):
            return True
    return False


def _is_general_chat_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in _GENERAL_CHAT_PATTERNS)


def _has_structured_match(query: str, chunks) -> bool:
    query_unit = extract_unit_number(query)
    query_experiment = extract_experiment_number(query)
    if not query_unit and not query_experiment:
        return False

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if query_unit and extract_unit_number(meta.get("unit_title")) == query_unit:
            return True
        if query_experiment and extract_experiment_number(meta.get("experiment_name")) == query_experiment:
            return True
    return False


def _attach_citations(chunks, web_results):
    cited_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        enriched = dict(chunk)
        enriched["citation"] = f"S{idx}"
        cited_chunks.append(enriched)

    cited_web_results = []
    for idx, result in enumerate(web_results, start=1):
        enriched = dict(result)
        enriched["citation"] = f"W{idx}"
        cited_web_results.append(enriched)

    return cited_chunks, cited_web_results


def _history_to_messages(history: list[dict], max_messages: int):
    messages = []
    for item in history[-max_messages:]:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _safe_load_history(path: Path) -> list[dict]:
    try:
        return load_history(path)
    except Exception:
        return []


def _safe_save_history(path: Path, messages: list[dict]) -> None:
    try:
        save_history(path, messages)
    except Exception as exc:
        st.warning(f"Failed to save chat history: {exc}")


@st.cache_resource(show_spinner=False)
def _load_base_index():
    try:
        embeddings = get_embeddings()
        store, unit_titles, experiments = load_or_build_index(
            config.DATA_DIR, config.INDEX_DIR, embeddings
        )
        return store, unit_titles, experiments
    except Exception as exc:
        raise RuntimeError(f"Base index initialization failed: {exc}") from exc


def _build_user_index(files) -> tuple | None:
    if not files:
        return None

    try:
        embeddings = get_embeddings()
        documents = []

        for f in files:
            name = f.name
            suffix = Path(name).suffix.lower()
            temp_path = None
            try:
                if suffix == ".pdf":
                    temp_path = Path("_tmp_upload.pdf")
                    temp_path.write_bytes(f.read())
                    pages = load_pdf(temp_path)
                    documents.extend(chunk_pdf_pages(pages, name))
                elif suffix == ".docx":
                    temp_path = Path("_tmp_upload.docx")
                    temp_path.write_bytes(f.read())
                    text = load_docx(temp_path)
                    documents.extend(chunk_text(text, name))
                elif suffix == ".txt":
                    text = f.read().decode("utf-8", errors="ignore")
                    documents.extend(chunk_text(text, name))
                elif suffix in [".png", ".jpg", ".jpeg"]:
                    temp_path = Path("_tmp_upload.png")
                    temp_path.write_bytes(f.read())
                    text = load_image(temp_path)
                    documents.extend(chunk_text(text, name))
            except Exception as exc:
                raise RuntimeError(f"Failed to process uploaded file '{name}': {exc}") from exc
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        if not documents:
            return None

        store = build_index_from_documents(documents, embeddings)
        experiments = extract_experiments_from_store(store)
        return store, experiments
    except Exception as exc:
        raise RuntimeError(f"Failed to build upload index: {exc}") from exc


def _render_structured_output(
    answer: str,
    chunks,
    web_results,
    web_attempted: bool,
    pdf_found: bool,
    source_mode: str,
    search_mode: str,
    show_sources: bool = True,
):
    try:
        if not show_sources:
            st.markdown(answer)
            return

        sources = format_sources_list(chunks)
        web_sources = format_web_sources(web_results)

        parts = []
        parts.append(f"**Source Mode:** {source_mode}")
        parts.append("**Answer**")
        parts.append(answer)

        if search_mode in {"Auto", "Documents"}:
            parts.append("**Sources**")
            if sources:
                parts.extend([f"- {s}" for s in sources])
            else:
                parts.append("- Not found in PDF.")

        if search_mode in {"Auto", "Web"}:
            parts.append("**Web Search**")
            if web_attempted:
                if web_sources:
                    parts.append("Sources:")
                    parts.extend([f"- {s}" for s in web_sources])
                else:
                    parts.append("Web search was attempted, but no results were returned.")
            elif search_mode == "Web":
                parts.append("Web search was not used.")

        st.markdown("\n\n".join(parts))
    except Exception as exc:
        st.error(f"Failed to render response output: {exc}")


def _inject_custom_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1220;
            --paper: rgba(15, 23, 42, 0.78);
            --ink: #e6eef8;
            --muted: #96a7bf;
            --teal: #22c1a7;
            --teal-deep: #0f766e;
            --amber: #f0b35b;
            --line: rgba(148, 163, 184, 0.16);
            --shadow: 0 20px 50px rgba(2, 6, 23, 0.38);
            --radius: 24px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(34, 193, 167, 0.12), transparent 24%),
                radial-gradient(circle at top right, rgba(240, 179, 91, 0.12), transparent 24%),
                linear-gradient(180deg, #09111e 0%, #0d1729 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 15, 28, 0.98), rgba(14, 22, 38, 0.96));
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.6rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        p, label, .stCaption, .stMarkdown, .stRadio, .stSelectbox, .stTextInput {
            color: var(--ink);
        }

        [data-testid="stChatMessage"] {
            background: var(--paper);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 0.4rem 0.8rem;
            backdrop-filter: blur(10px);
        }

        [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"],
        [data-testid="stChatMessageAvatarAssistant"] + [data-testid="stChatMessageContent"] {
            color: var(--ink);
        }

        [data-testid="stChatInput"] {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }

        [data-testid="stChatInput"] textarea {
            color: var(--ink) !important;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid transparent;
            background: linear-gradient(135deg, #0f766e, #22c1a7);
            color: white;
            font-weight: 700;
            padding: 0.62rem 1rem;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.28);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            border: 1px solid transparent;
            color: white;
            background: linear-gradient(135deg, #149081, #2dd4bf);
        }

        .stTextInput input, .stTextArea textarea {
            border-radius: 16px !important;
            background: rgba(15, 23, 42, 0.9) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }

        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 16px !important;
            background: rgba(15, 23, 42, 0.9) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }

        div[data-baseweb="radio"] > div {
            gap: 0.5rem;
        }

        div[data-baseweb="radio"] label,
        .stFileUploader label,
        .stSelectbox label,
        .stTextInput label {
            color: var(--ink) !important;
        }

        .stAlert {
            background: rgba(15, 23, 42, 0.92) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
            border-radius: 18px !important;
        }

        .app-hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.4rem;
            padding: 1.8rem 1.9rem;
            border-radius: 30px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background:
                linear-gradient(135deg, rgba(15,23,42,0.92), rgba(15,23,42,0.72)),
                linear-gradient(135deg, rgba(34,193,167,0.08), rgba(240,179,91,0.08));
            box-shadow: var(--shadow);
        }

        .app-hero::before {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            right: -60px;
            top: -80px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(34,193,167,0.22), transparent 70%);
        }

        .app-badge {
            display: inline-block;
            margin-bottom: 0.8rem;
            padding: 0.4rem 0.72rem;
            border-radius: 999px;
            background: rgba(34,193,167,0.12);
            color: #78e3d2;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .app-hero h1 {
            margin: 0 0 0.45rem;
            font-size: clamp(2rem, 3vw, 3rem);
        }

        .app-hero p {
            margin: 0;
            max-width: 780px;
            color: var(--muted);
            line-height: 1.7;
            font-size: 1rem;
        }

        .stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.3rem;
        }

        .stat-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
        }

        .stat-card strong {
            display: block;
            margin-bottom: 0.2rem;
            color: #7ee5d5;
            font-size: 1rem;
        }

        .stat-card span {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .stFileUploader > div {
            background: rgba(15, 23, 42, 0.72);
            border-radius: 20px;
            border: 1px dashed rgba(148, 163, 184, 0.22);
        }

        @media (max-width: 900px) {
            .stat-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header():
    st.markdown(
        """
        <section class="app-hero">
            <div class="app-badge">Grounded Chemistry Assistant</div>
            <h1>AI Chemistry Laboratory Assistant</h1>
            <p>
                A modern RAG-based chatbot for chemistry lab workflows, built with local document retrieval,
                live web fallback, short-term memory, and source-aware answers.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stat-row">
            <div class="stat-card">
                <strong>Documents First</strong>
                <span>Uses indexed manuals and uploaded files as the primary evidence source.</span>
            </div>
            <div class="stat-card">
                <strong>Web Fallback</strong>
                <span>Switches to current web evidence when local content is not enough.</span>
            </div>
            <div class="stat-card">
                <strong>Cleaner UX</strong>
                <span>Short-term memory, source modes, and general-chat handling keep the experience natural.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="AI Chemistry Laboratory Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_custom_styles()
    _render_app_header()

    if not config.OPENAI_API_KEY and config.EMBED_PROVIDER == "openai":
        st.error("OPENAI_API_KEY is required to build the embeddings index.")
        st.stop()

    try:
        base_store, unit_titles, base_experiments = _load_base_index()
    except Exception as e:
        st.error(f"Failed to load base index: {e}")
        st.stop()

    if "user_store" not in st.session_state:
        st.session_state.user_store = None
    if "user_experiments" not in st.session_state:
        st.session_state.user_experiments = []
    if "messages" not in st.session_state:
        st.session_state.messages = _safe_load_history(config.CHAT_HISTORY_PATH)

    combined_experiments = sorted(
        set(base_experiments + st.session_state.user_experiments)
    )

    with st.sidebar:
        st.header("Settings")
        provider_options = ["openai", "groq", "gemini"]
        default_provider_index = (
            provider_options.index(config.DEFAULT_PROVIDER)
            if config.DEFAULT_PROVIDER in provider_options
            else 0
        )
        provider = st.selectbox("Provider", provider_options, index=default_provider_index)
        model_name = st.text_input("Model", value=config.DEFAULT_MODEL)
        response_mode = st.radio("Response Mode", ["concise", "detailed"], index=0)
        search_mode = st.radio(
            "Search Mode", ["Auto", "Documents", "Web"], index=0
        )

        st.divider()
        st.caption(
            f"Short-term memory uses the last {config.MAX_MEMORY_MESSAGES} messages."
        )
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []
            _safe_save_history(config.CHAT_HISTORY_PATH, st.session_state.messages)
            st.rerun()

        st.divider()
        st.subheader("Upload Documents")
        uploads = st.file_uploader(
            "Upload PDFs, DOCX, TXT, or images",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if st.button("Upload", use_container_width=True):
            try:
                result = _build_user_index(uploads)
                if result:
                    st.session_state.user_store, st.session_state.user_experiments = result
                    st.success("Uploaded documents indexed successfully.")
                else:
                    st.warning("No valid content was found in the uploaded files.")
            except Exception as exc:
                st.error(str(exc))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a lab question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        _safe_save_history(config.CHAT_HISTORY_PATH, st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm(provider, model_name)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                is_general_chat = _is_general_chat_query(prompt)
                try:
                    filters = {}
                    inferred_unit = _infer_unit_filter(prompt, unit_titles)
                    inferred_experiment = _infer_experiment_filter(prompt, combined_experiments)

                    if inferred_unit:
                        filters["unit_title"] = inferred_unit
                    else:
                        inferred_unit_number = extract_unit_number(prompt)
                        if inferred_unit_number:
                            filters["unit_number"] = inferred_unit_number

                    if inferred_experiment:
                        filters["experiment_name"] = inferred_experiment
                    else:
                        inferred_experiment_number = extract_experiment_number(prompt)
                        if inferred_experiment_number:
                            filters["experiment_number"] = inferred_experiment_number

                    chunks = []
                    pdf_found = False
                    if not is_general_chat and search_mode != "Web":
                        base_results = retrieve_with_scores(
                            base_store, prompt, config.TOP_K, filters
                        )
                        user_store = st.session_state.user_store
                        user_results = []
                        if user_store is not None:
                            user_results = retrieve_with_scores(
                                user_store, prompt, config.TOP_K, filters
                            )

                        all_results = base_results + user_results
                        all_results.sort(key=lambda x: x[1])
                        top_results = all_results[: config.TOP_K]

                        chunks = [
                            {
                                "text": doc.page_content,
                                "metadata": doc.metadata,
                                "score": score,
                            }
                            for doc, score in top_results
                        ]

                        overlap_ok = _has_relevant_overlap(prompt, chunks)
                        structured_match_ok = _has_structured_match(prompt, chunks)
                        pdf_found = len(chunks) > 0 and (overlap_ok or structured_match_ok)

                    use_web = False
                    if is_general_chat:
                        use_web = False
                    elif search_mode == "Web":
                        use_web = True
                    elif search_mode == "Auto":
                        use_web = _should_use_web(prompt, len(chunks)) or not pdf_found

                    web_results = search_web(prompt) if use_web else []
                    if pdf_found and search_mode != "Web":
                        chunks, _ = _attach_citations(chunks, [])
                        web_results = []
                    else:
                        chunks = []
                        _, web_results = _attach_citations([], web_results)

                    if chunks:
                        source_mode = "Documents"
                    elif web_results:
                        source_mode = "Web"
                    else:
                        source_mode = "None"

                    system_prompt = build_system_prompt(response_mode)
                    context_text = format_context(chunks)
                    web_text = format_web_results(web_results)

                    if is_general_chat:
                        source_mode = "General"

                    if not chunks and not web_results and not is_general_chat:
                        answer = (
                            "I could not find this in the indexed documents or current web results."
                        )
                    else:
                        today = date.today().isoformat()
                        if response_mode == "detailed":
                            response_instruction = (
                                "- Write a genuinely detailed answer.\n"
                                "- Use these sections when relevant: Direct Answer, Explanation, Steps or Breakdown, Important Notes or Precautions.\n"
                                "- Make the answer clearly more descriptive than concise mode.\n"
                                "- If evidence is limited, still explain the available evidence fully without inventing missing details.\n"
                            )
                        else:
                            response_instruction = (
                                "- Keep the answer concise.\n"
                                "- Prefer 3 to 5 short bullets or one compact paragraph.\n"
                                "- Give only the most important information.\n"
                            )
                        user_prompt = (
                            f"Current Date: {today}\n"
                            f"Active Source Mode: {source_mode}\n\n"
                            f"Question: {prompt}\n\n"
                            f"PDF Context:\n{context_text or 'None'}\n\n"
                            f"Web Results:\n{web_text or 'None'}\n\n"
                            "Instructions:\n"
                            "- If this is casual or personal conversation, answer naturally and do not mention missing sources.\n"
                            "- Use PDF context when Source Mode is Documents.\n"
                            "- Use web results when Source Mode is Web.\n"
                            "- If both are empty, say the answer was not found.\n"
                            "- For date-sensitive questions, use the Current Date above.\n"
                            "- Cite factual statements with the provided source labels.\n"
                            f"{response_instruction}"
                        )
                        memory_messages = _history_to_messages(
                            st.session_state.messages[:-1], config.MAX_MEMORY_MESSAGES
                        )

                        try:
                            response = llm.invoke(
                                [
                                    SystemMessage(content=system_prompt),
                                    *memory_messages,
                                    HumanMessage(content=user_prompt),
                                ]
                            )
                            answer = response.content
                        except Exception as exc:
                            answer = f"Error getting response: {exc}"
                except Exception as exc:
                    answer = f"Error while processing the request: {exc}"
                    chunks = []
                    web_results = []
                    pdf_found = False
                    use_web = False
                    source_mode = "None"

                _render_structured_output(
                    answer,
                    chunks,
                    web_results,
                    web_attempted=use_web,
                    pdf_found=pdf_found,
                    source_mode=source_mode,
                    search_mode=search_mode,
                    show_sources=not is_general_chat,
                )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        _safe_save_history(config.CHAT_HISTORY_PATH, st.session_state.messages)


if __name__ == "__main__":
    main()
