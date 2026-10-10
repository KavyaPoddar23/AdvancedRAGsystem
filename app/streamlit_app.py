
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

from app.corrective_rag import CorrectiveRAG
from app.conversation_memory import ConversationMemory


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Nexus RAG | Intelligent Knowledge Assistant",
    page_icon="✳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM DESIGN SYSTEM
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
        --bg: #0b0d14;
        --panel: #121522;
        --panel-soft: #171b2b;
        --border: #282d41;
        --text: #f2f3fa;
        --muted: #a0a6bd;
        --accent: #a78bfa;
        --accent-blue: #7c9cff;
        --success: #65ddb0;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(110, 76, 210, 0.13),
                transparent 34%
            ),
            radial-gradient(
                ellipse at 90% 15%,
                rgba(63, 112, 224, 0.10),
                transparent 30%
            ),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 13, 20, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #10121d;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.035em;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.06em;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .brand-mark {
        color: #bba5ff;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.70rem;
        font-weight: 700;
        color: #b8a4ff;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 5vw, 4.1rem);
        line-height: 1.08;
        letter-spacing: -0.065em;
        font-weight: 700;
        color: #f8f8ff;
        margin: 0 0 1rem 0;
    }

    .hero-gradient {
        background: linear-gradient(
            100deg,
            #c4b5fd 0%,
            #a5b4fc 45%,
            #7dd3fc 100%
        );
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-description {
        color: #aeb4ca;
        font-size: 1.02rem;
        line-height: 1.8;
        max-width: 760px;
        margin-bottom: 1.5rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.4rem 0.75rem;
        border: 1px solid #344c4a;
        border-radius: 999px;
        background: rgba(101, 221, 176, 0.07);
        color: #8be8c3;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem 0.3rem 0.3rem 0;
    }

    .tech-pill {
        display: inline-block;
        padding: 0.4rem 0.65rem;
        border: 1px solid #30354b;
        border-radius: 8px;
        background: #151827;
        color: #c4c9dd;
        font-size: 0.74rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    .section-label {
        font-size: 0.77rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #929ab6;
        margin: 1.8rem 0 0.85rem 0;
    }

    .feature-card {
        min-height: 165px;
        padding: 1.1rem;
        border-radius: 15px;
        border: 1px solid var(--border);
        background: linear-gradient(
            145deg,
            rgba(26, 30, 47, 0.95),
            rgba(17, 20, 32, 0.95)
        );
        transition: border-color 0.2s ease;
    }

    .feature-icon {
        font-size: 1.35rem;
        margin-bottom: 0.8rem;
    }

    .feature-title {
        color: #f4f4fc;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.45rem;
    }

    .feature-description {
        color: #a3aac2;
        font-size: 0.81rem;
        line-height: 1.65;
    }

    .metric-card {
        border: 1px solid var(--border);
        background: rgba(22, 25, 40, 0.8);
        border-radius: 13px;
        padding: 0.95rem 1rem;
    }

    .metric-label {
        color: #9ca4bf;
        font-size: 0.75rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        color: #f6f6ff;
        font-weight: 700;
    }

    .metric-note {
        color: #8be8c3;
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }

    .sidebar-note {
        color: #929ab4;
        font-size: 0.78rem;
        line-height: 1.65;
    }

    .footer {
        text-align: center;
        color: #747c97;
        font-size: 0.78rem;
        padding-top: 2rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(47, 52, 76, 0.75);
        border-radius: 15px;
        background: rgba(19, 22, 36, 0.72);
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stChatInput"] {
        border-color: #414664;
        background: #141725;
    }

    div[data-testid="stChatInput"] textarea {
        color: #f5f5ff;
    }

    .stButton > button {
        border: 1px solid #333950;
        border-radius: 10px;
        background: #171a2a;
        color: #eceeff;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #a78bfa;
        color: #ffffff;
        background: #222039;
    }

    [data-testid="stMetric"] {
        background: rgba(22, 25, 40, 0.8);
        border: 1px solid var(--border);
        padding: 0.9rem;
        border-radius: 12px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(18, 21, 34, 0.65);
    }

    hr {
        border-color: #282d41;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero-title {
            font-size: 2.3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION INITIALIZATION
# ---------------------------------------------------------

def initialize_session():
    """Initialize session variables only once."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(
            session_id="nexus-rag-session"
        )

    if "rag" not in st.session_state:
        st.session_state.rag = None

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if "show_transcript" not in st.session_state:
        st.session_state.show_transcript = False

    if "show_state" not in st.session_state:
        st.session_state.show_state = False

    if "show_diagnostics" not in st.session_state:
        st.session_state.show_diagnostics = False


initialize_session()


# ---------------------------------------------------------
# KNOWLEDGE BASE STATISTICS
# ---------------------------------------------------------

def get_knowledge_base_stats():
    """Read document and chunk counts without loading the models."""

    documents_dir = Path("data/documents")
    chunks_file = Path("data/processed/chunks.jsonl")

    document_count = 0
    chunk_count = 0

    if documents_dir.exists():
        document_count = sum(
            1
            for path in documents_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".txt", ".pdf"}
        )

    if chunks_file.exists():
        try:
            with chunks_file.open("r", encoding="utf-8") as file:
                chunk_count = sum(
                    1 for line in file if line.strip()
                )
        except OSError:
            chunk_count = 0

    return document_count, chunk_count


def get_rag():
    """Create the RAG pipeline when it is first needed."""

    if st.session_state.rag is None:
        with st.spinner("Initializing the retrieval pipeline..."):
            st.session_state.rag = CorrectiveRAG()

    return st.session_state.rag


# ---------------------------------------------------------
# MESSAGE AND SOURCE HELPERS
# ---------------------------------------------------------

def display_sources(sources):
    """Display source information returned by the RAG pipeline."""

    if not sources:
        return

    with st.expander(
        f"📚 Evidence and citations · {len(sources)} source(s)",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            if isinstance(source, str):
                source_name = source
                page = None
            elif isinstance(source, dict):
                source_name = (
                    source.get("source")
                    or source.get("filename")
                    or source.get("document")
                    or source.get("file")
                    or "Unknown source"
                )
                page = source.get("page")
            else:
                source_name = str(source)
                page = None

            st.markdown(f"**[{index}] {source_name}**")

            if page is not None:
                st.caption(f"Page {page}")

            if isinstance(source, dict):
                excerpt = (
                    source.get("text")
                    or source.get("content")
                    or source.get("snippet")
                )

                if excerpt:
                    st.markdown(f"> {excerpt}")

            if index < len(sources):
                st.divider()


def display_diagnostics(metadata):
    """Display retrieval and generation details when available."""

    if not metadata:
        st.caption("No diagnostic information is available.")
        return

    with st.expander("Inspect retrieval and reasoning metadata"):
        resolved_question = metadata.get("resolved_question")

        if resolved_question:
            st.markdown("**Resolved question**")
            st.write(resolved_question)

        retry_performed = metadata.get("retry_performed")

        if retry_performed is not None:
            st.write(f"Corrective retry performed: {retry_performed}")

        attempts = metadata.get("attempts", [])

        if attempts:
            st.markdown("**Retrieval attempts**")

            for index, attempt in enumerate(attempts, start=1):
                st.markdown(f"Attempt {index}")

                if isinstance(attempt, dict):
                    st.json(attempt, expanded=False)
                else:
                    st.write(attempt)

        if metadata.get("error"):
            st.warning("An error was reported by the pipeline.")
            st.code(str(metadata["error"]))


def add_message(role, content, **extra):
    """Store a chat message for display."""

    message = {
        "role": role,
        "content": content,
        **extra,
    }

    st.session_state.messages.append(message)


def reset_conversation():
    """Clear chat history and session memory."""

    st.session_state.memory.clear()
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.session_state.show_transcript = False
    st.session_state.show_state = False
    st.session_state.show_diagnostics = False


def process_question(question):
    """Run the RAG pipeline and store the response."""

    question = question.strip()

    if not question:
        return

    # Get the previous history before saving the current question.
    conversation_context = (
        st.session_state.memory.get_context(max_turns=5)
    )

    add_message("user", question)
    st.session_state.memory.add_turn("user", question)

    try:
        rag = get_rag()

        with st.spinner(
            "Searching documents, checking evidence, and composing an answer..."
        ):
            result = rag.answer(
                question,
                conversation_context=conversation_context,
            )

        answer = result.get(
            "answer",
            "The pipeline did not return an answer.",
        )

        sources = result.get("sources", [])
        error = result.get("error")

        metadata = {
            "sources": sources,
            "insufficient_evidence": result.get(
                "insufficient_evidence"
            ),
            "retry_performed": result.get("retry_performed"),
            "attempts": result.get("attempts", []),
            "resolved_question": result.get("resolved_question"),
            "error": str(error) if error else None,
        }

        add_message(
            "assistant",
            answer,
            sources=sources,
            metadata=metadata,
            error=str(error) if error else None,
            insufficient_evidence=result.get(
                "insufficient_evidence", False
            ),
        )

        st.session_state.memory.add_turn(
            "assistant",
            answer,
            metadata=metadata,
        )

    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"

        fallback_answer = (
            "I couldn't complete this request because an application "
            "or model-service error occurred. Please check the diagnostic "
            "details and try again when the service is available."
        )

        metadata = {
            "error": error_message,
            "sources": [],
        }

        add_message(
            "assistant",
            fallback_answer,
            sources=[],
            metadata=metadata,
            error=error_message,
            insufficient_evidence=False,
        )

        st.session_state.memory.add_turn(
            "assistant",
            fallback_answer,
            metadata=metadata,
        )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

document_count, chunk_count = get_knowledge_base_stats()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-mark">✳</span> Nexus RAG
        </div>
        <div class="sidebar-note">
            Intelligent answers. Verifiable evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Knowledge base")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Documents", document_count)

    with metric_col2:
        st.metric("Text chunks", chunk_count)

    st.caption(
        "Counts are read from your local data directory. "
        "They are not live retrieval-quality measurements."
    )

    st.divider()

    st.markdown("### Your session")

    assistant_messages = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "assistant"
    )

    user_messages = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "user"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Questions", user_messages)

    with col2:
        st.metric("Responses", assistant_messages)

    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()

    st.divider()

    st.markdown("### Developer tools")

    st.session_state.show_transcript = st.toggle(
        "Conversation transcript",
        value=st.session_state.show_transcript,
    )

    st.session_state.show_state = st.toggle(
        "Inspect session state",
        value=st.session_state.show_state,
    )

    st.session_state.show_diagnostics = st.toggle(
        "Show retrieval diagnostics",
        value=st.session_state.show_diagnostics,
    )

    transcript = st.session_state.memory.format_transcript()

    st.download_button(
        label="↓ Export conversation",
        data=transcript or "No conversation yet.",
        file_name=(
            "nexus_rag_transcript_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".txt"
        ),
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Pipeline capabilities")

    capabilities = [
        "Hybrid vector + BM25 retrieval",
        "Query transformation",
        "Cross-encoder reranking",
        "Evidence grading and corrective retry",
        "Grounded answers with citations",
        "Conversational memory",
    ]

    for capability in capabilities:
        st.markdown(
            f"<div class='sidebar-note'>✓ {capability}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="footer">
            Built with Streamlit, Sentence Transformers,
            BM25, and Gemini API.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# HERO / LANDING AREA
# ---------------------------------------------------------

st.markdown(
    """
    <div class="eyebrow">ADVANCED RETRIEVAL-AUGMENTED GENERATION</div>

    <div class="hero-title">
        Your knowledge,<br>
        <span class="hero-gradient">with evidence.</span>
    </div>

    <div class="hero-description">
        Ask questions across your document collection.
        Nexus retrieves relevant passages, reranks evidence,
        checks whether the context is sufficient, and generates
        answers grounded in your knowledge base.
    </div>

    <span class="status-pill">● Evidence-aware answers</span>
    <span class="tech-pill">Hybrid search</span>
    <span class="tech-pill">Reranking</span>
    <span class="tech-pill">Corrective RAG</span>
    <span class="tech-pill">Conversation memory</span>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# FEATURE CARDS
# ---------------------------------------------------------

st.markdown(
    "<div class='section-label'>Built for more reliable retrieval</div>",
    unsafe_allow_html=True,
)

feature_columns = st.columns(3, gap="medium")

features = [
    (
        "⌕",
        "Hybrid retrieval",
        "Combines semantic vector search with keyword-based BM25 "
        "retrieval, then fuses ranked results.",
    ),
    (
        "⤳",
        "Query intelligence",
        "Transforms complex questions and resolves follow-ups "
        "using conversation history.",
    ),
    (
        "◈",
        "Evidence reranking",
        "Uses a cross-encoder to prioritize passages that are "
        "more relevant to the question.",
    ),
    (
        "✓",
        "Corrective retrieval",
        "Grades retrieved evidence and retries retrieval when "
        "the available context is insufficient.",
    ),
    (
        "▤",
        "Grounded responses",
        "Returns source-aware answers and abstains when the "
        "documents do not support a reliable answer.",
    ),
    (
        "↗",
        "Inspectable memory",
        "Keeps conversational history available for follow-ups "
        "and provides transcript and state inspection.",
    ),
]

for index, feature in enumerate(features):
    with feature_columns[index % 3]:
        icon, title, description = feature

        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# SUGGESTED QUESTIONS
# ---------------------------------------------------------

st.markdown(
    "<div class='section-label'>Start exploring your documents</div>",
    unsafe_allow_html=True,
)

suggested_questions = [
    "How can SQL injection be prevented?",
    "What is the difference between vector search and BM25?",
    "What is the purpose of embeddings in RAG?",
    "What should a RAG system do when evidence is missing?",
]

suggestion_columns = st.columns(2, gap="small")

for index, suggested_question in enumerate(suggested_questions):
    with suggestion_columns[index % 2]:
        if st.button(
            f"↗  {suggested_question}",
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggested_question
            st.rerun()


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

st.markdown(
    "<div class='section-label'>Conversation</div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info(
        "Your workspace is ready. Ask a question above to search "
        "your documents and generate an evidence-grounded response."
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(message.get("sources", []))

            if message.get("insufficient_evidence"):
                st.info(
                    "The available documents did not provide enough "
                    "evidence for a reliable answer."
                )

            if message.get("error"):
                st.warning(
                    "This response was affected by an application "
                    "or model-service error."
                )

                with st.expander("Error details"):
                    st.code(message["error"])

            if st.session_state.show_diagnostics:
                display_diagnostics(message.get("metadata", {}))


# ---------------------------------------------------------
# TRANSCRIPT AND SESSION INSPECTOR
# ---------------------------------------------------------

if st.session_state.show_transcript:
    st.divider()
    st.subheader("Conversation transcript")
    st.text(st.session_state.memory.format_transcript())


if st.session_state.show_state:
    st.divider()
    st.subheader("Inspectable session state")

    session_state = st.session_state.memory.get_state()
    st.json(session_state, expanded=True)


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

chat_question = st.chat_input(
    "Ask a question about your knowledge base..."
)

question_to_process = (
    st.session_state.pending_question
    or chat_question
)

if question_to_process:
    st.session_state.pending_question = None

    process_question(question_to_process)

    # Render the saved response and updated conversation.
    st.rerun()


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        NEXUS RAG · Advanced Retrieval-Augmented Generation<br>
        Designed to make document search more contextual,
        transparent, and evidence-aware.
    </div>
    """,
    unsafe_allow_html=True,
)
