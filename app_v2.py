"""Jerusalem RAG Explorer v2 - Multilingual Medieval Manuscripts.

Enhanced Streamlit UI with:
- Language filtering
- Source language indicators
- Original text display toggle
- Comparative source mode
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Check for API key early
if "GEMINI_API_KEY" not in os.environ:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

from rag.retrieve_v2 import Retriever, format_context, format_sources_summary
from rag.prompts_v2 import build, get_available_modes, get_mode_description
from rag.gemini_client import ask_gemini
from rag.models import get_language_flag, get_language_name, LANGUAGE_NAMES

# Page config
st.set_page_config(
    page_title="Jerusalem RAG v2",
    page_icon="🏰",
    layout="wide",
)

st.title("🏰 Kingdom of Jerusalem RAG Explorer v2")
st.caption("Multilingual medieval manuscript research powered by AI")

# Initialize retriever
@st.cache_resource
def get_retriever():
    try:
        return Retriever(index_dir="data/index_v2")
    except FileNotFoundError:
        return None

retriever = get_retriever()

if retriever is None:
    st.warning(
        "**v2 index not found.** Please run the ingestion pipeline first:\n\n"
        "```bash\n"
        "python -m rag.gallica_fetch  # Fetch Gallica manuscripts\n"
        "python -m rag.injest_v2      # Build v2 index\n"
        "```\n\n"
        "Or use the original app.py with the v1 index."
    )
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")

    # Response mode
    mode = st.selectbox(
        "Response Mode",
        get_available_modes(),
        format_func=lambda x: f"{x.replace('_', ' ').title()} - {get_mode_description(x)}",
    )

    # Number of sources
    top_k = st.slider("Sources to retrieve", min_value=2, max_value=12, value=6)

    st.divider()

    # Language filters
    st.subheader("🌐 Language Filters")
    st.caption("Select which source languages to include")

    col1, col2 = st.columns(2)
    with col1:
        include_en = st.checkbox("🇬🇧 English", value=True)
        include_la = st.checkbox("🇻🇦 Latin", value=True)
    with col2:
        include_ar = st.checkbox("🇸🇦 Arabic", value=True)
        include_el = st.checkbox("🇬🇷 Greek", value=True)

    # Build language filter list
    languages = []
    if include_en:
        languages.append("en")
    if include_la:
        languages.append("la")
    if include_ar:
        languages.append("ar")
    if include_el:
        languages.append("el")

    if not languages:
        st.warning("Select at least one language")
        languages = ["en"]

    st.divider()

    # Display options
    st.subheader("📖 Display Options")
    show_original = st.checkbox(
        "Show original text",
        value=False,
        help="Display original language text alongside translations",
    )

# Main query interface
st.markdown("---")

question = st.text_area(
    "Ask a question about the Crusades and Kingdom of Jerusalem",
    placeholder="e.g., What happened at the Battle of Hattin? Who was Baldwin IV?",
    height=100,
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if search_btn and question.strip():
    # Retrieve relevant chunks
    with st.spinner("Searching multilingual corpus..."):
        results = retriever.retrieve(
            question,
            top_k=top_k,
            languages=languages if len(languages) < 4 else None,
        )

    if not results:
        st.warning("No relevant sources found. Try adjusting your language filters or query.")
        st.stop()

    # Format context for LLM
    context = format_context(results, include_original=show_original)
    prompt = build(mode, question, context)

    # Generate response
    with st.spinner("Generating scholarly response..."):
        answer = ask_gemini(prompt)

    # Display answer
    st.markdown("## Answer")
    st.markdown(answer)

    st.markdown("---")

    # Display sources
    st.markdown("## 📚 Retrieved Sources")

    sources = format_sources_summary(results)

    for i, source in enumerate(sources):
        with st.container():
            # Header row with badges
            cols = st.columns([4, 1, 1, 1])

            with cols[0]:
                st.markdown(f"**{source['chunk_id']}**")
            with cols[1]:
                flag = source['flag']
                lang_name = source['language_name']
                st.caption(f"{flag} {lang_name}")
            with cols[2]:
                if source['is_translation']:
                    orig = get_language_name(source['original_language'])
                    st.caption(f"🔄 From {orig}")
                else:
                    st.caption("📜 Original")
            with cols[3]:
                st.caption(f"Score: {source['score']:.3f}")

            # Metadata
            meta_parts = []
            if source['author']:
                meta_parts.append(f"*{source['author']}*")
            if source['title']:
                meta_parts.append(source['title'])
            meta_parts.append(f"Source: {source['source_repository']}")

            st.caption(" | ".join(meta_parts))

            # Preview
            st.info(source['preview'])

            st.markdown("---")

    # Debug: show full context
    with st.expander("🔧 Show full context sent to LLM"):
        st.text(context)

    with st.expander("🔧 Show full prompt"):
        st.text(prompt)

# Footer
st.markdown("---")
st.caption(
    "Jerusalem RAG Explorer v2 | "
    "Sources: Archive.org, Wikipedia, Gallica (BnF) | "
    "Powered by Google Gemini"
)
