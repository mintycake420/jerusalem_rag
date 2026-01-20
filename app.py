from dotenv import load_dotenv
import streamlit as st

from rag.retrieve import retrieve, format_context
from rag.prompts import build
from rag.gemini_client import ask_gemini

load_dotenv()

st.title("🏰 Kingdom of Jerusalem RAG Explorer")

mode = st.selectbox("Mode", ["default", "chronology", "dossier", "claim_check", "retrieval"])
top_k = st.slider("Top-k chunks", 2, 8, 4)

q = st.text_input("Ask a question")

if st.button("Answer") and q.strip():
    results = retrieve(q, top_k=top_k)
    context = format_context(results)
    prompt = build(mode, q, context)

    ans = ask_gemini(prompt)

    st.subheader("Answer")
    st.write(ans)

    st.subheader("Retrieved Sources")
    for score, c in results:
        st.markdown(f"**{c['chunk_id']}** ({score:.3f})")
        st.caption(c["text"][:250] + "…")

    with st.expander("Show full retrieved context"):
        st.text(context)
# To run this app, use the command:
# streamlit run app.py