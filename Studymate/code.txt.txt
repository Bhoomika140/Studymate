import os
import io
from typing import List, Dict, Any

import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

# Transformers pipeline for IBM Granite fallback
from transformers import pipeline

# -----------------------------
# Config & Page
# -----------------------------
st.set_page_config(page_title="StudyMate · AI PDF Q&A", page_icon="📚", layout="wide")

st.markdown(
    """
    <div style="background:linear-gradient(90deg,#5b86e5,#36d1dc);padding:18px;border-radius:18px;margin-bottom:12px;">
      <h1 style="margin:0;color:white;font-weight:800;">StudyMate 📚</h1>
      <p style="margin:4px 0 0;color:rgba(255,255,255,0.95);font-size:16px;">
        Conversational assistant for textbooks, notes, and research papers—powered by semantic search + Hugging Face / Watsonx / Granite.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar: Navigation
# -----------------------------
with st.sidebar:
    st.header("🧭 Navigation")
    page = st.radio("Go to", ["Home", "Upload PDFs", "Ask Questions"])

# -----------------------------
# Session State
# -----------------------------
if "docs" not in st.session_state:
    st.session_state.docs: List[Dict[str, Any]] = []
if "chunks" not in st.session_state:
    st.session_state.chunks: List[Dict[str, Any]] = []
if "vector" not in st.session_state:
    st.session_state.vector = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

# -----------------------------
# Helper: PDF text extraction
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    docs = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                docs.append({"text": text, "meta": {"source": filename, "page": i}})
    return docs

# -----------------------------
# Helper: Chunking
# -----------------------------
def chunk_docs(docs: List[Dict[str, Any]], chunk_size=1000, chunk_overlap=150) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for d in docs:
        parts = splitter.split_text(d["text"])
        for idx, part in enumerate(parts):
            meta = dict(d["meta"])
            meta["chunk"] = idx
            chunks.append({"text": part, "meta": meta})
    return chunks

# -----------------------------
# Helper: Build FAISS vector
# -----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector(chunks: List[Dict[str, Any]]):
    texts = [c["text"] for c in chunks]
    metadatas = [c["meta"] for c in chunks]
    embeddings = get_embeddings()
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# -----------------------------
# Helper: LLM
# -----------------------------
# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# watsonx_token = os.getenv("WATSONX_API_KEY")

# # Granite fallback pipeline
# granite_pipe = pipeline("text-generation", model="ibm-granite/granite-3.2-2b-instruct")

def call_llm(prompt: str, temperature=0.1, max_new_tokens=512) -> str:
    try:
        if hf_token:
            llm = HuggingFaceHub(
                repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens},
            )
            return llm(prompt).strip()
        elif watsonx_token:
            from ibm_watsonx_ai.foundation_models import Model
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
            from ibm_watsonx_ai import Credentials

            model = Model(
                model_id="mistralai/mixtral-8x7b-instruct-v0.1",
                params={
                    GenParams.DECODING_METHOD: "greedy",
                    GenParams.MAX_NEW_TOKENS: max_new_tokens,
                    GenParams.TEMPERATURE: temperature,
                    GenParams.STOP_SEQUENCES: [],
                },
                credentials=Credentials(api_key=watsonx_token, url="https://us-south.ml.cloud.ibm.com"),
                project_id=os.getenv("WATSONX_PROJECT_ID", "")
            )
            res = model.generate(prompt=prompt)
            return res["results"][0]["generated_text"].strip()
        else:
            # Use Granite pipeline fallback
            res = granite_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
            return res[0]["generated_text"].strip()
    except Exception as e:
        return f"Error generating LLM response: {e}"

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.subheader("Why StudyMate?")
    st.markdown(
        """
        - 🔎 Ask questions about your textbooks, notes, or papers.
        - 🧠 Semantic search finds the right passage fast.
        - 📑 Grounded answers with citations.
        - 🖥 Works locally with Hugging Face embeddings & LLM (or Granite fallback).
        """
    )

# -----------------------------
# UPLOAD PDFs
# -----------------------------
elif page == "Upload PDFs":
    st.subheader("Upload one or more PDFs")
    files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True)
    build_btn = st.button("Process & Build Index")

    if files:
        st.session_state.upload_cache = files
        st.info(f"Selected {len(files)} file(s). Click *Process & Build Index*.")

    if build_btn:
        if not files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        all_docs = []
        with st.spinner("Extracting text from PDFs..."):
            for f in files:
                all_docs.extend(extract_text_from_pdf(f.read(), f.name))
        st.session_state.docs = all_docs

        with st.spinner("Splitting into semantic chunks..."):
            chunks = chunk_docs(all_docs, chunk_size=1100, chunk_overlap=150)
            st.session_state.chunks = chunks

        with st.spinner("Building vector store..."):
            st.session_state.vector = build_vector(chunks)
        st.session_state.ready = True
        st.success(f"Indexed {len(chunks)} chunks from {len(all_docs)} pages.")

# -----------------------------
# ASK QUESTIONS
# -----------------------------
elif page == "Ask Questions":
    st.subheader("Ask a question about your PDFs")

    if not st.session_state.ready:
        st.warning("Please upload and process PDFs first.")
        st.stop()

    query = st.text_input("Your question")
    k = st.slider("Top-k passages to retrieve", 3, 10, 5)

    if st.button("Ask") and query.strip():
        retriever = st.session_state.vector.as_retriever(search_kwargs={"k": k})
        rel_docs = retriever.get_relevant_documents(query)

        # Build prompt
        context = "\n".join([f"[{i+1}] {d.page_content}" for i,d in enumerate(rel_docs)])
        prompt = f"Answer the question strictly using the context below:\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"

        # Generate answer
        answer = call_llm(prompt)
        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("📚 Sources"):
            for i, d in enumerate(rel_docs, start=1):
                meta = d.metadata
                st.markdown(f"[{i}]** {meta.get('source','document')} — page {meta.get('page','?')}")
                st.caption(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": answer})

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🧾 Conversation History")
        for h in st.session_state.history[-10:]:
            if h["role"] == "user":
                st.markdown(f"*You:* {h['content']}")
            else:
                st.markdown(f"*StudyMate 🤖:* {h['content']}")
