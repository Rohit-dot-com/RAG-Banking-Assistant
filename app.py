import os
import hashlib
import tempfile
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"   # same name that worked in test_llm.py

st.set_page_config(page_title="Banking Document Assistant", layout="wide")

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant that answers questions about a document.\n"
     "Rules:\n"
     "1. Use ONLY the context below. Do not use outside knowledge.\n"
     "2. If the context does not contain the answer, reply exactly: "
     "\"I could not find this in the document.\"\n"
     "3. Be concise and cite the page numbers you used, like (p. 12).\n"
     "4. If the context describes other researchers' work (for example a literature "
     "review), say so. Never present other studies' methods or findings as this "
     "document's own.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


def as_text(content):
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def build_index(uploaded_file):
    data = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    if not chunks:
        return None, len(pages), 0

    for c in chunks:
        c.metadata = {
            "source": uploaded_file.name,
            "page": c.metadata.get("page_label", str(c.metadata.get("page", 0) + 1)),
        }

    collection = "doc_" + hashlib.md5(data).hexdigest()[:16]
    store = Chroma.from_documents(chunks, get_embeddings(), collection_name=collection)
    return store, len(pages), len(chunks)


def ask(store, question, k):
    results = store.similarity_search_with_score(question, k=k)
    context = "\n\n".join(f"[Page {d.metadata['page']}] {d.page_content}" for d, _ in results)
    messages = PROMPT.invoke({"context": context, "question": question})
    response = get_llm().invoke(messages)
    sources = [
        f"Page {d.metadata['page']} (distance {score:.2f}): {d.page_content[:300]}..."
        for d, score in results
    ]
    return as_text(response.content), sources


# ---------- Session state ----------
for key, default in [("store", None), ("doc_id", None), ("messages", []), ("stats", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    k = st.slider("Chunks to retrieve (top-k)", 2, 10, 8)

    if uploaded:
        doc_id = f"{uploaded.name}-{uploaded.size}"
        if doc_id != st.session_state.doc_id:
            with st.spinner("Reading and indexing the document..."):
                store, n_pages, n_chunks = build_index(uploaded)
            st.session_state.store = store
            st.session_state.doc_id = doc_id
            st.session_state.messages = []
            st.session_state.stats = (n_pages, n_chunks)

    if st.session_state.stats:
        n_pages, n_chunks = st.session_state.stats
        if n_chunks == 0:
            st.error("No text found. This may be a scanned PDF.")
        else:
            st.success(f"{n_pages} pages, {n_chunks} chunks indexed")

# ---------- Main page ----------
st.title("Banking Document Assistant")
st.caption("Ask questions about a PDF. Answers come only from the document, with page citations.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(s)

question = st.chat_input(
    "Ask a question about the document",
    disabled=st.session_state.store is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(st.session_state.store, question, k)
            except Exception as e:
                answer, sources = f"Something went wrong: {e}", []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )