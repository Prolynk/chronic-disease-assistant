"""
App:     app.py (Hugging Face Spaces deployment)
Purpose: Streamlit demo of Strategy 1 dense retrieval
         RAG pipeline for chronic disease self-management.
         Deployed on Hugging Face Spaces, free public URL.
Project: MSc AI Dissertation, Habeeb Adekeye
         Northumbria University, KF7029
Note:    ChromaDB index is rebuilt from chunks_v2.jsonl
         on first startup then cached in /tmp/chroma_db/
"""

import streamlit as st
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -- Page config ---------------------------------------
st.set_page_config(
    page_title="Chronic Disease Self-Management Assistant",
    page_icon="🏥",
    layout="centered"
)

# -- Constants ------------------------------------------
CHROMA_PATH  = "/tmp/chroma_db"
CHUNKS_FILE  = "chunks_v2.jsonl"
COLLECTION   = "strategy1_dense"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
LLM_MODEL    = "gpt-4o-mini"
LLM_TEMP     = 0
LLM_MAX_TOK  = 500
TOP_K        = 5

SYSTEM_PROMPT = """You are a chronic disease
self-management assistant. Your role is to provide
accurate, evidence-based answers to questions about
diabetes, hypertension, and asthma management.

Answer ONLY based on the context provided below.
If the context does not contain enough information
to answer the question, say:
"I cannot find sufficient information in the
guidelines to answer this question."

Do not add information from outside the provided
context. Do not make up statistics or recommendations.
Be concise and specific. Cite the source document
when possible."""

# -- Load and cache everything --------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline():
    """
    Loads or builds the ChromaDB index from chunks_v2.jsonl.
    Uses /tmp/ for storage since HF Spaces has no
    persistent disk outside the repo.
    Downloads BAAI/bge-small-en-v1.5 from HuggingFace.
    """
    embedder = SentenceTransformer(EMBED_MODEL)
    db       = chromadb.PersistentClient(path=CHROMA_PATH)

    # Check if collection already exists in /tmp/
    existing = [c.name for c in db.list_collections()]

    if COLLECTION not in existing:
        st.info("Building knowledge base index, "
                "this takes about 2 minutes on first run...")
        collection = db.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        # Load chunks from jsonl
        chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        # Embed and add in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch  = chunks[i:i+batch_size]
            texts  = [c["text"] for c in batch]
            ids    = [str(c["chunk_id"]) for c in batch]
            metas  = [{
                "source":   c.get("source", ""),
                "disease":  c.get("disease", ""),
                "page":     str(c.get("page", "")),
                "chunk_id": str(c.get("chunk_id", "")),
            } for c in batch]

            embeddings = embedder.encode(
                texts,
                show_progress_bar=False
            ).tolist()

            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metas,
                embeddings=embeddings
            )
    else:
        collection = db.get_collection(COLLECTION)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm     = OpenAI(api_key=api_key)

    return collection, embedder, llm

def run_query(question, collection, embedder, llm):
    """Embed question, retrieve top-k chunks, generate answer."""
    embedding = embedder.encode([question]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        source = meta.get("source", "Unknown")
        page   = meta.get("page", "?")
        context_parts.append(
            f"[{i}] Source: {source}, Page {page}\n{doc}"
        )
    context = "\n\n".join(context_parts)

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOK,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": (f"Context:\n{context}"
                         f"\n\nQuestion: {question}")}
        ]
    )
    answer = response.choices[0].message.content
    return answer, docs, metas, distances

# -- UI --------------------------------------------------
st.title("🏥 Chronic Disease Self-Management Assistant")

st.markdown("""
**MSc AI Dissertation Demo, Strategy 1: Dense Retrieval**
*Habeeb Adekeye | KF7029 | Northumbria University*

This assistant answers questions about **diabetes,
hypertension, and asthma** self-management using
evidence retrieved directly from clinical guidelines:

| Guideline | Coverage |
|---|---|
| 📗 ADA Standards of Care 2026 | Diabetes (all types) |
| 📘 AHA/ACC Hypertension Guidelines 2025 | Hypertension |
| 📙 GINA Asthma Strategy Report 2026 | Asthma |
""")

st.divider()

# Example questions
st.markdown("**Try one of these example questions:**")
examples = [
    "What is the recommended HbA1c target for Type 2 diabetes?",
    "At what blood pressure level should medication be started?",
    "What reliever medication is used for acute asthma?",
    "What lifestyle changes help manage high blood pressure?",
    "What are the first-line medications for Type 2 diabetes?",
    "What foods should a person with diabetes avoid?",
    "How is asthma severity classified?",
    "What is the blood pressure target for most adults?",
]

cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(
            ex, key=f"ex_{i}",
            use_container_width=True):
        st.session_state["question"] = ex

st.divider()

question = st.text_area(
    "Or type your own question:",
    value=st.session_state.get("question", ""),
    height=80,
    placeholder="e.g. What are the symptoms of hypoglycemia?",
)

search_clicked = st.button(
    "🔍  Get Answer",
    type="primary",
    use_container_width=True
)

if search_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI API key not configured.")
    else:
        with st.spinner(
            "Searching guidelines and generating answer..."
        ):
            try:
                collection, embedder, llm = load_pipeline()
                answer, docs, metas, distances = run_query(
                    question, collection, embedder, llm
                )

                st.markdown("### 💬 Answer")
                st.markdown(answer)
                st.divider()

                st.markdown("### 📄 Retrieved Evidence")
                st.caption(
                    "The answer was grounded in these "
                    "guideline sections:"
                )
                for i, (doc, meta, dist) in enumerate(
                        zip(docs, metas, distances), 1):
                    source  = meta.get("source", "Unknown")
                    page    = meta.get("page", "?")
                    disease = meta.get("disease", "?")
                    score   = round(1 - float(dist), 3)
                    with st.expander(
                        f"📌 [{i}] {source}, "
                        f"Page {page}, "
                        f"Relevance: {score}",
                        expanded=(i == 1)
                    ):
                        st.caption(
                            f"Disease domain: **{disease}**"
                        )
                        st.markdown(doc)

            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()
st.caption(
    "⚠️ Research prototype, not for clinical use. "
    "Always consult a qualified healthcare professional. "
    "MSc AI Dissertation, Northumbria University, 2026."
)
