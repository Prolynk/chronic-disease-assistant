"""
App:     app.py (Streamlit Community Cloud deployment)
Purpose: Streamlit demo of Strategy 1 (dense retrieval) and
         Strategy 2 (KG-augmented retrieval) RAG pipelines for
         chronic disease self-management, selectable at runtime.
Project: MSc AI Dissertation, Habeeb Adekeye
         Northumbria University, KF7029
Note:    ChromaDB index is rebuilt from chunks_v2.jsonl
         on first startup then cached in /tmp/chroma_db/
"""

import os

# hf-xet's fast-download backend hangs indefinitely on Streamlit Cloud's
# sandboxed network instead of falling back to plain HTTP. Disable it
# before any huggingface_hub call triggers a model download.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import re
import streamlit as st
import json
import pandas as pd
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

SYSTEM_PROMPT_KG = """You are a chronic disease
self-management assistant. Your role is to provide
accurate, evidence-based answers to questions about
diabetes, hypertension, and asthma management.

You have two sources of grounding: (1) Guideline
passages retrieved from clinical guideline documents,
and (2) Structured knowledge graph facts extracted
from UMLS. Answer ONLY using these two sources. If
neither contains enough information to answer the
question, say: "I cannot find sufficient information
in the guidelines to answer this question."

Do not add information from outside the provided
sources. Do not make up statistics or recommendations.
Be concise and specific. Cite the guideline source
document when using a guideline passage, and note
when a statement draws on a knowledge graph fact."""

# -- Knowledge graph constants (Strategy 2) --------------
CONCEPTS_FILE = "umls_concepts.csv"
TRIPLES_FILE = "umls_triples_named_only.csv"

MAX_ENTITIES = 5
MAX_TRIPLES_PER_ENTITY = 5
MAX_TRIPLES_TOTAL = 15
MIN_STR_LEN = 4
EXCLUDE_CUIS = {"C1298908"}  # STR == "No" (SNOMEDCT_US Finding) - false-positive risk

RELATION_TIER_EXCLUDE = {
    "isa", "inverse isa", "has member", "member of",
    "mapped to", "mapped from", "other mapped to", "other mapped from",
    "subset includes concept", "concept in subset", "was a", "inverse was a",
    "has component", "component of", "has answer", "answer to",
}

RELATION_TIER_A = {
    "may treat", "may be treated by", "may prevent", "may be prevented by",
    "has contraindicated drug", "contraindicated with disease",
    "has contraindicated class", "contraindicated class of",
    "cause of", "due to", "has causative agent", "causative agent of",
    "has manifestation", "manifestation of",
    "has sign or symptom", "sign or symptom of",
    "has finding site", "finding site of",
    "disease has finding", "is finding of disease",
    "has associated finding", "associated finding of", "associated with",
    "has active ingredient", "active ingredient of",
    "disease has associated anatomic site", "disease has primary anatomic site",
}

# Curated colloquial-term -> seed-CUI aliases (see 10_strategy2_kg_augmented.py
# for why: the concept table's canonical name sometimes picked a formal
# synonym a real question would never use, e.g. Hypertension's canonical STR
# is "Hypertensive disease").
ALIAS_MAP = {
    "type 2 diabetes": "C0011860",
    "type ii diabetes": "C0011860",
    "type 1 diabetes": "C0011854",
    "type i diabetes": "C0011854",
    "gestational diabetes": "C0085207",
    "prediabetes": "C0362046",
    "pre-diabetes": "C0362046",
    "impaired glucose tolerance": "C0271650",
    "hypoglycemia": "C0020615",
    "hypoglycaemia": "C0020615",
    "diabetes mellitus": "C0011849",
    "diabetes": "C0011849",
    "hypertension": "C0020538",
    "high blood pressure": "C0020538",
    "asthma": "C0004096",
}

# -- Load and cache everything --------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline():
    """
    Loads or builds the ChromaDB index from chunks_v2.jsonl.
    Uses /tmp/ for storage since HF Spaces has no
    persistent disk outside the repo.
    Downloads BAAI/bge-small-en-v1.5 from HuggingFace.
    """
    print("[pipeline] loading embedder...", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    print("[pipeline] embedder loaded", flush=True)
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
        print(f"[pipeline] loaded {len(chunks)} chunks from {CHUNKS_FILE}",
              flush=True)

        # Embed and add in batches of 100
        batch_size = 100
        n_batches = (len(chunks) + batch_size - 1) // batch_size
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

            print(f"[pipeline] embedding batch {i // batch_size + 1}/"
                  f"{n_batches}...", flush=True)
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
        print("[pipeline] index build complete", flush=True)
    else:
        collection = db.get_collection(COLLECTION)
        print("[pipeline] reusing cached collection", flush=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    print(f"[pipeline] OPENAI_API_KEY present: {bool(api_key)}", flush=True)
    llm     = OpenAI(api_key=api_key)

    return collection, embedder, llm

@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_kg():
    """Load concept table + named triples, precompute matcher/ranking structures."""
    print("[kg] loading concepts + triples...", flush=True)
    concepts = pd.read_csv(CONCEPTS_FILE)
    concepts["STR"] = concepts["STR"].astype(str)
    concepts = concepts[
        (concepts["STR"].str.len() >= MIN_STR_LEN)
        & (~concepts["CUI"].isin(EXCLUDE_CUIS))
    ].copy()
    concepts["_len"] = concepts["STR"].str.len()
    concepts = concepts.sort_values(
        "_len", ascending=False, kind="stable").reset_index(drop=True)

    triples = pd.read_csv(TRIPLES_FILE)
    triples["relation_label"] = triples["relation_label"].fillna("")
    triples["linearised"] = triples["linearised"].fillna("")
    triples = triples.drop_duplicates(subset=["head_cui", "relation_label", "tail_cui"])
    rel_lower = triples["relation_label"].str.lower()
    keep_mask = (
        (~rel_lower.isin(RELATION_TIER_EXCLUDE))
        & (~rel_lower.str.contains("authorized value"))
    )
    triples = triples[keep_mask].copy()
    triples["_tier"] = rel_lower[keep_mask].apply(lambda r: 0 if r in RELATION_TIER_A else 1)
    print(f"[kg] loaded {len(concepts)} concepts, {len(triples)} triples", flush=True)

    return concepts, triples


def match_entities(question, concepts, max_entities=MAX_ENTITIES):
    """Span-aware, longest-match-first, word-boundary lexical matching.

    Checks the curated ALIAS_MAP first (colloquial disease terms -> seed
    CUI), then falls back to scanning the full concept table for anything
    else (drugs, symptoms, other concepts) not already covered.
    """
    matched = []
    consumed_spans = []

    def try_match(phrase, cui, display_name):
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        m = pattern.search(question)
        if not m:
            return False
        span = m.span()
        if any(s < span[1] and span[0] < e for s, e in consumed_spans):
            return False
        consumed_spans.append(span)
        matched.append({"cui": cui, "name": display_name})
        return True

    for phrase in sorted(ALIAS_MAP, key=len, reverse=True):
        if len(matched) >= max_entities:
            break
        cui = ALIAS_MAP[phrase]
        name_rows = concepts.loc[concepts["CUI"] == cui, "STR"]
        display_name = name_rows.iloc[0] if len(name_rows) else phrase
        try_match(phrase, cui, display_name)

    if len(matched) < max_entities:
        for _, row in concepts.iterrows():
            if len(matched) >= max_entities:
                break
            try_match(row["STR"], row["CUI"], row["STR"])

    return matched


def get_kg_facts(matched_entities, triples,
                  max_total=MAX_TRIPLES_TOTAL, max_per_entity=MAX_TRIPLES_PER_ENTITY):
    """Round-robin across matched entities' tier-ranked candidate facts, deduped."""
    per_entity = []
    for ent in matched_entities:
        cui = ent["cui"]
        cand = triples[(triples["head_cui"] == cui) | (triples["tail_cui"] == cui)]
        cand = cand.sort_values("_tier", kind="stable").head(max_per_entity)
        per_entity.append(list(cand["linearised"]))

    facts, seen = [], set()
    i = 0
    while len(facts) < max_total and any(i < len(lst) for lst in per_entity):
        for lst in per_entity:
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i])
                facts.append(lst[i])
                if len(facts) >= max_total:
                    break
        i += 1
    return facts[:max_total]


def run_query(question, collection, embedder, llm, concepts=None, triples=None):
    """Embed question, retrieve top-k chunks, optionally KG-augment, generate answer."""
    print(f"[query] embedding question: {question!r}", flush=True)
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

    is_kg = concepts is not None and triples is not None
    matched_entities, kg_facts = [], []
    if is_kg:
        print("[query] matching KG entities...", flush=True)
        matched_entities = match_entities(question, concepts)
        kg_facts = get_kg_facts(matched_entities, triples)

    if is_kg:
        kg_block = "\n".join(f"- {f}" for f in kg_facts) if kg_facts else "(no matched KG facts)"
        user_content = (
            f"Guideline passages:\n{context}\n\n"
            f"Structured knowledge graph facts:\n{kg_block}\n\n"
            f"Question: {question}"
        )
        system_prompt = SYSTEM_PROMPT_KG
    else:
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
        system_prompt = SYSTEM_PROMPT

    print("[query] calling OpenAI...", flush=True)
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOK,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    print("[query] got OpenAI response", flush=True)
    answer = response.choices[0].message.content
    return answer, docs, metas, distances, matched_entities, kg_facts

# -- UI --------------------------------------------------
st.title("🏥 Chronic Disease Self-Management Assistant")

st.markdown("""
**MSc AI Dissertation Demo**
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

strategy = st.radio(
    "Retrieval strategy:",
    ["Strategy 1: Dense Retrieval", "Strategy 2: KG-Augmented Retrieval"],
    horizontal=True,
    help=(
        "Strategy 1 retrieves passages from the guideline PDFs only. "
        "Strategy 2 adds structured facts (drug treatments, "
        "contraindications, etc.) from a UMLS knowledge graph."
    ),
)
is_strategy2 = strategy.startswith("Strategy 2")

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
                concepts, triples = load_kg() if is_strategy2 else (None, None)
                answer, docs, metas, distances, matched_entities, kg_facts = run_query(
                    question, collection, embedder, llm, concepts, triples
                )

                st.markdown("### 💬 Answer")
                st.markdown(answer)
                st.divider()

                if is_strategy2:
                    st.markdown("### 🧬 Knowledge Graph Facts Used")
                    if matched_entities:
                        st.caption(
                            "Matched entities: "
                            + ", ".join(e["name"] for e in matched_entities)
                        )
                    if kg_facts:
                        for f in kg_facts:
                            st.markdown(f"- {f}")
                    else:
                        st.caption("No matching knowledge graph facts found for this question.")
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
