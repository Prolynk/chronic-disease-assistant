"""
App:     app.py (Streamlit Community Cloud deployment)
Purpose: Streamlit demo of Strategy 1 (dense retrieval) and
         Strategy 2 (KG-only retrieval) RAG pipelines for chronic
         disease self-management, selectable at runtime.
Project: MSc AI Dissertation, Habeeb Adekeye
         Northumbria University, KF7029
Note:    ChromaDB index is rebuilt from chunks_v2.jsonl on first
         startup then cached in /tmp/chroma_db/. Strategy 2 uses
         NO PDF text at all - it is grounded solely in UMLS
         knowledge graph triples, per the project approval form.
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
import networkx as nx
import spacy
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
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
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

# -- Knowledge graph constants (Strategy 2, KG-ONLY) -----
CONCEPTS_FILE = "umls_concepts.csv"
TRIPLES_FILE = "umls_triples.csv"
NER_MODEL = "en_core_sci_sm"

MAX_ENTITIES = 5
MAX_TRIPLES_PER_ENTITY = 5
MAX_TRIPLES_TOTAL = 15
MIN_STR_LEN = 4
EXCLUDE_CUIS = {"C1298908"}  # STR == "No" (SNOMEDCT_US Finding) - false-positive risk

# Taxonomic/structural/mapping relations with near-zero clinical QA value,
# plus LOINC survey-form artifacts and the generic catch-all "related to".
RELATION_TIER_EXCLUDE = {
    "isa", "inverse isa",
    "has member", "member of",
    "is parent of", "has subtype",
    "is broader than", "is narrower than",
    "part of", "has part",
    "mapped to", "mapped from", "other mapped to", "other mapped from",
    "subset includes concept", "concept in subset", "was a", "inverse was a",
    "has component", "component of", "has answer", "answer to",
    "related to",
}

# Clinically salient relations, preferred over generic filler facts.
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

# Curated colloquial-term -> seed-CUI aliases: the concept table's canonical
# name sometimes picked a formal synonym a real question would never use,
# e.g. the Hypertension seed CUI's canonical STR is "Hypertensive disease".
# Also acts as a safety net for scispaCy's lightweight NER model, which has
# a verified recall gap (it drops "asthma" from "acute asthma symptoms").
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

SYSTEM_PROMPT_KG = """You are a chronic disease self-management assistant answering using ONLY structured knowledge graph facts extracted from UMLS (not clinical guideline text). Facts take the form of (head, relation, tail) statements, e.g. "Metformin may treat Type 2 Diabetes."

If one or more facts name a relevant drug, contraindication, cause, or finding, state it plainly and attribute it to the knowledge graph (e.g. "According to the knowledge graph, Albuterol may treat Asthma."). Do NOT claim guideline-level status ("first-line", specific dosing, numeric thresholds) unless a fact explicitly states it - these facts describe relationships, not clinical protocols. Only say "I cannot find sufficient information in the knowledge graph to answer this question" if NONE of the facts are relevant at all.

Be concise. Do not invent information beyond what the facts state."""

# -- Graph-linearised constants (Strategy 3) -------------
STRATEGY3_COLLECTION = "strategy3_linearised"

SYSTEM_PROMPT_S3 = """You are a chronic disease self-management assistant answering using ONLY structured knowledge graph facts extracted from UMLS (not clinical guideline text). Facts take the form of (head, relation, tail) statements, e.g. "Metformin may treat Type 2 Diabetes." They were retrieved by semantic search over the knowledge graph, not by looking up a specific entity.

If one or more facts name a relevant drug, contraindication, cause, or finding, state it plainly and attribute it to the knowledge graph (e.g. "According to the knowledge graph, Albuterol may treat Asthma."). Do NOT claim guideline-level status ("first-line", specific dosing, numeric thresholds) unless a fact explicitly states it - these facts describe relationships, not clinical protocols. Only say "I cannot find sufficient information in the knowledge graph to answer this question" if NONE of the retrieved facts are relevant at all.

Because facts are retrieved by semantic similarity rather than precise entity lookup, they may occasionally include two facts that conflict (e.g. one says a drug may treat a condition while another says the same drug is contraindicated for it). If you notice such a conflict among the retrieved facts, explicitly say the knowledge graph contains conflicting information about that point, rather than silently picking one side.

Be concise. Do not invent information beyond what the facts state."""

# -- Load and cache everything --------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    """Shared BGE embedder, cached once and reused by Strategy 1 and 3."""
    print("[embedder] loading...", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    print("[embedder] loaded", flush=True)
    return embedder


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_pipeline():
    """
    Loads or builds the ChromaDB index from chunks_v2.jsonl.
    Uses /tmp/ for storage since HF Spaces has no
    persistent disk outside the repo.
    Downloads BAAI/bge-small-en-v1.5 from HuggingFace.
    """
    embedder = load_embedder()
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


@st.cache_resource(show_spinner=False)
def load_llm_client():
    """Lightweight OpenAI client loader - does NOT touch the ChromaDB
    index or embedder, so Strategy 2 never pays for Strategy 1's setup."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner="Loading scispaCy model...")
def load_ner_model():
    print("[kg] loading scispaCy model...", flush=True)
    nlp = spacy.load(NER_MODEL)
    print("[kg] scispaCy model loaded", flush=True)
    return nlp


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_kg():
    """Load the concept table + full triples file, build a NetworkX graph.
    KG-ONLY strategy: umls_triples.csv is the sole knowledge source."""
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

    graph = nx.MultiDiGraph()
    for _, row in triples.iterrows():
        graph.add_edge(
            row["head_cui"], row["tail_cui"],
            relation_label=row["relation_label"],
            linearised=row["linearised"],
            tier=int(row["_tier"]),
        )
    print(f"[kg] loaded {len(concepts)} concepts, {len(triples)} triples -> "
          f"graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges",
          flush=True)

    return concepts, graph


def build_s3_corpus():
    """Clean umls_triples.csv the same way as load_kg() (dedupe + relation-
    tier-exclude) so both KG strategies share the identical fact base -
    they differ ONLY in retrieval mechanism (graph traversal vs. dense
    embedding search over these same linearised sentences)."""
    triples = pd.read_csv(TRIPLES_FILE)
    triples["relation_label"] = triples["relation_label"].fillna("")
    triples["linearised"] = triples["linearised"].fillna("")
    triples = triples.drop_duplicates(subset=["head_cui", "relation_label", "tail_cui"])
    rel_lower = triples["relation_label"].str.lower()
    keep_mask = (
        (~rel_lower.isin(RELATION_TIER_EXCLUDE))
        & (~rel_lower.str.contains("authorized value"))
        & (triples["linearised"].str.len() > 0)
    )
    triples = triples[keep_mask].copy().reset_index(drop=True)
    triples["triple_id"] = triples.index.astype(str)
    return triples


@st.cache_resource(show_spinner="Building graph-linearised knowledge base...")
def load_s3_index(_embed_model):
    """Build (or reuse) the strategy3_linearised ChromaDB collection,
    embedding each cleaned linearised KG sentence as its own document."""
    print("[s3] building corpus...", flush=True)
    corpus = build_s3_corpus()
    print(f"[s3] corpus size: {len(corpus)}", flush=True)

    db = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in db.list_collections()]

    if STRATEGY3_COLLECTION in existing:
        collection = db.get_collection(STRATEGY3_COLLECTION)
        if collection.count() == len(corpus):
            print("[s3] reusing cached collection", flush=True)
            return collection
        db.delete_collection(STRATEGY3_COLLECTION)

    st.info("Building graph-linearised knowledge base, "
            "this takes about a minute on first run...")
    collection = db.create_collection(
        name=STRATEGY3_COLLECTION, metadata={"hnsw:space": "cosine"})

    batch_size = 500
    n = len(corpus)
    for i in range(0, n, batch_size):
        batch = corpus.iloc[i:i + batch_size]
        texts = batch["linearised"].tolist()
        ids = batch["triple_id"].tolist()
        metas = [{
            "head": str(r["head"]), "relation_label": str(r["relation_label"]),
            "tail": str(r["tail"]), "sab": str(r["sab"]),
        } for _, r in batch.iterrows()]

        print(f"[s3] embedding {min(i + batch_size, n)}/{n}...", flush=True)
        embeddings = _embed_model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        collection.add(documents=texts, ids=ids, metadatas=metas, embeddings=embeddings)

    print("[s3] index build complete", flush=True)
    return collection


def extract_entities(question, nlp):
    """scispaCy biomedical NER - returns (text, start_char, end_char) spans."""
    doc = nlp(question)
    return [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents]


def map_to_cuis(question, entity_spans, concepts, max_entities=MAX_ENTITIES):
    """Map scispaCy mentions (+ curated disease-alias safety net) to CUIs
    in the project's own filtered concept table.

    Span-aware: once a character range in the question is claimed by a
    match, nothing overlapping it can match again - otherwise "type 2
    diabetes" and the plain "diabetes" alias fire independently and the
    noisy generic "Diabetes" CUI the alias map was built to avoid sneaks
    back in via the second pass.
    """
    matched = []
    seen_cuis = set()
    consumed_spans = []

    def span_free(s, e):
        return not any(cs < e and s < ce for cs, ce in consumed_spans)

    for phrase in sorted(ALIAS_MAP, key=len, reverse=True):
        if len(matched) >= max_entities:
            break
        cui = ALIAS_MAP[phrase]
        if cui in seen_cuis:
            continue
        m = re.search(r"\b" + re.escape(phrase) + r"\b", question, re.IGNORECASE)
        if not m or not span_free(*m.span()):
            continue
        consumed_spans.append(m.span())
        name_rows = concepts.loc[concepts["CUI"] == cui, "STR"]
        name = name_rows.iloc[0] if len(name_rows) else phrase
        matched.append({"cui": cui, "name": name})
        seen_cuis.add(cui)

    for text, start, end in entity_spans:
        if len(matched) >= max_entities:
            break
        if not span_free(start, end):
            continue
        span_lower = text.lower().strip()
        if not span_lower:
            continue
        exact = concepts[concepts["STR"].str.lower() == span_lower]
        if len(exact) and exact.iloc[0]["CUI"] not in seen_cuis:
            consumed_spans.append((start, end))
            matched.append({"cui": exact.iloc[0]["CUI"], "name": exact.iloc[0]["STR"]})
            seen_cuis.add(exact.iloc[0]["CUI"])
            continue
        for _, row in concepts.iterrows():
            if row["CUI"] in seen_cuis:
                continue
            if re.search(r"\b" + re.escape(row["STR"]) + r"\b", text, re.IGNORECASE):
                consumed_spans.append((start, end))
                matched.append({"cui": row["CUI"], "name": row["STR"]})
                seen_cuis.add(row["CUI"])
                break

    return matched[:max_entities]


def traverse_graph(matched_entities, graph,
                    max_total=MAX_TRIPLES_TOTAL, max_per_entity=MAX_TRIPLES_PER_ENTITY):
    """1-hop NetworkX traversal from each matched CUI (both edge directions),
    tier-ranked, round-robin across entities, deduped."""
    per_entity = []
    for ent in matched_entities:
        cui = ent["cui"]
        if cui not in graph:
            per_entity.append([])
            continue
        edges = []
        for _, _tail, data in graph.out_edges(cui, data=True):
            edges.append((data["tier"], data["linearised"]))
        for _head, _, data in graph.in_edges(cui, data=True):
            edges.append((data["tier"], data["linearised"]))
        edges.sort(key=lambda e: e[0])  # Python sort is stable
        per_entity.append([e[1] for e in edges[:max_per_entity]])

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


def run_query_dense(question, collection, embedder, llm):
    """Strategy 1: embed question, retrieve top-k guideline chunks, generate."""
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

    print("[query] calling OpenAI...", flush=True)
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOK,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    print("[query] got OpenAI response", flush=True)
    answer = response.choices[0].message.content
    return answer, docs, metas, distances


def run_query_kg(question, nlp, concepts, graph, llm):
    """Strategy 2: extract entities -> map to CUIs -> traverse graph ->
    generate, grounded SOLELY in KG triples. Zero PDF text anywhere."""
    print(f"[kg-query] extracting entities from: {question!r}", flush=True)
    entity_spans = extract_entities(question, nlp)
    matched_entities = map_to_cuis(question, entity_spans, concepts)
    kg_facts = traverse_graph(matched_entities, graph)
    entity_mentions = [text for text, _s, _e in entity_spans]

    kg_block = "\n".join(f"- {f}" for f in kg_facts) if kg_facts else "(no matched KG facts)"
    user_content = f"Knowledge graph facts:\n{kg_block}\n\nQuestion: {question}"

    print("[kg-query] calling OpenAI...", flush=True)
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOK,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_KG},
            {"role": "user", "content": user_content}
        ]
    )
    print("[kg-query] got OpenAI response", flush=True)
    answer = response.choices[0].message.content
    return answer, matched_entities, kg_facts, entity_mentions


def run_query_s3(question, embed_model, collection, llm):
    """Strategy 3: dense-retrieve top-k linearised KG sentences (semantic
    search, NOT entity-linking/traversal), generate grounded answer."""
    print(f"[s3-query] embedding question: {question!r}", flush=True)
    qe = embed_model.encode(
        [QUERY_PREFIX + question], normalize_embeddings=True)[0].tolist()
    res = collection.query(
        query_embeddings=[qe], n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    distances = res["distances"][0]
    scores = [1 - d for d in distances]

    kg_block = "\n".join(f"- {d}" for d in docs) if docs else "(no matched KG facts)"
    user_content = f"Knowledge graph facts:\n{kg_block}\n\nQuestion: {question}"

    print("[s3-query] calling OpenAI...", flush=True)
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        max_tokens=LLM_MAX_TOK,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_S3},
            {"role": "user", "content": user_content}
        ]
    )
    print("[s3-query] got OpenAI response", flush=True)
    answer = response.choices[0].message.content
    return answer, docs, metas, scores

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
    [
        "Strategy 1: Dense Retrieval",
        "Strategy 2: KG-Only Retrieval (Graph Traversal)",
        "Strategy 3: KG-Only Retrieval (Dense/Linearised)",
    ],
    horizontal=True,
    help=(
        "Strategy 1 retrieves passages from the guideline PDFs and "
        "answers from that text alone. Strategies 2 and 3 use ZERO "
        "guideline text - both answer using ONLY structured UMLS "
        "knowledge graph facts, but via different mechanisms: "
        "Strategy 2 extracts entities with scispaCy and symbolically "
        "traverses the graph from them; Strategy 3 embeds every KG "
        "fact and retrieves by semantic similarity to your question, "
        "the same mechanism Strategy 1 uses over PDF text. Both KG "
        "strategies will often correctly decline to answer questions "
        "that need clinical protocol detail (numeric targets, "
        "'first-line' status, lifestyle advice) that isn't represented "
        "as a graph relationship - that's an expected limitation, not "
        "a bug."
    ),
)
is_strategy2 = strategy.startswith("Strategy 2")
is_strategy3 = strategy.startswith("Strategy 3")

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
        if is_strategy2:
            spinner_text = "Extracting entities and traversing knowledge graph..."
        elif is_strategy3:
            spinner_text = "Searching knowledge graph facts and generating answer..."
        else:
            spinner_text = "Searching guidelines and generating answer..."

        with st.spinner(spinner_text):
            try:
                if is_strategy2:
                    nlp = load_ner_model()
                    concepts, graph = load_kg()
                    llm = load_llm_client()
                    answer, matched_entities, kg_facts, entity_mentions = run_query_kg(
                        question, nlp, concepts, graph, llm
                    )

                    st.markdown("### 💬 Answer")
                    st.markdown(answer)
                    st.caption(
                        "⚙️ Grounded solely in UMLS knowledge graph triples "
                        "(symbolic entity-linking + graph traversal). "
                        "No guideline PDF text was used for this answer."
                    )
                    st.divider()

                    st.markdown("### 🧬 Knowledge Graph Facts Used")
                    st.caption(
                        "scispaCy-extracted mentions: "
                        + (", ".join(entity_mentions) or "(none)")
                    )
                    if matched_entities:
                        st.caption(
                            "Matched UMLS entities: "
                            + ", ".join(f"{e['name']} ({e['cui']})" for e in matched_entities)
                        )
                    if kg_facts:
                        for f in kg_facts:
                            st.markdown(f"- {f}")
                    else:
                        st.caption("No matching knowledge graph facts found for this question.")

                elif is_strategy3:
                    embed_model = load_embedder()
                    collection = load_s3_index(embed_model)
                    llm = load_llm_client()
                    answer, docs, metas, scores = run_query_s3(
                        question, embed_model, collection, llm
                    )

                    st.markdown("### 💬 Answer")
                    st.markdown(answer)
                    st.caption(
                        "⚙️ Grounded solely in UMLS knowledge graph triples "
                        "(dense/semantic search over linearised facts). "
                        "No guideline PDF text was used for this answer."
                    )
                    st.divider()

                    st.markdown("### 🧬 Retrieved Knowledge Graph Facts")
                    st.caption(
                        "Retrieved by semantic similarity to your question, "
                        "not by looking up a specific entity:"
                    )
                    for i, (doc, meta, score) in enumerate(zip(docs, metas, scores), 1):
                        with st.expander(
                            f"📌 [{i}] {meta.get('sab', 'UMLS')}, "
                            f"Relevance: {round(score, 3)}",
                            expanded=(i == 1)
                        ):
                            st.markdown(doc)

                else:
                    collection, embedder, llm = load_pipeline()
                    answer, docs, metas, distances = run_query_dense(
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
