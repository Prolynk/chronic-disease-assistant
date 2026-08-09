"""
Script:  11_strategy3_graph_linearised.py
Purpose: Strategy 3 - Graph-linearised retrieval, per the project
         approval form: "Linearised UMLS triples only." Unlike
         Strategy 2 (symbolic entity-linking + NetworkX graph
         traversal), Strategy 3 uses the SAME dense-retrieval
         mechanism as Strategy 1 (BGE embeddings + ChromaDB cosine
         search), but the retrieval corpus is linearised UMLS
         triples ("Metformin may treat Type 2 Diabetes.") instead
         of PDF guideline text. This isolates the retrieval
         MECHANISM as the variable being compared: Strategy 1 =
         dense retrieval over prose, Strategy 2 = symbolic graph
         traversal over triples, Strategy 3 = dense retrieval over
         triples-as-text.
Project: MSc AI Dissertation - Habeeb Adekeye
LLM: gpt-4o-mini, temperature=0 (same as Strategy 1 and 2, for a
     fair three-way comparison)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
BASE = Path(r"C:/Users/New PC/Downloads/MSC PROJECT/rag-project")
TRIPLES_PATH = BASE / "outputs" / "umls_filtered" / "umls_triples.csv"

CHROMA_DIR = BASE / "outputs" / "strategy3" / "chroma_db"
OUT_DIR = BASE / "outputs" / "strategy3"
TASK_RUN_PATH = OUT_DIR / "task_run.txt"
COMPARISON_PATH = OUT_DIR / "s1_s2_s3_comparison.txt"

COLLECTION = "strategy3_linearised"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOP_K = 5  # same as Strategy 1, for a fair three-way comparison

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 500

# Identical to 10_strategy2_kg_augmented.py's cleaning rules - both KG
# strategies share the same fact base.
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

SYSTEM_PROMPT = """You are a chronic disease self-management assistant answering using ONLY structured knowledge graph facts extracted from UMLS (not clinical guideline text). Facts take the form of (head, relation, tail) statements, e.g. "Metformin may treat Type 2 Diabetes." They were retrieved by semantic search over the knowledge graph, not by looking up a specific entity.

If one or more facts name a relevant drug, contraindication, cause, or finding, state it plainly and attribute it to the knowledge graph (e.g. "According to the knowledge graph, Albuterol may treat Asthma."). Do NOT claim guideline-level status ("first-line", specific dosing, numeric thresholds) unless a fact explicitly states it - these facts describe relationships, not clinical protocols. Only say "I cannot find sufficient information in the knowledge graph to answer this question" if NONE of the retrieved facts are relevant at all.

Because facts are retrieved by semantic similarity rather than precise entity lookup, they may occasionally include two facts that conflict (e.g. one says a drug may treat a condition while another says the same drug is contraindicated for it). If you notice such a conflict among the retrieved facts, explicitly say the knowledge graph contains conflicting information about that point, rather than silently picking one side.

Be concise. Do not invent information beyond what the facts state."""

# Identical to 06_strategy1_dense_retrieval.py's TEST_QUESTIONS - required
# for a fair three-way comparison.
TEST_QUESTIONS = [
    "What is the recommended HbA1c target for most adults with Type 2 diabetes?",
    "At what blood pressure level should medication be started for hypertension?",
    "What are the first-line medications for Type 2 diabetes according to current guidelines?",
    "What reliever medication is recommended for acute asthma symptoms?",
    "What lifestyle changes are recommended for managing high blood pressure?",
]

STRATEGY1_ANSWERS = [
    "The recommended HbA1c target for many nonpregnant adults with Type 2 diabetes is <7% (<53 mmol/mol) (Source: ADA_2026, p.141).",
    "Medication to lower blood pressure should be initiated when average systolic blood pressure (SBP) is ≥130 mm Hg or average diastolic blood pressure (DBP) is ≥80 mm Hg, particularly in adults with diabetes or chronic kidney disease (CKD) or at increased short-term cardiovascular disease (CVD) risk (i.e., estimated 10-year CVD risk ≥7.5%) (AHA_ACC_2025, p.33). Additionally, for adults with average blood pressure ≥140/90 mm Hg, medication is recommended (AHA_ACC_2025, p.5).",
    "According to the current guidelines, metformin is commonly used as the first-line medication for type 2 diabetes. It is effective, safe, inexpensive, and widely available, and it reduces risks of microvascular complications, cardiovascular events, and death (Source: ADA_2026, p.202).",
    "Short-acting inhaled beta2-agonist bronchodilators (SABA), such as salbutamol (albuterol) and terbutaline, are recommended for acute asthma symptoms. They provide quick relief of asthma symptoms and bronchoconstriction and can also be used as pretreatment before exercise. However, SABAs should be used only as needed and at the lowest dose and frequency required (Source: GINA_2026, p.248).",
    "Recommended lifestyle changes for managing high blood pressure include: 1. Weight loss: Aim for at least a 5% reduction in body weight. 2. DASH eating plan. 3. Sodium intake: <2300 mg/day. 4. Potassium intake: 3500-5000 mg/day. 5. Physical activity: 150 min/week aerobic + resistance training. 6. Alcohol limits. 7. Stress management (Source: AHA_ACC_2025, p.29; ADA_2026, p.227).",
]
STRATEGY2_ANSWERS = [
    "I cannot find sufficient information in the knowledge graph to answer this question.",
    "I cannot find sufficient information in the knowledge graph to answer this question.",
    "I cannot find sufficient information in the knowledge graph to answer this question.",
    "According to the knowledge graph, Albuterol may treat Asthma.",
    "I cannot find sufficient information in the knowledge graph to answer this question.",
]

load_dotenv(BASE / ".env")


def build_corpus():
    """Load umls_triples.csv, dedupe + relation-tier-exclude (identical
    cleaning to Strategy 2), return the cleaned linearised-sentence corpus."""
    triples = pd.read_csv(TRIPLES_PATH)
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


def build_index(embed_model, corpus):
    """Build (or reuse) the strategy3_linearised ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    existing = [c.name for c in client.list_collections()]

    if COLLECTION in existing:
        collection = client.get_collection(COLLECTION)
        if collection.count() == len(corpus):
            return collection
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"})

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
        embeddings = embed_model.encode(texts, show_progress_bar=False,
                                         normalize_embeddings=True).tolist()
        collection.add(documents=texts, ids=ids, metadatas=metas, embeddings=embeddings)
        print(f"[index] embedded {min(i + batch_size, n)}/{n}", flush=True)

    return collection


def run_strategy3(question, embed_model, collection, llm_client, top_k=TOP_K):
    """Dense-retrieve top_k linearised KG sentences, generate grounded answer."""
    qe = embed_model.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0].tolist()
    res = collection.query(query_embeddings=[qe], n_results=top_k)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    scores = [1 - d for d in dists]

    kg_block = "\n".join(f"- {d}" for d in docs) if docs else "(no matched KG facts)"
    user_msg = f"Knowledge graph facts:\n{kg_block}\n\nQuestion: {question}"

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = completion.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "contexts": docs,
        "sources": [f"UMLS KG ({m['sab']})" for m in metas],
        "scores": scores,
        "kg_facts": docs,
    }


def main():
    t0 = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TASK_RUN_PATH, "w", encoding="utf-8") as run_f:
        def emit(line=""):
            print(line)
            run_f.write(str(line) + "\n")
            run_f.flush()

        emit(f"START :: {t0:%Y-%m-%d %H:%M:%S}")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        corpus = build_corpus()
        emit(f"Corpus: {len(corpus)} linearised triples "
             f"(deduped + relation-tier-excluded from {TRIPLES_PATH.name})")

        embed_model = SentenceTransformer(EMBED_MODEL)
        collection = build_index(embed_model, corpus)
        llm_client = OpenAI(api_key=api_key)
        emit(f"Collection '{COLLECTION}': {collection.count()} vectors\n")
        emit("NOTE: Strategy 3 context contains ZERO PDF/guideline text. "
             "Retrieval is dense (semantic) search over linearised KG triples,"
             " NOT symbolic entity-linking/graph traversal (that's Strategy 2).\n")

        results = []
        for i, q in enumerate(TEST_QUESTIONS, 1):
            emit("=" * 78)
            emit(f"Q{i}: {q}")
            emit("=" * 78)
            r = run_strategy3(q, embed_model, collection, llm_client)
            results.append(r)
            emit(json.dumps(r, indent=2, ensure_ascii=False))
            emit()

        emit("Strategy 3 Graph-Linearised Retrieval - Pipeline Summary")
        emit(f"  Embedding model:    {EMBED_MODEL}")
        emit(f"  Collection:         {COLLECTION}")
        emit(f"  Corpus size:        {len(corpus)} linearised triples")
        emit(f"  LLM:                {LLM_MODEL}")
        emit(f"  Temperature:        {TEMPERATURE}")
        emit(f"  Context window:     top {TOP_K} retrieved facts per query")
        emit(f"  Test questions:     {len(TEST_QUESTIONS)}/{len(TEST_QUESTIONS)} attempted")

        t1 = datetime.now()
        emit(f"\nEND :: {t1:%Y-%m-%d %H:%M:%S}  ({(t1 - t0).total_seconds():.1f}s)")

    with open(COMPARISON_PATH, "w", encoding="utf-8") as cmp_f:
        header = (f"{'Question':<55} | {'S1 (dense/PDF)':<70} | "
                   f"{'S2 (KG traversal)':<55} | {'S3 (dense/KG)'}")
        cmp_f.write(header + "\n")
        cmp_f.write("-" * len(header) + "\n")
        print(header)
        print("-" * len(header))
        for q, s1, s2, r in zip(TEST_QUESTIONS, STRATEGY1_ANSWERS, STRATEGY2_ANSWERS, results):
            s3 = r["answer"]
            line = f"{q[:55]:<55} | {s1[:70]:<70} | {s2[:55]:<55} | {s3[:70]}"
            cmp_f.write(line + "\n")
            print(line)

    print(f"\nWrote transcript to {TASK_RUN_PATH}")
    print(f"Wrote S1 vs S2 vs S3 comparison to {COMPARISON_PATH}")

    return results


if __name__ == "__main__":
    main()
