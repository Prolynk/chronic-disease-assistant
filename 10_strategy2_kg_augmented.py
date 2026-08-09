
"""
Script:  10_strategy2_kg_augmented.py
Purpose: Strategy 2 - KG-ONLY retrieval, per the project approval form.
         Extracts entities from the question with scispaCy, maps them
         to UMLS CUIs via the project's own curated concept table,
         traverses a NetworkX graph built from umls_triples.csv to
         pull the top-15 most relevant (head, relation, tail) triples,
         and generates an answer grounded SOLELY in those triples.
         The LLM context contains ZERO PDF/guideline text - this is
         what makes Strategy 2 distinguishable from Strategy 1 for
         the dissertation's strategy comparison to be meaningful.
Project: MSc AI Dissertation - Habeeb Adekeye
LLM: gpt-4o-mini, temperature=0 (same as Strategy 1, for a fair
     apples-to-apples comparison - only the grounding source differs)
 
Entity linking note: the approval form specifies "extract entities
using scispaCy" and "map entities to UMLS CUIs" as two steps. This
script uses scispaCy's en_core_sci_sm for the first step (genuine
biomedical NER), but maps extracted mentions to CUIs via this
project's OWN filtered 2026AA concept table rather than scispaCy's
built-in EntityLinker. Reason: that linker ships a precomputed KB
from UMLS 2020AB (~3M concepts); a CUI it returns is not guaranteed
to exist in this project's own disease-scoped filtered subgraph
(2,695 concepts), which would silently produce empty traversals for
otherwise-valid entities. A curated ALIAS_MAP also acts as a safety
net directly on the question text for the three target diseases,
since en_core_sci_sm's lightweight mention detector has a verified
recall gap: it drops "asthma" entirely from "acute asthma symptoms"
(confirmed by direct testing), which would silently cost disease-
level coverage if scispaCy spans were the only input to CUI mapping.

Expected limitation:
UMLS triples encode semantic relationships (e.g. "Metformin may treat
Type 2 Diabetes"), not numeric clinical protocol values (e.g. "HbA1c
< 7%"). Strategy 2 answers are expected to be less numerically
specific than Strategy 1's guideline-grounded answers.
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import networkx as nx
import spacy
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
BASE = Path(r"C:/Users/New PC/Downloads/MSC PROJECT/rag-project")
TRIPLES_PATH = BASE / "outputs" / "umls_filtered" / "umls_triples.csv"
CONCEPTS_PATH = BASE / "outputs" / "umls_filtered" / "umls_concepts.csv"

OUT_DIR = BASE / "outputs" / "strategy2"
TASK_RUN_PATH = OUT_DIR / "task_run.txt"
MATCH_LOG_PATH = OUT_DIR / "kg_match_log.txt"
COMPARISON_PATH = OUT_DIR / "s1_vs_s2_comparison.txt"

NER_MODEL = "en_core_sci_sm"

MAX_ENTITIES = 5
MAX_TRIPLES_PER_ENTITY = 5
MAX_TRIPLES_TOTAL = 15
MIN_STR_LEN = 4
EXCLUDE_CUIS = {"C1298908"}  # STR == "No" (SNOMEDCT_US Finding) - false-positive risk

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 500

# Taxonomic/structural/mapping relations with near-zero clinical QA value
# (covers rel_type isa/inverse_isa, RN/RB narrower-broader, PAR/CHD parent-
# child, and LOINC survey-form artifacts), plus the generic catch-all
# "related to". Clinical-trial-metadata relations (GDC/PCDC/SARS2/SEROnet)
# are caught by the "authorized value" substring check in load_kg().
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

SYSTEM_PROMPT = """You are a chronic disease self-management assistant answering using ONLY structured knowledge graph facts extracted from UMLS (not clinical guideline text). Facts take the form of (head, relation, tail) statements, e.g. "Metformin may treat Type 2 Diabetes."

If one or more facts name a relevant drug, contraindication, cause, or finding, state it plainly and attribute it to the knowledge graph (e.g. "According to the knowledge graph, Albuterol may treat Asthma."). Do NOT claim guideline-level status ("first-line", specific dosing, numeric thresholds) unless a fact explicitly states it - these facts describe relationships, not clinical protocols. Only say "I cannot find sufficient information in the knowledge graph to answer this question" if NONE of the facts are relevant at all.

Be concise. Do not invent information beyond what the facts state."""

# Identical to 06_strategy1_dense_retrieval.py's TEST_QUESTIONS - required for
# a fair Strategy 1 vs. Strategy 2 comparison.
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

load_dotenv(BASE / ".env")


def load_kg():
    """Load the concept table + full triples file, build the NetworkX graph."""
    concepts = pd.read_csv(CONCEPTS_PATH)
    concepts["STR"] = concepts["STR"].astype(str)
    concepts = concepts[
        (concepts["STR"].str.len() >= MIN_STR_LEN)
        & (~concepts["CUI"].isin(EXCLUDE_CUIS))
    ].copy()
    concepts["_len"] = concepts["STR"].str.len()
    concepts = concepts.sort_values(
        "_len", ascending=False, kind="stable").reset_index(drop=True)

    triples = pd.read_csv(TRIPLES_PATH)
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

    return concepts, triples, graph


def extract_entities(question, nlp):
    """scispaCy biomedical NER - returns (text, start_char, end_char) spans."""
    doc = nlp(question)
    return [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents]


def map_to_cuis(question, entity_spans, concepts, max_entities=MAX_ENTITIES):
    """Map scispaCy mentions (+ curated disease-alias safety net) to CUIs
    in the project's own filtered concept table.

    Span-aware: once a character range in the question is claimed by a
    match, nothing overlapping it can match again. Without this, "type 2
    diabetes" and the plain "diabetes" alias both fire independently (they
    overlap in the text) and the noisy generic "Diabetes" CUI the alias map
    was specifically built to avoid sneaks back in via the second pass.
    """
    matched = []
    seen_cuis = set()
    consumed_spans = []

    def span_free(s, e):
        return not any(cs < e and s < ce for cs, ce in consumed_spans)

    # Pass 1: curated alias map directly on the question (disease safety net;
    # scispaCy's lightweight NER has verified recall gaps, see docstring).
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


def run_strategy2(question, nlp, concepts, graph, llm_client):
    """Extract entities -> map to CUIs -> traverse graph -> generate,
    grounded solely in KG triples."""
    entity_spans = extract_entities(question, nlp)
    matched_entities = map_to_cuis(question, entity_spans, concepts)
    kg_facts = traverse_graph(matched_entities, graph)
    entity_mentions = [text for text, _start, _end in entity_spans]

    kg_block = "\n".join(f"- {f}" for f in kg_facts) if kg_facts else "(no matched KG facts)"
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
        "contexts": kg_facts,               # KG facts ARE the context
        "sources": ["UMLS KG"] * len(kg_facts),
        "matched_entities": matched_entities,
        "entity_mentions_extracted": entity_mentions,
        "kg_facts": kg_facts,
    }


def main():
    t0 = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TASK_RUN_PATH, "w", encoding="utf-8") as run_f, \
         open(MATCH_LOG_PATH, "w", encoding="utf-8") as match_f:

        def emit(line=""):
            print(line)
            run_f.write(str(line) + "\n")
            run_f.flush()

        def log_match(line=""):
            match_f.write(str(line) + "\n")
            match_f.flush()

        emit(f"START :: {t0:%Y-%m-%d %H:%M:%S}")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        emit(f"Loading scispaCy model '{NER_MODEL}'...")
        nlp = spacy.load(NER_MODEL)
        concepts, triples, graph = load_kg()
        llm_client = OpenAI(api_key=api_key)

        emit(f"KG concepts loaded: {len(concepts)} (after exclusions)")
        emit(f"KG triples loaded: {len(triples)} (after dedupe + tier filtering, "
             f"source: umls_triples.csv)")
        emit(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")
        emit("NOTE: Strategy 2 context contains ZERO PDF/guideline text. "
             "Grounding is solely UMLS knowledge graph triples.\n")

        results = []
        for i, q in enumerate(TEST_QUESTIONS, 1):
            emit("=" * 78)
            emit(f"Q{i}: {q}")
            emit("=" * 78)
            r = run_strategy2(q, nlp, concepts, graph, llm_client)
            results.append(r)
            emit(json.dumps(r, indent=2, ensure_ascii=False))
            emit()

            log_match(f"Q{i}: {q}")
            log_match(f"  scispaCy entity mentions: {r['entity_mentions_extracted']}")
            log_match(
                "  matched_entities: "
                + (", ".join(f"{e['name']} ({e['cui']})" for e in r["matched_entities"]) or "(none)")
            )
            log_match(f"  kg_facts ({len(r['kg_facts'])}):")
            for f in r["kg_facts"]:
                log_match(f"    - {f}")
            log_match("")

        avg_entities = sum(len(r["matched_entities"]) for r in results) / len(results)
        avg_facts = sum(len(r["kg_facts"]) for r in results) / len(results)

        emit("Strategy 2 KG-Only Retrieval - Pipeline Summary")
        emit(f"  Entity extraction:     scispaCy ({NER_MODEL})")
        emit(f"  CUI mapping:            curated concept table"
             f"see script docstring)")
        emit(f"  Graph:                  NetworkX MultiDiGraph, {graph.number_of_nodes()} nodes, "
             f"{graph.number_of_edges()} edges")
        emit(f"  Avg matched entities:   {avg_entities:.1f} / question")
        emit(f"  Avg KG facts used:      {avg_facts:.1f} / question")
        emit(f"  LLM:                    {LLM_MODEL}")
        emit(f"  Temperature:            {TEMPERATURE}")
        emit(f"  Context:                up to {MAX_TRIPLES_TOTAL} KG facts")
        emit(f"  Test questions:         {len(TEST_QUESTIONS)}/{len(TEST_QUESTIONS)} attempted")
        emit("")
        emit("Expected limitation (documented findings): UMLS triples encode")
        emit("semantic relationships (e.g. 'Metformin may treat Type 2 Diabetes'), not")
        emit("numeric clinical protocol values (e.g. 'HbA1c < 7%'). Strategy 2 answers")
        emit("are expected to be less numerically specific than Strategy 1's guideline-")
        emit("grounded answers - this is a genuine structural-precision-vs-clinical-")
        emit("specificity trade-off.")

        t1 = datetime.now()
        emit(f"\nEND :: {t1:%Y-%m-%d %H:%M:%S}  ({(t1 - t0).total_seconds():.1f}s)")

    with open(COMPARISON_PATH, "w", encoding="utf-8") as cmp_f:
        header = f"{'Question':<70} | {'S1 answer (first 100 chars)':<103} | {'S2 answer (first 100 chars)'}"
        cmp_f.write(header + "\n")
        cmp_f.write("-" * len(header) + "\n")
        print(header)
        print("-" * len(header))
        for q, s1, r in zip(TEST_QUESTIONS, STRATEGY1_ANSWERS, results):
            s2 = r["answer"]
            line = f"{q[:70]:<70} | {s1[:100]:<103} | {s2[:100]}"
            cmp_f.write(line + "\n")
            print(line)

    print(f"\nWrote transcript to {TASK_RUN_PATH}")
    print(f"Wrote KG match audit trail to {MATCH_LOG_PATH}")
    print(f"Wrote S1 vs S2 comparison to {COMPARISON_PATH}")

    return results


if __name__ == "__main__":
    main()
