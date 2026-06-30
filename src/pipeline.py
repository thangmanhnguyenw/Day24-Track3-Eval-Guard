from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline."""
    latency: dict[str, float] = {}
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    latency["M1 Chunking"] = time.time() - t0
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({latency['M1 Chunking']:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    if os.getenv("SKIP_ENRICHMENT"):
        print(f"\n[2/4] Skipping enrichment (SKIP_ENRICHMENT=1)...", flush=True)
        latency["M5 Enrichment"] = 0.0
    else:
        print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
        enriched = enrich_chunks(all_chunks)
        latency["M5 Enrichment"] = time.time() - t0
        if enriched:
            all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
            print(f"  ✓ Enriched {len(enriched)} chunks ({latency['M5 Enrichment']:.1f}s)", flush=True)
        else:
            print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    latency["M2 Indexing"] = time.time() - t0
    print(f"  ✓ Indexed ({latency['M2 Indexing']:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    latency["M3 Reranker load"] = time.time() - t0
    print(f"  ✓ Reranker ready ({latency['M3 Reranker load']:.1f}s)", flush=True)

    return search, reranker, latency


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    from src.llm_client import chat, is_llm_available
    if is_llm_available() and contexts:
        try:
            context_str = "\n\n".join(contexts)
            answer = chat(
                "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'",
                f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                max_tokens=1024,
            )
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, latency: dict | None = None):
    """Run evaluation on test set."""
    latency = latency or {}
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    t_query = time.time()
    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)
    latency["Query + LLM (all)"] = time.time() - t_query

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    latency["M4 RAGAS eval"] = time.time() - t0
    print(f"  ✓ RAGAS done ({latency['M4 RAGAS eval']:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    print(f"{'Step':<25} {'Time (s)':>10}")
    print("-" * 37)
    for step, secs in latency.items():
        print(f"{step:<25} {secs:>10.1f}")
    print("-" * 37)
    print(f"{'Total':<25} {sum(latency.values()):>10.1f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, latency = build_pipeline()
    evaluate_pipeline(search, reranker, latency)
    print(f"\nTotal wall-clock: {time.time() - start:.1f}s")
