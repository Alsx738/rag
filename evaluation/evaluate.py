"""
evaluate.py — Compares Semantic-Only vs FTS-Only vs Hybrid (RRF) on gold truth test cases.

Metrics:
  - Hit Rate@1, @3, @5, @10  (strict: exact ProductId match)
  - Hit Rate@1 (relaxed: LLM judge deems top-1 relevant) — only with --judge flag
  - MRR (Mean Reciprocal Rank)

Usage (from project root):
    uv run python -m evaluation.evaluate            # strict metrics only
    uv run python -m evaluation.evaluate --judge    # strict + relaxed (LLM judge)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utility.db import engine
from utility.llm import client, EMBEDDING_MODEL

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
GOLD_TRUTH_FILE = Path(__file__).parent / "gold_truth.json"
REPORTS_DIR     = Path(__file__).parent / "reports"
TOP_K_VALUES    = [1, 3, 5, 10]
EVAL_LIMIT      = 10
JUDGE_MODEL     = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(query: str) -> list[float]:
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return response.data[0].embedding


# ── Search functions ──────────────────────────────────────────────────────────

def semantic_search(conn, query_vector: list[float], limit: int) -> list[str]:
    """Pure semantic search via cosine distance. Returns ProductIds ranked by similarity."""
    result = conn.execute(text("""
        SELECT "ProductId"
        FROM amazon_reviews
        ORDER BY embedding <=> CAST(:query_vector AS vector)
        LIMIT :limit
    """), {"query_vector": str(query_vector), "limit": limit})
    return [row[0] for row in result.fetchall()]


def fts_only_search(conn, query: str, limit: int) -> list[str]:
    """Pure full-text search via ts_rank. Returns empty list if no FTS matches."""
    try:
        result = conn.execute(text("""
            SELECT "ProductId"
            FROM amazon_reviews
            WHERE fts_vector @@ plainto_tsquery('english', :fts_query)
            ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :fts_query)) DESC
            LIMIT :limit
        """), {"fts_query": query, "limit": limit})
        return [row[0] for row in result.fetchall()]
    except Exception:
        return []


def hybrid_search(conn, query_vector: list[float], query: str, limit: int) -> list[str]:
    """
    Hybrid search: semantic + FTS merged with RRF.
    Falls back to pure semantic if plainto_tsquery produces no tokens.
    """
    pool = limit * 4
    try:
        result = conn.execute(text("""
            WITH semantic AS (
                SELECT
                    "Id", "ProductId",
                    ROW_NUMBER() OVER (
                        ORDER BY embedding <=> CAST(:query_vector AS vector)
                    ) AS rank
                FROM amazon_reviews
                ORDER BY embedding <=> CAST(:query_vector AS vector)
                LIMIT :pool
            ),
            fts AS (
                SELECT
                    "Id", "ProductId",
                    ROW_NUMBER() OVER (
                        ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :fts_query)) DESC
                    ) AS rank
                FROM amazon_reviews
                WHERE fts_vector @@ plainto_tsquery('english', :fts_query)
                LIMIT :pool
            ),
            combined AS (
                SELECT
                    COALESCE(s."ProductId", f."ProductId") AS "ProductId",
                    COALESCE(1.0 / (60 + s.rank), 0) +
                    COALESCE(1.0 / (60 + f.rank), 0) AS rrf_score
                FROM semantic s
                FULL OUTER JOIN fts f ON s."Id" = f."Id"
            )
            SELECT "ProductId" FROM combined
            ORDER BY rrf_score DESC
            LIMIT :limit
        """), {"query_vector": str(query_vector), "fts_query": query,
               "pool": pool, "limit": limit})
        return [row[0] for row in result.fetchall()]
    except Exception:
        return semantic_search(conn, query_vector, limit)


# ── LLM Judge ─────────────────────────────────────────────────────────────────

def fetch_reviews_for_product(conn, product_id: str) -> str:
    """
    Fetches the top 3 most helpful reviews for a product and returns them
    as a single text block for the LLM judge.
    3 reviews instead of 1 to avoid uninformative single reviews like 'love it'.
    """
    rows = conn.execute(text("""
        SELECT "Summary", LEFT("reviewText", 300)
        FROM amazon_reviews
        WHERE "ProductId" = :pid
        ORDER BY "HelpfulnessNumerator" DESC NULLS LAST
        LIMIT 3
    """), {"pid": product_id}).fetchall()

    if not rows:
        return "No reviews available."

    return "\n\n".join([
        f"Title: {row[0]}\nText: {row[1]}"
        for row in rows
    ])


def llm_judge(query: str, reviews_text: str) -> bool:
    """
    Asks the LLM whether the retrieved product is relevant to the query.
    Returns True if relevant, False otherwise.
    """
    prompt = (
        f"User query: {query}\n\n"
        f"Product reviews:\n{reviews_text}"
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an information retrieval judge. "
                    "Given a user search query and product reviews, decide if this product "
                    "is relevant — i.e. would a user searching for that query be satisfied "
                    "by this product? Answer with exactly one word: YES or NO."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")


# ── Metrics ───────────────────────────────────────────────────────────────────

def hit_at_k(results: list[str], target: str, k: int) -> bool:
    return target in results[:k]


def reciprocal_rank(results: list[str], target: str) -> float:
    try:
        return 1.0 / (results.index(target) + 1)
    except ValueError:
        return 0.0


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(data: dict):
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved → {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate search quality on gold truth.")
    parser.add_argument(
        "--judge", action="store_true",
        help="Enable LLM-as-judge for relaxed Hit Rate@1 (requires API calls)."
    )
    args = parser.parse_args()
    use_judge = args.judge

    print("=" * 62)
    print("  Search Evaluation — Semantic  |  FTS Only  |  Hybrid RRF")
    print("=" * 62)

    with open(GOLD_TRUTH_FILE, "r", encoding="utf-8") as f:
        gold_truth = json.load(f)

    total_queries = sum(len(e["queries"]) for e in gold_truth)
    judge_label = (
        f"  LLM Judge: ENABLED ({JUDGE_MODEL})"
        if use_judge else
        "  LLM Judge: disabled  (run with --judge to enable)"
    )
    print(f"\n  Products: {len(gold_truth)}  |  Queries: {total_queries}")
    print(f"  Metrics:  Hit Rate@{TOP_K_VALUES} + MRR (top-{EVAL_LIMIT})")
    print(judge_label + "\n")

    # Strict accumulators
    sem_hits = {k: 0 for k in TOP_K_VALUES}
    fts_hits = {k: 0 for k in TOP_K_VALUES}
    hyb_hits = {k: 0 for k in TOP_K_VALUES}
    sem_rr, fts_rr, hyb_rr = [], [], []

    # Relaxed accumulators (LLM judge, @1 only)
    sem_relaxed = fts_relaxed = hyb_relaxed = 0
    judge_calls = 0

    with engine.connect() as conn:
        for entry in tqdm(gold_truth, desc="Evaluating", unit="product"):
            product_id = entry["product_id"]

            for query in entry["queries"]:
                vec     = get_embedding(query)
                sem_res = semantic_search(conn, vec, EVAL_LIMIT)
                fts_res = fts_only_search(conn, query, EVAL_LIMIT)
                hyb_res = hybrid_search(conn, vec, query, EVAL_LIMIT)

                # Strict hits
                for k in TOP_K_VALUES:
                    if hit_at_k(sem_res, product_id, k): sem_hits[k] += 1
                    if hit_at_k(fts_res, product_id, k): fts_hits[k] += 1
                    if hit_at_k(hyb_res, product_id, k): hyb_hits[k] += 1

                sem_rr.append(reciprocal_rank(sem_res, product_id))
                fts_rr.append(reciprocal_rank(fts_res, product_id))
                hyb_rr.append(reciprocal_rank(hyb_res, product_id))

                # Relaxed @1: judge only when strict top-1 misses
                if use_judge:
                    for results, bucket in [
                        (sem_res, "sem"), (fts_res, "fts"), (hyb_res, "hyb")
                    ]:
                        if hit_at_k(results, product_id, 1):
                            # Already a strict hit — count as relaxed too
                            if bucket == "sem": sem_relaxed += 1
                            elif bucket == "fts": fts_relaxed += 1
                            else: hyb_relaxed += 1
                        elif results:
                            # Strict miss but results exist — ask LLM
                            reviews_text = fetch_reviews_for_product(conn, results[0])
                            if llm_judge(query, reviews_text):
                                if bucket == "sem": sem_relaxed += 1
                                elif bucket == "fts": fts_relaxed += 1
                                else: hyb_relaxed += 1
                            judge_calls += 1

    sem_mrr = sum(sem_rr) / len(sem_rr)
    fts_mrr = sum(fts_rr) / len(fts_rr)
    hyb_mrr = sum(hyb_rr) / len(hyb_rr)

    fts_coverage = sum(1 for r in fts_rr if r > 0.0) / total_queries * 100

    # ── Print: STRICT section ─────────────────────────────────────────────────
    w = 10
    print("\n" + "═" * 58)
    print("  STRICT METRICS — Exact ProductId match")
    print("═" * 58)
    print(f"  {'':20} {'Semantic':>{w}} {'FTS Only':>{w}} {'Hybrid':>{w}} {'vs Sem':>8}")
    print("  " + "─" * 54)
    for k in TOP_K_VALUES:
        s = sem_hits[k] / total_queries * 100
        f = fts_hits[k] / total_queries * 100
        h = hyb_hits[k] / total_queries * 100
        d = h - s
        sign = "+" if d >= 0 else ""
        print(f"  {'Hit Rate @'+str(k):20} {s:>{w-1}.1f}% {f:>{w-1}.1f}% {h:>{w-1}.1f}% {sign+f'{d:.1f}%':>8}")
    d_mrr = hyb_mrr - sem_mrr
    sign = "+" if d_mrr >= 0 else ""
    print(f"  {'MRR':20} {sem_mrr:>{w}.4f} {fts_mrr:>{w}.4f} {hyb_mrr:>{w}.4f} {sign+f'{d_mrr:.4f}':>8}")
    print(f"\n  FTS Coverage: {fts_coverage:.1f}% of queries returned ≥1 result")

    # ── Print: RELAXED section ────────────────────────────────────────────────
    if use_judge:
        print("\n" + "═" * 58)
        print(f"  RELAXED METRICS — LLM judge on top-1 miss ({JUDGE_MODEL})")
        print("═" * 58)
        print(f"  {'':20} {'Semantic':>{w}} {'FTS Only':>{w}} {'Hybrid':>{w}} {'vs Sem':>8}")
        print("  " + "─" * 54)
        s_r = sem_relaxed / total_queries * 100
        f_r = fts_relaxed / total_queries * 100
        h_r = hyb_relaxed / total_queries * 100
        d_r = h_r - s_r
        sign = "+" if d_r >= 0 else ""
        print(f"  {'Hit Rate @1':20} {s_r:>{w-1}.1f}% {f_r:>{w-1}.1f}% {h_r:>{w-1}.1f}% {sign+f'{d_r:.1f}%':>8}")
        sem_gap = s_r - (sem_hits[1] / total_queries * 100)
        hyb_gap = h_r - (hyb_hits[1] / total_queries * 100)
        print(f"\n  Underestimation gap — Semantic: +{sem_gap:.1f}pp | Hybrid: +{hyb_gap:.1f}pp")
        print(f"  LLM Judge calls: {judge_calls}")

    print("═" * 58)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "gold_truth_file": str(GOLD_TRUTH_FILE),
            "num_products": len(gold_truth),
            "num_queries": total_queries,
            "eval_limit": EVAL_LIMIT,
            "top_k_values": TOP_K_VALUES,
            "embedding_model": EMBEDDING_MODEL,
            "llm_judge_enabled": use_judge,
            "judge_model": JUDGE_MODEL if use_judge else None,
        },
        "summary": {
            "semantic": {
                "strict_hit_rate": {f"@{k}": round(sem_hits[k] / total_queries, 4) for k in TOP_K_VALUES},
                "mrr": round(sem_mrr, 4),
                **({"relaxed_hit_rate_at_1": round(sem_relaxed / total_queries, 4)} if use_judge else {}),
            },
            "fts_only": {
                "strict_hit_rate": {f"@{k}": round(fts_hits[k] / total_queries, 4) for k in TOP_K_VALUES},
                "mrr": round(fts_mrr, 4),
                "coverage_pct": round(fts_coverage, 1),
                **({"relaxed_hit_rate_at_1": round(fts_relaxed / total_queries, 4)} if use_judge else {}),
            },
            "hybrid": {
                "strict_hit_rate": {f"@{k}": round(hyb_hits[k] / total_queries, 4) for k in TOP_K_VALUES},
                "mrr": round(hyb_mrr, 4),
                **({"relaxed_hit_rate_at_1": round(hyb_relaxed / total_queries, 4)} if use_judge else {}),
            },
        },
        "analysis": {
            "fts_coverage_pct": round(fts_coverage, 1),
            **({"llm_judge_calls": judge_calls,
                "underestimation_gap_pp": {
                    "semantic": round(s_r - (sem_hits[1] / total_queries * 100), 1),
                    "hybrid":   round(h_r - (hyb_hits[1] / total_queries * 100), 1),
                }} if use_judge else {}),
        },
    }
    save_report(report)


if __name__ == "__main__":
    main()
