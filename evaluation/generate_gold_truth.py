"""
generate_gold_truth.py — Genera il dataset di valutazione sintetico.

Campiona 30 ProductId diversi dal DB, recupera le recensioni più utili
per ciascuno e usa GPT per generare 4 query naturali in inglese.
Salva il risultato in evaluation/gold_truth.json.

Uso (da root del progetto):
    uv run python -m evaluation.generate_gold_truth
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

# Gestisce sia l'esecuzione come modulo che come script diretto
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utility.db import engine
from utility.llm import client, CHAT_MODEL

load_dotenv()

# ── Configurazione ────────────────────────────────────────────────────────────
OUTPUT_FILE          = Path(__file__).parent / "gold_truth.json"
NUM_PRODUCTS         = 200
REVIEWS_PER_PRODUCT  = 3   # recensioni più utili da passare al LLM come contesto
QUERIES_PER_PRODUCT  = 4   # query sintetiche da generare per prodotto
MIN_REVIEWS          = 3   # soglia minima: ignora prodotti con poche recensioni


# ── DB helpers ────────────────────────────────────────────────────────────────

def sample_product_ids(conn, n: int, min_reviews: int) -> list[str]:
    """Campiona n ProductId distinti con almeno min_reviews recensioni."""
    result = conn.execute(text("""
        SELECT "ProductId"
        FROM amazon_reviews
        GROUP BY "ProductId"
        HAVING COUNT(*) >= :min_reviews
        ORDER BY RANDOM()
        LIMIT :n
    """), {"min_reviews": min_reviews, "n": n})
    return [row[0] for row in result.fetchall()]


def fetch_reviews(conn, product_id: str, limit: int) -> list[dict]:
    """Recupera le recensioni più votate per un prodotto."""
    result = conn.execute(text("""
        SELECT "Summary", "reviewText", "Score"
        FROM amazon_reviews
        WHERE "ProductId" = :pid
        ORDER BY "HelpfulnessNumerator" DESC NULLS LAST
        LIMIT :limit
    """), {"pid": product_id, "limit": limit})
    return [
        {"summary": str(row[0] or ""), "text": str(row[1] or ""), "score": row[2]}
        for row in result.fetchall()
    ]


# ── LLM query generation ──────────────────────────────────────────────────────

def generate_queries(reviews: list[dict], n_queries: int) -> list[str]:
    """Chiede a GPT di generare n query naturali in inglese per il prodotto."""
    reviews_block = "\n\n".join([
        f"[Review {i+1}]\nTitle: {r['summary']}\nText: {r['text'][:500]}"
        for i, r in enumerate(reviews)
    ])

    prompt = f"""You are simulating an Amazon shopper searching for a product.
Based on the following product reviews, generate exactly {n_queries} different \
natural language search queries in English that a user might type to find this product.

Rules:
- Write {n_queries} queries total with MIXED styles:
  * 1-2 queries should use specific keywords or brand names if you can recognise them
    from the reviews (e.g. "Purina cat food", "Blue Buffalo grain-free")
  * The remaining queries should be paraphrased, intent-based, without exact brand names
    (e.g. "best grain-free treat for sensitive dogs", "looking for USA-made dog jerky")
- Vary the phrasing: some problem/solution, some feature-focused, some brand/keyword-focused
- Do NOT include the product ID itself
- Return ONLY a JSON object: {{"queries": ["query1", "query2", "query3", "query4"]}}

Reviews:
{reviews_block}"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate search queries for product retrieval evaluation. "
                    "Always respond with valid JSON containing a 'queries' key "
                    "whose value is a list of strings."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,  # leggermente alta per diversificare le query
    )

    data = json.loads(response.choices[0].message.content)
    return data.get("queries", [])[:n_queries]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Gold Truth Generation")
    print("=" * 55)

    with engine.connect() as conn:
        print(f"\nCampionamento di {NUM_PRODUCTS} ProductId diversi...")
        product_ids = sample_product_ids(conn, NUM_PRODUCTS, MIN_REVIEWS)
        print(f"  ✓ {len(product_ids)} prodotti trovati\n")

        gold_truth = []

        for product_id in tqdm(product_ids, desc="Generating queries", unit="product"):
            reviews = fetch_reviews(conn, product_id, REVIEWS_PER_PRODUCT)
            if not reviews:
                continue

            queries = generate_queries(reviews, QUERIES_PER_PRODUCT)

            gold_truth.append({
                "product_id": product_id,
                "review_count": len(reviews),
                "sample_reviews": [
                    {"summary": r["summary"], "text": r["text"][:300]}
                    for r in reviews
                ],
                "queries": queries,
            })

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(gold_truth, f, indent=2, ensure_ascii=False)

    total_queries = sum(len(e["queries"]) for e in gold_truth)
    print(f"\n✅  Gold truth salvato in: {OUTPUT_FILE}")
    print(f"    {len(gold_truth)} prodotti × {QUERIES_PER_PRODUCT} query = {total_queries} test case totali\n")


if __name__ == "__main__":
    main()
