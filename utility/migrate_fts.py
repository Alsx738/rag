"""
migrate_fts.py — One-time script to enable hybrid Full-Text Search.

What it does:
  1. Adds a `fts_vector tsvector` column to the amazon_reviews table
  2. Populates the column in resumable BATCHES (safe to interrupt and re-run)
  3. Creates a GIN index on the column for fast FTS queries

All steps are idempotent: safe to re-run thanks to IF NOT EXISTS / WHERE fts_vector IS NULL.

Usage (from project root):
    uv run python -m utility.migrate_fts
"""

from dotenv import load_dotenv
from sqlalchemy import text
from tqdm import tqdm

try:
    from utility.db import engine
except ImportError:
    from db import engine  # direct execution from the utility folder

load_dotenv()

TABLE      = "amazon_reviews"
BATCH_SIZE = 10_000


def main():
    print("=" * 55)
    print("  FTS Migration — amazon_reviews")
    print("=" * 55)

    # ── Step 1: Add fts_vector column (idempotent) ────────────────────────────
    print("\n[1/3] Adding fts_vector column (tsvector)...")
    with engine.begin() as conn:
        conn.execute(text(f"""
            ALTER TABLE {TABLE}
            ADD COLUMN IF NOT EXISTS fts_vector tsvector;
        """))
    print("      ✓ Column ready.")

    # ── Step 2: Populate fts_vector in batches (resumable) ────────────────────
    # Summary has weight 'A' (higher relevance — product title/synopsis)
    # reviewText has weight 'B' (review body)
    # 'english' → English stemming (running → run, foods → food, etc.)
    with engine.connect() as conn:
        total_null = conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLE} WHERE fts_vector IS NULL;")
        ).scalar()

    if total_null == 0:
        print("\n[2/3] fts_vector already populated for all rows. Skipping.")
    else:
        print(f"\n[2/3] Populating fts_vector — {total_null:,} rows to process "
              f"(batch size: {BATCH_SIZE:,})...")
        print("      ✓ Resumable: if interrupted, will restart from remaining rows.\n")

        processed = 0
        with tqdm(total=total_null, desc="Populating fts_vector", unit="rows") as pbar:
            while True:
                # Each batch updates only BATCH_SIZE rows with fts_vector IS NULL
                # and commits immediately → automatic checkpoint every batch
                with engine.begin() as conn:
                    result = conn.execute(text(f"""
                        UPDATE {TABLE}
                        SET fts_vector =
                            setweight(to_tsvector('english', COALESCE("Summary",    '')), 'A') ||
                            setweight(to_tsvector('english', COALESCE("reviewText", '')), 'B')
                        WHERE "Id" IN (
                            SELECT "Id"
                            FROM   {TABLE}
                            WHERE  fts_vector IS NULL
                            LIMIT  :batch_size
                        );
                    """), {"batch_size": BATCH_SIZE})

                    rows_updated = result.rowcount

                if rows_updated == 0:
                    break  # no rows left: done!

                processed += rows_updated
                pbar.update(rows_updated)

        print(f"\n      ✓ fts_vector populated ({processed:,} total rows).")

    # ── Step 3: Create GIN index (idempotent) ─────────────────────────────────
    # GIN (Generalized Inverted Index) is the standard index type for tsvector in PostgreSQL
    print("\n[3/3] Creating GIN index (amazon_reviews_fts_idx)...")
    print("      (may take a few minutes on large datasets)")
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS amazon_reviews_fts_idx
            ON {TABLE} USING GIN (fts_vector);
        """))
    print("      ✓ GIN index created.")

    print("\n✅ FTS migration completed successfully!")
    print("   You can now run main.py to test hybrid search.\n")


if __name__ == "__main__":
    main()
