import os
import math
import argparse
from tqdm import tqdm
import sqlalchemy
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment vars
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for Amazon reviews")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of reviews to process in this run")
    args = parser.parse_args()

    # 1. API Configuration from separate module
    from utility.llm import client, EMBEDDING_MODEL
    
    # 2. Database Connection using the centralized module
    from utility.db import engine
    
    table_name = "amazon_reviews"
    
    # 3. Database Schema Setup
    with engine.begin() as conn:     # .begin() automatically commits
        print("Ensuring pgvector extension and embedding column exist...")
        # Create vector extension if not present
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Add embedding column if it doesn't exist
        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding vector(1536);"))

    # 4. Count remaining rows directly from the DB
    with engine.connect() as conn:
        count_query = text(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NULL;")
        remaining_rows = conn.execute(count_query).scalar()

    if remaining_rows == 0:
        print("All rows already have embeddings! Nothing to do.")
        return

    # Apply the CLI limit if provided
    if args.limit is not None:
        target_rows = min(remaining_rows, args.limit)
        print(f"CLI limit applied: processing {target_rows} reviews (out of {remaining_rows} remaining).")
    else:
        target_rows = remaining_rows
        print(f"Found {remaining_rows} rows requiring embeddings.")
    
    # 5. Batch Processing
    batch_size = 1000
    model_name = EMBEDDING_MODEL
    processed_count = 0
    
    with engine.connect() as conn:
        # Start the progress bar
        with tqdm(total=target_rows, desc="Generating Embeddings", unit="rows") as pbar:
            while True:
                # Check if we have reached the requested limit
                if processed_count >= target_rows:
                    break
                    
                # Calculate how much to take in this batch (up to 1000, but do not exceed limit)
                current_limit = min(batch_size, target_rows - processed_count)
                
                # Fetch a batch of rows where embedding is NULL
                # Using summary and reviewText, casting just in case
                fetch_query = text(f"""
                    SELECT "Id", "Summary", "reviewText" 
                    FROM {table_name} 
                    WHERE embedding IS NULL 
                    LIMIT :limit
                """)
                rows = conn.execute(fetch_query, {"limit": current_limit}).fetchall()
                
                if not rows:
                    break # We are done!
                
                # Prepare text for embedding
                # Format: "Title: {summary} - Text: {review_text}"
                texts = []
                ids = []
                for row_id, summary, review_text in rows:
                    # Clean up NaNs or None if they exist
                    safe_summary = summary if summary is not None else ""
                    safe_review = review_text if review_text is not None else ""
                    combined_text = f"Title: {safe_summary} - Text: {safe_review}"
                    texts.append(combined_text)
                    ids.append(row_id)
                
                # Fetch embeddings from OpenAI
                response = client.embeddings.create(
                    input=texts,
                    model=model_name
                )

                # OpenAI returns embeddings in the same order as input
                embeddings = [data.embedding for data in response.data]

                # True bulk UPDATE for PostgreSQL
                # Builds a single query that updates all rows in one shot
                # (much faster than executemany or individual updates)
                values_parts = []
                bind_params = {}
                for i, (rec_id, emb) in enumerate(zip(ids, embeddings)):
                    vec_str = f"[{','.join(map(str, emb))}]"

                    # Use CAST instead of ::vector to avoid SQLAlchemy parser issues
                    values_parts.append(f"(:id_{i}, CAST(:emb_{i} AS vector))")
                    bind_params[f"id_{i}"] = rec_id
                    bind_params[f"emb_{i}"] = vec_str
                    
                values_sql = ",\n".join(values_parts)

                # This query bypasses SQLAlchemy's internal loop entirely.
                # It updates all rows in a single round-trip — much faster per batch.
                bulk_update_query = text(f"""
                    UPDATE {table_name} 
                    SET embedding = data.emb 
                    FROM (VALUES {values_sql}) AS data(id, emb) 
                    WHERE "{table_name}"."Id" = data.id;
                """)
                
                conn.execute(bulk_update_query, bind_params)
                conn.commit()
                        
                # Update progress tracking
                processed_count += len(rows)
                pbar.update(len(rows))

    # 6. Create an HNSW index to speed up vector similarity searches
    with engine.begin() as conn:
        print("\nCreating HNSW index for fast semantic search (this might take a moment)...")
        index_query = text(f"CREATE INDEX IF NOT EXISTS amazon_reviews_embedding_idx ON {table_name} USING hnsw (embedding vector_cosine_ops);")
        conn.execute(index_query)

    print("Embedding generation completed successfully!")

if __name__ == "__main__":
    main()
