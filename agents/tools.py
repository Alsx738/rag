from pydantic import BaseModel, Field
from langchain_core.tools import tool
from sqlalchemy import text
from utility.db import engine
from utility.llm import client, EMBEDDING_MODEL

# --- PYDANTIC INPUT SCHEMAS ---

class SimilarReviewsInput(BaseModel):
    query: str = Field(..., description="The natural language query based on meaning or textual content.")
    limit: int = Field(default=5, description="Maximum number of reviews to return. Default is 5.")

class OtherUserReviewsInput(BaseModel):
    user_id: str = Field(..., description="The alphanumeric ID of the user whose other reviews to search.")
    query: str = Field(..., description="Natural language query to find contextually relevant products.")
    exclude_product_id: str = Field(..., description="Product ID to exclude from results (the one the user is already viewing).")
    limit: int = Field(default=5, description="Maximum number of results to return.")

# --- TOOLS ---

@tool(args_schema=SimilarReviewsInput)
def get_similar_reviews(query: str, limit: int = 5) -> str:
    """
    Use this tool to search for reviews on Amazon based on the meaning or textual content.
    Provide a natural language query and you will receive the most relevant reviews,
    including the product_id, user_id, summary, and review text.
    Uses hybrid search (semantic vector search + full-text search) fused with
    Reciprocal Rank Fusion (RRF) for best-of-both-worlds relevance.
    """
    # 1. Generate embedding for semantic search
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_vector = response.data[0].embedding

    # 2. Candidate pool: fetch limit*4 from each branch before merging
    pool = limit * 4

    # 3. Hybrid query: two CTEs (semantic + FTS) merged with RRF.
    #    plainto_tsquery converts natural language to tsquery (English stemming).
    #    RRF formula: 1/(60 + semantic_rank) + 1/(60 + fts_rank).
    #    FULL OUTER JOIN on "Id" (PK): documents appearing in only one list
    #    contribute 0 from the missing branch — graceful fallback to pure semantic.
    hybrid_query = text("""
        WITH semantic AS (
            SELECT
                "Id",
                "ProductId",
                "UserId",
                "Summary",
                "reviewText",
                ROW_NUMBER() OVER (
                    ORDER BY embedding <=> CAST(:query_vector AS vector)
                ) AS rank
            FROM amazon_reviews
            WHERE "Score" >= 3.5
            ORDER BY embedding <=> CAST(:query_vector AS vector)
            LIMIT :pool
        ),
        fts AS (
            SELECT
                "Id",
                "ProductId",
                "UserId",
                "Summary",
                "reviewText",
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :fts_query)) DESC
                ) AS rank
            FROM amazon_reviews
            WHERE fts_vector @@ plainto_tsquery('english', :fts_query) AND "Score" >= 3.5
            LIMIT :pool
        ),
        combined AS (
            SELECT
                COALESCE(s."ProductId", f."ProductId") AS "ProductId",
                COALESCE(s."UserId",    f."UserId")    AS "UserId",
                COALESCE(s."Summary",   f."Summary")   AS "Summary",
                COALESCE(s."reviewText",f."reviewText") AS "reviewText",
                COALESCE(1.0 / (60 + s.rank), 0) +
                COALESCE(1.0 / (60 + f.rank), 0)      AS rrf_score
            FROM semantic s
            FULL OUTER JOIN fts f ON s."Id" = f."Id"
        )
        SELECT "ProductId", "UserId", "Summary", "reviewText", rrf_score
        FROM combined
        ORDER BY rrf_score DESC
        LIMIT :limit
    """)

    with engine.connect() as conn:
        result = conn.execute(hybrid_query, {
            "query_vector": str(query_vector),
            "fts_query": query,
            "pool": pool,
            "limit": limit,
        })
        rows = result.fetchall()

    if not rows:
        return "No reviews found."

    formatted_results = "\n\n".join([
        f"Product ID: {row[0]}\nUser ID: {row[1]}\nSummary: {row[2]}\nReview: {row[3]}\nRRF Score: {row[4]:.6f}"
        for row in rows
    ])
    return formatted_results

@tool(args_schema=OtherUserReviewsInput)
def get_other_user_reviews(user_id: str, query: str, exclude_product_id: str, limit: int = 5) -> str:
    """
    Use this tool to find other products reviewed by a specific user.
    It takes the user_id, a natural language query to find contextually relevant products,
    and the exclude_product_id to avoid suggesting the product the user is currently looking at.
    """
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_vector = response.data[0].embedding

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    "ProductId",
                    "Summary",
                    "reviewText"
                FROM amazon_reviews
                WHERE "UserId" = :user_id AND "ProductId" != :exclude_product_id AND "Score" >= 3.5
                ORDER BY embedding <=> CAST(:query_vector AS vector)
                LIMIT :limit
            """),
            {
                "user_id": user_id,
                "exclude_product_id": exclude_product_id,
                "query_vector": str(query_vector),
                "limit": limit
            }
        )
        rows = result.fetchall()

    if not rows:
        return "No other relevant products found for this user."

    formatted_results = "\n\n".join([
        f"Product ID: {row[0]}\nSummary: {row[1]}\nReview: {row[2]}"
        for row in rows
    ])
    return formatted_results