"""Import the existing collection without calling or retraining any model."""

import json
from pathlib import Path

import numpy as np
import psycopg
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "embeddings"
CHUNKS_PATH = DATA / "bge_small_en_v1_5_chunks.json"
VECTORS_PATH = DATA / "bge_small_en_v1_5.npy"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
EXPECTED_DOCUMENTS = 10
EXPECTED_CHUNKS = 2994


def validate_collection(chunks, vectors):
    if not isinstance(chunks, list) or len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(f"Expected a list of {EXPECTED_CHUNKS} chunks.")
    if vectors.shape != (len(chunks), DIMENSIONS):
        raise ValueError("Embedding shape does not match the chunk collection.")
    if vectors.dtype.kind != "f" or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain only finite floating-point numbers.")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=0.001):
        raise ValueError("Expected the previously normalised BGE embeddings.")

    seen_ids = set()
    documents = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at position {index} is not a record.")
        for field in ("chunk_id", "document_name", "chunk_text"):
            value = chunk.get(field)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"Invalid {field} at chunk position {index}.")
        page = chunk.get("page_number")
        row = chunk.get("embedding_row")
        if type(page) is not int or page < 1:
            raise ValueError(f"Invalid page number at chunk position {index}.")
        if type(row) is not int or row != index:
            raise ValueError(f"Embedding-row alignment failed at position {index}.")
        if chunk["chunk_id"] in seen_ids:
            raise ValueError(f"Duplicate chunk ID at position {index}.")
        seen_ids.add(chunk["chunk_id"])
        documents.add(chunk["document_name"])

    if len(documents) != EXPECTED_DOCUMENTS:
        raise ValueError(f"Expected {EXPECTED_DOCUMENTS} distinct documents.")
    return sorted(documents), np.ascontiguousarray(vectors, dtype=np.float32)


def load_collection():
    with CHUNKS_PATH.open("r", encoding="utf-8-sig") as source:
        chunks = json.load(source)
    vectors = np.load(VECTORS_PATH, allow_pickle=False)
    documents, vectors = validate_collection(chunks, vectors)
    return documents, chunks, vectors


def read_password():
    env_path = ROOT / ".env.db"
    if not env_path.is_file():
        raise ValueError("The existing .env.db file is missing from the project folder.")
    settings = dotenv_values(env_path, interpolate=False, encoding="utf-8-sig")
    password = settings.get("EDI_DB_PASSWORD")
    if not password or not password.strip():
        raise ValueError("EDI_DB_PASSWORD is missing or empty in .env.db.")
    if password == "REPLACE_WITH_YOUR_OWN_UNIQUE_PASSWORD":
        raise ValueError("Replace the example password in .env.db before importing.")
    return password


def metadata_row(chunk):
    return (
        chunk["chunk_id"], chunk["document_name"], chunk["page_number"],
        chunk["chunk_text"], chunk["embedding_row"], MODEL_NAME,
    )


def verify_database(cursor, documents, chunks, vectors):
    cursor.execute("SELECT document_name FROM edi.documents")
    stored_documents = [row[0] for row in cursor.fetchall()]
    if sorted(stored_documents) != documents:
        raise ValueError("Stored document names differ from the source collection.")

    cursor.execute("""
        SELECT chunk_id, document_name, page_number, chunk_text,
               embedding_row, embedding_model, embedding::text
        FROM edi.chunks ORDER BY embedding_row
    """)
    stored_chunks = cursor.fetchall()
    if len(stored_chunks) != len(chunks):
        raise ValueError("Stored chunk count differs from the source collection.")
    for index, record in enumerate(stored_chunks):
        if record[:6] != metadata_row(chunks[index]):
            raise ValueError(f"Stored text or metadata differs at embedding row {index}.")
        stored_vector = np.asarray(json.loads(record[6]), dtype=np.float32)
        if not np.array_equal(stored_vector, vectors[index]):
            raise ValueError(f"Stored embedding differs at embedding row {index}.")


def import_collection(documents, chunks, vectors, password):
    with psycopg.connect(
        host="127.0.0.1", port=5433,
        dbname="document_intelligence", user="edi_admin",
        password=password, connect_timeout=10,
        application_name="edi_collection_import",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute(
                "LOCK TABLE edi.documents, edi.chunks IN SHARE ROW EXCLUSIVE MODE"
            )
            cursor.execute("""
                SELECT (SELECT COUNT(*) FROM edi.documents),
                       (SELECT COUNT(*) FROM edi.chunks)
            """)
            empty_database = cursor.fetchone() == (0, 0)

            if empty_database:
                print("Importing document records, passages and saved vectors...")
                cursor.executemany(
                    "INSERT INTO edi.documents (document_name) VALUES (%s)",
                    [(name,) for name in documents],
                )
                cursor.executemany("""
                    INSERT INTO edi.chunks (
                        chunk_id, document_name, page_number, chunk_text,
                        embedding_row, embedding_model, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                """, (
                    metadata_row(chunk) + (
                        json.dumps(vectors[index].tolist(), allow_nan=False),
                    )
                    for index, chunk in enumerate(chunks)
                ))
            else:
                print("Data already exists. Checking it without changing it...")

            print("Verifying stored text, source details and all embeddings...")
            verify_database(cursor, documents, chunks, vectors)

    # Reaching this point means the transaction finished successfully.
    return empty_database


def main():
    documents, chunks, vectors = load_collection()
    password = read_password()
    print(f"Source validated: {len(documents)} documents, {len(chunks)} chunks.")
    print(f"Embedding shape: {vectors.shape}")
    print("Connecting to local PostgreSQL on port 5433...")
    inserted = import_collection(documents, chunks, vectors, password)

    print("\nImport complete." if inserted else "\nExisting collection verified; no rows inserted.")
    print(f"Documents stored: {len(documents)}")
    print(f"Chunks stored: {len(chunks)}")
    print(f"Embedding dimensions: {DIMENSIONS}")
    print("Verification: all stored records match the saved collection.")
    print("Original files unchanged. No models called or embeddings regenerated.")


if __name__ == "__main__":
    try:
        main()
    except psycopg.Error as error:
        print(f"\nDatabase operation failed. SQLSTATE: {error.sqlstate or 'unavailable'}")
        print("Check Docker, the database setup and your local .env.db credentials.")
        print("Import success is not confirmed. Rerunning checks existing data first.")
        print("Do not share your database password.")
        raise SystemExit(1) from None
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"\nImport stopped: {error}")
        print("Existing rows were not overwritten. Source files were not changed.")
        raise SystemExit(1) from None