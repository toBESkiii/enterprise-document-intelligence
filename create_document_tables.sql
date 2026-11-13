-- Structure for the existing BGE collection. No document data is imported.
BEGIN;

CREATE SCHEMA IF NOT EXISTS edi;

CREATE TABLE IF NOT EXISTS edi.documents (
    document_name TEXT PRIMARY KEY CHECK (btrim(document_name) <> ''),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edi.chunks (
    chunk_id TEXT PRIMARY KEY CHECK (btrim(chunk_id) <> ''),
    document_name TEXT NOT NULL REFERENCES edi.documents(document_name),
    page_number INTEGER NOT NULL CHECK (page_number >= 1),
    chunk_text TEXT NOT NULL CHECK (btrim(chunk_text) <> ''),
    embedding_row INTEGER NOT NULL UNIQUE CHECK (embedding_row >= 0),
    embedding_model TEXT NOT NULL
        CHECK (embedding_model = 'BAAI/bge-small-en-v1.5'),
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_document_page_idx
    ON edi.chunks (document_name, page_number);

COMMIT;

SELECT 'documents' AS table_name, COUNT(*) AS rows_stored
FROM edi.documents
UNION ALL
SELECT 'chunks', COUNT(*)
FROM edi.chunks;