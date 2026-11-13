"""Database-backed hybrid Q&A; reuse answer handling from document_qa.py.

This version uses the fixed imported collection. Restart it after database edits.
"""

import argparse
import json
import os
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

import document_qa as base


def connect_database():
    env_path = Path(__file__).resolve().parent / ".env.db"
    if not env_path.is_file():
        raise ValueError("The existing .env.db file is missing.")
    settings = dotenv_values(env_path, interpolate=False, encoding="utf-8-sig")
    password = settings.get("EDI_DB_PASSWORD")
    if not password or not password.strip():
        raise ValueError("EDI_DB_PASSWORD is missing or empty in .env.db.")
    return psycopg.connect(
        host="127.0.0.1", port=5433,
        dbname="document_intelligence", user="edi_admin",
        password=password, connect_timeout=10, autocommit=True,
        options="-c default_transaction_read_only=on -c statement_timeout=15000",
        row_factory=dict_row, application_name="edi_hybrid_qa",
    )


class PostgresDocumentQA(base.DocumentQA):
    def __init__(self):
        # Do not call the file-based constructor: load passage text from SQL.
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT chunk_id, document_name, page_number, chunk_text,
                           embedding_row, embedding_model,
                           vector_dims(embedding) AS dimensions
                    FROM edi.chunks ORDER BY embedding_row
                """)
                self.chunks = cursor.fetchall()
        if not self.chunks:
            raise ValueError("The database collection is empty. Import it first.")
        if any(
            c["embedding_row"] != i or c["dimensions"] != 384
            or c["embedding_model"] != base.EMBEDDING_MODEL
            for i, c in enumerate(self.chunks)
        ):
            raise ValueError("Database rows or embedding settings do not match this collection.")
        self.document_names = np.array([c["document_name"] for c in self.chunks])
        self.documents = tuple(sorted(set(self.document_names.tolist())))
        self.keep_length = np.array([
            len(c["chunk_text"].split()) >= base.MIN_WORDS for c in self.chunks
        ])
        tokens = [base.tokenise(c["chunk_text"]) for c in self.chunks]
        if not any(tokens):
            raise ValueError("The collection contains no searchable terms.")
        self.token_sets = [set(t) for t in tokens]
        # Keep BM25 statistics over the same full collection as before.
        self.bm25 = base.BM25Okapi(tokens, k1=1.5, b=0.75)
        self.model = base.SentenceTransformer(
            base.EMBEDDING_MODEL, device="cpu", local_files_only=True,
        )
        self.tokenizer = None
        self.ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
        self._lock = RLock()

    def _vector_ranking(self, vector, document, eligible):
        eligible_rows = np.flatnonzero(eligible).tolist()
        with connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT embedding_row, chunk_id,
                           1 - (embedding <=> %(vector)s::vector) AS similarity
                    FROM edi.chunks
                    WHERE (%(document)s::text IS NULL OR document_name = %(document)s)
                      AND embedding_model = %(model)s
                      AND embedding_row = ANY(%(rows)s::integer[])
                    ORDER BY embedding <=> %(vector)s::vector, embedding_row
                """, {
                    "vector": json.dumps(vector.tolist(), allow_nan=False),
                    "document": document,
                    "model": base.EMBEDDING_MODEL,
                    "rows": eligible_rows,
                })
                hits = cursor.fetchall()
        if {h["embedding_row"] for h in hits} != set(eligible_rows):
            raise ValueError("Database collection changed. Restart the application.")
        scores = np.full(len(self.chunks), np.nan)
        ranking = []
        for hit in hits:
            row = hit["embedding_row"]
            if hit["chunk_id"] != self.chunks[row]["chunk_id"]:
                raise ValueError("Database chunk IDs changed. Restart the application.")
            scores[row] = hit["similarity"]
            ranking.append(row)
        if not np.isfinite(scores[eligible]).all():
            raise ValueError("Database search returned invalid similarities.")
        return np.asarray(ranking, dtype=int), scores

    def search(self, question, document):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Please enter a question.")
        question = question.strip()
        if len(question) > 2000:
            raise ValueError("Please keep the question below 2,001 characters.")
        if document is not None and document not in self.documents:
            raise ValueError("Please select a document from this collection.")
        with self._lock:
            started = perf_counter()
            eligible = self.keep_length.copy()
            if document is not None:
                eligible &= self.document_names == document
            if not eligible.any():
                raise ValueError("This document has no eligible passages.")
            query = base.QUERY_PREFIX + question
            ids = self.model.tokenizer.encode(
                query, add_special_tokens=True, truncation=False,
            )
            if len(ids) > self.model.max_seq_length:
                raise ValueError("The question exceeds BGE's token limit. Please shorten it.")
            vector = self.model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            if vector.shape != (384,) or not np.isfinite(vector).all():
                raise ValueError("Invalid question embedding.")
            if np.linalg.norm(vector) == 0:
                raise ValueError("The question embedding has zero length.")
            vector_rows, vector_scores = self._vector_ranking(vector, document, eligible)

            terms = list(dict.fromkeys(base.tokenise(question)))
            keyword_scores = self.bm25.get_scores(terms)
            matches = np.array([bool(set(terms) & t) for t in self.token_sets])
            if not np.isfinite(keyword_scores).all():
                raise ValueError("Keyword search returned invalid scores.")
            keyword_rows = np.argsort(keyword_scores)[::-1]
            keyword_rows = keyword_rows[(eligible & matches)[keyword_rows]]

            rrf = np.zeros(len(self.chunks), dtype=float)
            for ranking in (vector_rows, keyword_rows):
                for rank, row in enumerate(ranking[:base.FUSION_WINDOW], 1):
                    rrf[row] += 1.0 / (base.RRF_CONSTANT + rank)
            pool = np.flatnonzero(rrf > 0)
            rows = pool[np.lexsort((pool, -rrf[pool]))][:base.TOP_K]
            sources, scores = [], []
            for rank, row in enumerate(rows, 1):
                chunk = self.chunks[int(row)]
                sources.append({
                    "id": f"S{rank}", "document": chunk["document_name"],
                    "pdf_page": chunk["page_number"], "chunk_id": chunk["chunk_id"],
                    "text": chunk["chunk_text"],
                })
                scores.append({
                    "chunk_id": chunk["chunk_id"],
                    "similarity": float(vector_scores[row]), "rrf": float(rrf[row]),
                })
            return {
                "question": question, "document": document, "sources": sources,
                "scores": scores, "retrieval_seconds": perf_counter() - started,
            }

    def ask(self, question, document):
        with self._lock:
            if self.tokenizer is None:
                self.tokenizer = base.AutoTokenizer.from_pretrained(
                    base.ANSWER_TOKENIZER, local_files_only=True,
                )
            # The existing answer method calls this class's database search.
            return super().ask(question, document)


def main():
    parser = argparse.ArgumentParser(description="PostgreSQL hybrid document Q&A.")
    parser.add_argument("--answer", action="store_true", help="Also ask Qwen locally.")
    parser.add_argument("--document", default="UL08_Audit_Requirements.pdf")
    args = parser.parse_args()
    print("Loading passage text from PostgreSQL and the cached BGE model...")
    engine = PostgresDocumentQA()
    if args.document not in engine.documents:
        raise ValueError("Unknown document. Available: " + ", ".join(engine.documents))
    print(f"Documents: {len(engine.documents)} | Passages: {len(engine.chunks)}")
    print(f"Selected document: {args.document}")
    print("Hybrid: pgvector + BM25 + RRF; up to 50 per method, final top 5.")
    question = input("\nEnter your question: ").strip()

    if args.answer:
        print("Searching, then asking Qwen locally. CPU answering may take a while.", flush=True)
        result = engine.ask(question, args.document)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nPassages retrieved: {result['retrieved_count']}")
        print(f"Passages supplied to Qwen: {len(result['sources'])}")
        print(f"Prompt tokens: estimated {result['prompt_tokens_estimated']}; "
              f"Ollama reported {result['prompt_tokens_actual']}")
        print(f"Local answering request: {result['answer_seconds']:.1f}s")
        print(f"Total excluding startup: {result['total_seconds']:.1f}s")
        for warning in result["warnings"]:
            print(f"Note: {warning}")
        print("\nSources actually supplied to Qwen:")
    else:
        result = engine.search(question, args.document)
        print("\nSearch only: no Qwen request or Qwen tokenizer load.")
        print("Retrieved sources:")
    for source in result["sources"]:
        print(f"[{source['id']}] {source['document']} | PDF page {source['pdf_page']}")
        print(f"     Chunk: {source['chunk_id']}")
    if not args.answer:
        print(f"\nTop passage:\n{result['sources'][0]['text']}")
    print(f"Retrieval time: {result['retrieval_seconds']:.3f}s")
    print("Database and original files unchanged. No passage embeddings regenerated.")


if __name__ == "__main__":
    try:
        main()
    except psycopg.Error as error:
        print(f"\nDatabase error. SQLSTATE: {error.sqlstate or 'unavailable'}")
        print("Check Docker and your local .env.db settings. Do not share your password.")
        raise SystemExit(1) from None
    except (OSError, ValueError, RuntimeError) as error:
        print(f"\nStopped: {error}")
        raise SystemExit(1) from None