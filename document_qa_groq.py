"""Neon hybrid retrieval with hosted answers; existing local scripts stay unchanged."""

import argparse
import json
import os
import re
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg
from dotenv import dotenv_values

from document_qa import NO_EVIDENCE, SYSTEM_MESSAGE
from document_qa_postgres import PostgresDocumentQA
from neon_database import connect_neon_database

MODEL_NAME = "openai/gpt-oss-20b"
MAX_COMPLETION_TOKENS = 1024
MAX_PROMPT_BYTES = 18000  # A request-size guard, NOT a token count.

SYSTEM_MESSAGE = SYSTEM_MESSAGE + (
    " When stating a requirement, include any condition that determines whether "
    "it applies, such as a risk assessment, score, partner type or exception. "
    "Do not turn 'may be required' or a conditional requirement into a rule "
    "that applies to everyone. Keeping the answer short must not remove "
    "conditions that change its meaning."
)


def read_groq_key():
    settings = dotenv_values(
        Path(__file__).with_name(".env.app"),
        encoding="utf-8-sig", interpolate=False,
    )
    key = os.environ.get("GROQ_API_KEY", settings.get("GROQ_API_KEY") or "").strip()
    if not key or key == "PASTE_YOUR_NEW_KEY_HERE":
        raise ValueError("Add your private GROQ_API_KEY to .env.app.")
    if not key.isascii() or any(char.isspace() for char in key):
        raise ValueError("The Groq key contains unexpected characters. Copy it again privately.")
    return key


def request_groq(messages, api_key):
    body = {
        "model": MODEL_NAME, "messages": messages,
        "temperature": 0, "stream": False,
        "reasoning_effort": "low", "include_reasoning": False,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "EnterpriseDocumentIntelligence/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        explanations = {
            401: "Check your Groq key privately.",
            403: "Check model access in your Groq account; do not upgrade.",
            413: "The request is too large for the current allowance. Stop and share this status.",
            429: "A usage limit was reached. Wait for it to reset; do not upgrade.",
        }
        hint = explanations.get(error.code, "Share this status code, not your API key.")
        raise RuntimeError(f"Groq HTTP {error.code}. {hint}") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Groq connection failed or timed out. No automatic retry was made.") from None
    except ValueError:
        raise RuntimeError("Groq returned unreadable JSON.") from None


class GroqDocumentQA(PostgresDocumentQA):
    def __init__(self):
        self._groq_key = read_groq_key()
        super().__init__(connection_factory=connect_neon_database)

    def ask(self, question, document=None):
        with self._lock:
            started = perf_counter()
            search = self.search(question, document)
            sources = list(search["sources"])
            dropped = []

            while sources:
                evidence = [
                    {key: source[key] for key in ("id", "document", "pdf_page", "text")}
                    for source in sources
                ]
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": json.dumps({
                        "question": search["question"],
                        "selected_document": document,
                        "sources": evidence,
                    }, ensure_ascii=False)},
                ]
                size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
                if size <= MAX_PROMPT_BYTES:
                    break
                dropped.append(sources.pop()["chunk_id"])
            if not sources:
                raise ValueError("No complete passage fits the request-size limit.")

            answer_started = perf_counter()
            response = request_groq(messages, self._groq_key)
            answer_seconds = perf_counter() - answer_started
            try:
                choice = response["choices"][0]
                answer = choice["message"]["content"]
                finish_reason = choice["finish_reason"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError("Groq returned an unexpected answer format.") from None
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("Groq returned no answer text. Stop and share this message.")
            if finish_reason not in {"stop", "length"}:
                raise RuntimeError("Groq did not finish with a normal text answer.")
            answer = answer.strip()

            cited = set(re.findall(r"\[\s*(S[0-9]+)\s*\]", answer))
            allowed = {source["id"] for source in sources}
            warnings = ["Citation-label checks do not verify that sources support each claim."]
            if cited - allowed:
                warnings.append("Unknown source labels: " + ", ".join(sorted(cited - allowed)))
            if not cited and answer != NO_EVIDENCE:
                warnings.append("The answer has no recognised source citations.")
            if dropped:
                warnings.append(f"Omitted {len(dropped)} whole passages to limit request size.")
            if finish_reason == "length":
                warnings.append("The answer reached its output limit and may be incomplete.")
            usage = response.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}

            return {
                "question": search["question"], "document": document,
                "answer": answer, "sources": sources, "warnings": warnings,
                "refused": answer == NO_EVIDENCE,
                "retrieved_count": len(search["sources"]), "dropped_chunk_ids": dropped,
                "prompt_tokens_estimated": None, "prompt_bytes": size,
                "prompt_tokens_actual": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "retrieval_seconds": search["retrieval_seconds"],
                "answer_seconds": answer_seconds,
                "total_seconds": perf_counter() - started,
                "answer_model": MODEL_NAME, "provider": "Groq",
            }


def main():
    parser = argparse.ArgumentParser(description="Ask about Neon documents using Groq.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--document", default="UL08_Audit_Requirements.pdf")
    scope.add_argument("--all-documents", action="store_true")
    args = parser.parse_args()
    document = None if args.all_documents else args.document
    print("Loading Neon passages and cached BGE; no Qwen model or tokenizer is loaded.")
    engine = GroqDocumentQA()
    print(f"Documents: {len(engine.documents)} | Passages: {len(engine.chunks)}")
    print(f"Search scope: {document or 'Whole collection'}")
    print(f"Hosted answering model: {MODEL_NAME}")
    print("Your question and selected source passages will be sent to Groq.")
    question = input("\nEnter your question: ").strip()
    print("Searching Neon, then requesting an answer from Groq...", flush=True)
    result = engine.ask(question, document)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nPassages retrieved: {result['retrieved_count']}")
    print(f"Passages supplied: {len(result['sources'])}")
    print(f"Retrieval: {result['retrieval_seconds']:.3f}s")
    print(f"Groq request: {result['answer_seconds']:.2f}s")
    print(f"Total excluding startup: {result['total_seconds']:.2f}s")
    print(f"Groq-reported prompt tokens: {result['prompt_tokens_actual']}")
    print("\nSources supplied:")
    for source in result["sources"]:
        print(f"[{source['id']}] {source['document']} | PDF page {source['pdf_page']}")
    for warning in result["warnings"]:
        print(f"Note: {warning}")
    print("No database changes or passage re-embedding. Streamlit is unchanged.")


if __name__ == "__main__":
    try:
        main()
    except psycopg.Error as error:
        print(f"Database request failed. SQLSTATE: {error.sqlstate or 'unavailable'}")
        raise SystemExit(1) from None
    except (ValueError, RuntimeError) as error:
        print(f"Stopped: {error}")
        raise SystemExit(1) from None
    except OSError:
        print("Setup or network error. Check local files and connectivity; keep credentials private.")
        raise SystemExit(1) from None
