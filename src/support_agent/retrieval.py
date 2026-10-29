from threading import Lock

import numpy as np
import openai
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg import register_vector

from .config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from .results import Rejected, digest, result


class OpenAIEmbedder:
    def __init__(self, settings):
        self.settings, self._client, self._lock = settings, None, Lock()

    def embed(self, text):
        """Embed exact text remotely; use this same method for stored document passages."""
        if not isinstance(text, str) or not text.strip():
            raise Rejected("embedding_input", "Embedding input must contain non-whitespace text.")
        with self._lock:
            if self._client is None:
                if (
                    not self.settings.embedding_api_key
                    or not self.settings.embedding_api_key.get_secret_value()
                ):
                    raise Rejected(
                        "embedding_configuration",
                        "Set SUPPORT_EMBEDDING_API_KEY or OPENAI_API_KEY for OpenAI embeddings.",
                    )
                try:
                    self._client = OpenAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        dimensions=EMBEDDING_DIMENSIONS,
                        api_key=self.settings.embedding_api_key,
                        base_url=self.settings.embedding_base_url,
                        request_timeout=self.settings.embedding_timeout_seconds,
                        max_retries=self.settings.embedding_max_retries,
                        check_embedding_ctx_length=False,
                        model_kwargs={"encoding_format": "float"},
                    )
                except (ValueError, openai.OpenAIError) as exc:
                    raise Rejected(
                        "embedding_configuration",
                        "Cannot configure the OpenAI embedding client; check credentials and endpoint.",
                    ) from exc
        try:
            return validated_vector(self._client.embed_query(text))
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise Rejected(
                "embedding_configuration",
                "OpenAI embedding credentials or model access were rejected.",
            ) from exc
        except openai.RateLimitError as exc:
            raise Rejected(
                "embedding_rate_limit", "OpenAI embedding quota or rate limit reached."
            ) from exc
        except openai.APITimeoutError as exc:
            raise Rejected("embedding_timeout", "OpenAI embedding request timed out.") from exc
        except openai.BadRequestError as exc:
            raise Rejected(
                "embedding_input",
                "OpenAI rejected the embedding input; check passage length and API limits.",
            ) from exc
        except openai.APIError as exc:
            raise Rejected("embedding_service", "OpenAI embedding service request failed.") from exc
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise Rejected(
                "embedding_response", "OpenAI returned an invalid embedding response."
            ) from exc


def validated_vector(value):
    vector = np.asarray(value, dtype=np.float32)
    if (
        vector.shape != (EMBEDDING_DIMENSIONS,)
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
    ):
        raise Rejected(
            "embedding_incompatible",
            f"Embedding must be a normalized {EMBEDDING_DIMENSIONS}-dimensional vector.",
        )
    return vector


class Retrieval:
    def __init__(self, database, settings, embedder=None):
        self.db, self.settings = database, settings
        self.embedder = embedder or OpenAIEmbedder(settings)

    def search(self, ctx, query):
        settings = self.settings
        ctx.record(
            "retrieval_query",
            query=query,
            embedding_model=EMBEDDING_MODEL,
            embedding_provider="openai",
            embedding_dimensions=EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            preprocessing="unmodified text; no splitting or truncation; normalized float32",
        )
        with self.db.transaction(read_only=True) as tx:
            compatibility = tx.execute(
                """
                SELECT EXISTS (SELECT 1 FROM knowledge.support_chunks) AS populated,
                  EXISTS (SELECT 1 FROM knowledge.support_chunks WHERE embedding_model=%s
                          AND embedding_dimensions=%s) AS compatible
            """,
                (EMBEDDING_MODEL, EMBEDDING_DIMENSIONS),
            ).fetchone()
        if not compatibility["populated"]:
            return result("no_matches", {"passages": []})
        if not compatibility["compatible"]:
            raise Rejected(
                "embedding_incompatible",
                "Repository has no passages for the required OpenAI model and dimensions.",
            )
        vector = validated_vector(self.embedder.embed(query))
        with self.db.transaction(read_only=True) as tx:
            register_vector(tx.connection)
            rows = tx.execute(
                """
                SELECT chunk_id, document_id, chunk_index, title, source_uri, content,
                       content_digest, metadata, embedding <=> %s AS distance,
                       vector_norm(embedding) AS embedding_norm
                FROM knowledge.support_chunks WHERE embedding_model = %s AND embedding_dimensions = %s
                ORDER BY embedding <=> %s, chunk_id LIMIT %s
            """,
                (
                    vector,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIMENSIONS,
                    vector,
                    settings.retrieval_count,
                ),
            ).fetchall()
        passages = []
        for row in rows:
            if not np.isclose(row.pop("embedding_norm"), 1, atol=1e-4):
                raise Rejected(
                    "embedding_incompatible", "Retrieved document embedding is not normalized."
                )
            original = row["content"]
            if row["content_digest"] != digest(original.encode("utf-8")):
                raise Rejected(
                    "content_digest_mismatch",
                    "Stored passage digest does not match its exact UTF-8 content.",
                )
            content = original[: settings.passage_char_limit]
            passages.append(
                {
                    **row,
                    "reference": f"doc:{row['chunk_id']}",
                    "content": content,
                    "truncated": len(content) < len(original),
                    "delivered_content_digest": digest(content.encode("utf-8")),
                }
            )
        return result(
            "ok" if passages else "no_matches",
            {
                "passages": passages,
                "score_meaning": "Cosine distance; lower ranks nearer. Not factual confidence.",
            },
        )
