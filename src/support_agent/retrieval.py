from threading import Lock
import json

import numpy as np
import openai
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg import register_vector
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    ReadTimeoutError,
    NoCredentialsError,
    PartialCredentialsError,
)

from .config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    TITAN_EMBEDDING_DIMENSIONS,
    TITAN_EMBEDDING_MODEL,
)
from .aws import bedrock_client
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
                        base_url=self.settings.embedding_base_url or "https://api.openai.com/v1",
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


class TitanEmbedder:
    def __init__(self, settings):
        self.settings, self._client, self._lock = settings, None, Lock()

    def embed(self, text):
        """Invoke Titan V2 directly so query and document text stay byte-for-byte identical."""
        if not isinstance(text, str) or not text.strip():
            raise Rejected("embedding_input", "Embedding input must contain non-whitespace text.")
        if len(text) > 50000:
            raise Rejected("embedding_input", "Titan V2 input exceeds its 50,000-character limit.")
        with self._lock:
            if self._client is None:
                try:
                    self._client = bedrock_client(
                        self.settings,
                        timeout=self.settings.embedding_timeout_seconds,
                        max_retries=self.settings.embedding_max_retries,
                        endpoint_url=self.settings.embedding_base_url,
                    )
                except (BotoCoreError, ClientError, ValueError) as exc:
                    raise Rejected(
                        "embedding_configuration",
                        "Cannot configure Bedrock embeddings; check AWS region, profile and credentials.",
                    ) from exc
        try:
            response = self._client.invoke_model(
                modelId=TITAN_EMBEDDING_MODEL,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(
                    {
                        "inputText": text,
                        "dimensions": TITAN_EMBEDDING_DIMENSIONS,
                        "normalize": True,
                        "embeddingTypes": ["float"],
                    }
                ),
            )
            with response["body"] as body:
                data = json.loads(body.read())
            return validated_vector(data["embedding"], TITAN_EMBEDDING_DIMENSIONS)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {
                "AccessDeniedException",
                "UnrecognizedClientException",
                "ExpiredTokenException",
                "InvalidSignatureException",
                "ResourceNotFoundException",
            }:
                raise Rejected(
                    "embedding_configuration",
                    "Bedrock credentials, model access or model availability were rejected.",
                ) from exc
            if code in {
                "ThrottlingException",
                "ServiceQuotaExceededException",
                "ModelNotReadyException",
            }:
                raise Rejected(
                    "embedding_rate_limit",
                    "Bedrock embedding quota, rate limit or model readiness limit reached.",
                ) from exc
            if code == "ValidationException":
                raise Rejected(
                    "embedding_input",
                    "Bedrock rejected the embedding input; check passage length and API limits.",
                ) from exc
            if code == "ModelTimeoutException":
                raise Rejected("embedding_timeout", "Bedrock embedding request timed out.") from exc
            raise Rejected(
                "embedding_service", "Bedrock embedding service request failed."
            ) from exc
        except (ConnectTimeoutError, ReadTimeoutError) as exc:
            raise Rejected("embedding_timeout", "Bedrock embedding request timed out.") from exc
        except (NoCredentialsError, PartialCredentialsError) as exc:
            raise Rejected(
                "embedding_configuration", "AWS credentials are missing or incomplete."
            ) from exc
        except BotoCoreError as exc:
            raise Rejected("embedding_service", "Bedrock embedding transport failed.") from exc
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise Rejected(
                "embedding_response", "Bedrock returned an invalid embedding response."
            ) from exc


def build_embedder(settings):
    return (
        TitanEmbedder(settings)
        if settings.embedding_provider == "bedrock"
        else OpenAIEmbedder(settings)
    )


def validated_vector(value, dimensions=EMBEDDING_DIMENSIONS):
    vector = np.asarray(value, dtype=np.float32)
    if (
        vector.shape != (dimensions,)
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
    ):
        raise Rejected(
            "embedding_incompatible",
            f"Embedding must be a normalized {dimensions}-dimensional vector.",
        )
    return vector


class Retrieval:
    def __init__(self, database, settings, embedder=None):
        self.db, self.settings = database, settings
        self.embedder = embedder or build_embedder(settings)

    def search(self, ctx, query):
        settings = self.settings
        ctx.record(
            "retrieval_query",
            query=query,
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
            embedding_dimensions=settings.embedding_dimensions,
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
                (settings.embedding_model, settings.embedding_dimensions),
            ).fetchone()
        if not compatibility["populated"]:
            return result("no_matches", {"passages": []})
        if not compatibility["compatible"]:
            raise Rejected(
                "embedding_incompatible",
                "Repository has no passages for the selected embedding model and dimensions.",
            )
        vector = validated_vector(self.embedder.embed(query), settings.embedding_dimensions)
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
                    settings.embedding_model,
                    settings.embedding_dimensions,
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
