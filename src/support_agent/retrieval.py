from threading import Lock

import numpy as np
from pgvector.psycopg import register_vector

from .config import EMBEDDING_MODEL
from .results import Rejected, digest, result


class LocalEmbedder:
    def __init__(self, settings):
        self.settings, self._model, self._lock = settings, None, Lock()

    def embed(self, query):
        # Lazy local load; identical pinned revision, no prompts, default tokenization.
        with self._lock:
            if self._model is None:
                if self.settings.embedding_local_path and (
                    self.settings.embedding_local_path.resolve().name
                    != self.settings.embedding_revision
                    or self.settings.embedding_local_path.resolve().parent.name != "snapshots"
                ):
                    raise Rejected(
                        "embedding_configuration",
                        "Local model path must name the configured revision in a Hugging Face snapshots directory.",
                    )
                from sentence_transformers import SentenceTransformer

                try:
                    self._model = SentenceTransformer(
                        str(self.settings.embedding_local_path)
                        if self.settings.embedding_local_path
                        else EMBEDDING_MODEL,
                        revision=self.settings.embedding_revision,
                        cache_folder=str(self.settings.embedding_cache_path)
                        if self.settings.embedding_cache_path
                        else None,
                        local_files_only=self.settings.embedding_local_files_only,
                        trust_remote_code=False,
                        device="cpu",
                    )
                except (OSError, ValueError) as exc:
                    raise Rejected(
                        "embedding_configuration",
                        "Cannot load the pinned local embedding model; check model path, revision and cache.",
                    ) from exc
            vector = self._model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                precision="float32",
                prompt="",
            )
        return np.asarray(vector, dtype=np.float32)


class Retrieval:
    def __init__(self, database, settings, embedder=None):
        self.db, self.settings = database, settings
        self.embedder = embedder or LocalEmbedder(settings)

    def search(self, ctx, query):
        settings = self.settings
        ctx.record(
            "retrieval_query",
            query=query,
            embedding_model=EMBEDDING_MODEL,
            embedding_revision=settings.embedding_revision,
            normalize_embeddings=True,
            preprocessing="unmodified text; default tokenizer/truncation; no prompt; float32",
        )
        with self.db.transaction(read_only=True) as tx:
            compatibility = tx.execute(
                """
                SELECT EXISTS (SELECT 1 FROM knowledge.support_chunks) AS populated,
                  EXISTS (SELECT 1 FROM knowledge.support_chunks WHERE embedding_model=%s
                          AND embedding_revision=%s) AS compatible
            """,
                (EMBEDDING_MODEL, settings.embedding_revision),
            ).fetchone()
        if not compatibility["populated"]:
            return result("no_matches", {"passages": []})
        if not compatibility["compatible"]:
            raise Rejected(
                "embedding_incompatible",
                "Repository has no passages for the configured model revision.",
            )
        vector = self.embedder.embed(query)
        if (
            vector.shape != (384,)
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
        ):
            raise Rejected(
                "embedding_incompatible",
                "Query embedding must be a normalized 384-dimensional vector.",
            )
        with self.db.transaction(read_only=True) as tx:
            register_vector(tx.connection)
            rows = tx.execute(
                """
                SELECT chunk_id, document_id, chunk_index, title, source_uri, content,
                       content_digest, metadata, embedding <=> %s AS distance,
                       vector_norm(embedding) AS embedding_norm
                FROM knowledge.support_chunks WHERE embedding_model = %s AND embedding_revision = %s
                ORDER BY embedding <=> %s, chunk_id LIMIT %s
            """,
                (
                    vector,
                    EMBEDDING_MODEL,
                    settings.embedding_revision,
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
