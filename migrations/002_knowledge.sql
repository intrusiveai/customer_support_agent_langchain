-- For new installations only. Reconcile an existing populated repository explicitly.
BEGIN;
CREATE TABLE knowledge.support_chunks (
    chunk_id uuid PRIMARY KEY,
    document_id text NOT NULL,
    chunk_index integer NOT NULL,
    title text NOT NULL,
    source_uri text NOT NULL,
    content text NOT NULL,
    content_digest text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL,
    embedding vector NOT NULL,
    updated_at timestamptz NOT NULL,
    UNIQUE (document_id, chunk_index),
    CONSTRAINT chunks_index CHECK (chunk_index >= 0),
    CONSTRAINT chunks_content CHECK (char_length(content) > 0),
    CONSTRAINT chunks_digest CHECK (content_digest ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT chunks_metadata_object CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT chunks_embedding_identity CHECK (
        (embedding_model = 'text-embedding-3-large' AND embedding_dimensions = 3072)
        OR (embedding_model = 'amazon.titan-embed-text-v2:0' AND embedding_dimensions = 1024)
    ),
    CONSTRAINT chunks_vector_dimensions CHECK (vector_dims(embedding) = embedding_dimensions)
);
COMMENT ON COLUMN knowledge.support_chunks.embedding IS
    'OpenAI text-embedding-3-large (3072) or Amazon Titan Text Embeddings V2 (1024); unmodified input text, no splitting or truncation; normalized float32';
COMMIT;
