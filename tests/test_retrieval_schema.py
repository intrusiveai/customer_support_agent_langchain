import pytest
import numpy as np
from pgvector.psycopg import register_vector

from support_agent.results import Rejected
from support_agent.retrieval import Retrieval
from support_agent.schema import SchemaError, check_schema

from conftest import UnitEmbedder

pytestmark = pytest.mark.integration


def test_retrieval_truncation_digest_model_and_empty(database, db_settings, ctx):
    limited = db_settings.model_copy(update={"passage_char_limit": 12})
    retrieval = Retrieval(database, limited, UnitEmbedder())
    row = retrieval.search(ctx, "refund")["data"]["passages"][0]
    assert row["truncated"] and len(row["content"]) == 12
    assert row["content_digest"] != row["delivered_content_digest"]
    assert row["source_uri"].endswith("/refunds") and row["reference"].startswith("doc:")
    other_provider = Retrieval(
        database, limited.model_copy(update={"embedding_provider": "bedrock"})
    )
    with pytest.raises(Rejected, match="no passages"):
        other_provider.search(ctx, "refund")
    with database.transaction() as tx:
        tx.execute("UPDATE knowledge.support_chunks SET content='changed without updating digest'")
    with pytest.raises(Rejected, match="digest"):
        retrieval.search(ctx, "refund")
    with database.transaction() as tx:
        tx.execute("DELETE FROM knowledge.support_chunks")
    assert retrieval.search(ctx, "refund")["outcome"] == "no_matches"


def test_stored_nonunit_vectors_are_rejected_without_rewriting(database, db_settings, ctx):
    vector = UnitEmbedder().embed("") * np.float32(1.0004726255)
    with database.pool.connection() as connection:
        register_vector(connection)
        connection.execute("UPDATE knowledge.support_chunks SET embedding=%s", (vector,))
    with pytest.raises(Rejected, match="measured norm=") as error:
        Retrieval(database, db_settings, UnitEmbedder()).search(ctx, "refund")
    assert error.value.code == "embedding_incompatible"
    with database.pool.connection() as connection:
        register_vector(connection)
        stored = connection.execute("SELECT embedding FROM knowledge.support_chunks").fetchone()
    np.testing.assert_array_equal(stored[0], vector)


def test_startup_detects_missing_constraints_types_and_indexes(database, db_settings):
    with database.pool.connection() as connection:
        # Roll back intentional test schema damage.
        with connection.transaction(force_rollback=True):
            connection.execute("ALTER TABLE support.orders DROP CONSTRAINT orders_refunded")
            connection.execute("ALTER TABLE support.customers ALTER COLUMN name DROP NOT NULL")
            connection.execute("DROP INDEX support.orders_customer_order_idx")
            with pytest.raises(SchemaError) as error:
                check_schema(connection, db_settings)
            text = str(error.value)
            assert "support.orders" in text and "refunded_minor" in text
            assert "support.customers.name" in text and "missing index" in text
        assert check_schema(connection, db_settings) == []


def test_failed_commit_acknowledgement_is_uncertain(database, db_settings, ctx, monkeypatch):
    from contextlib import contextmanager
    import psycopg
    from support_agent.accounts import Accounts

    original = database.transaction

    @contextmanager
    def lose_acknowledgement(*args, **kwargs):
        with original(*args, **kwargs) as tx:
            yield tx
        raise psycopg.OperationalError("simulated connection loss after commit")

    monkeypatch.setattr(database, "transaction", lose_acknowledgement)
    result = Accounts(database, db_settings).issue_refund(
        ctx, "alice-order", "1.00", "uncertainty test"
    )
    assert result["outcome"] == "uncertain"
    monkeypatch.setattr(database, "transaction", original)
    # It really committed; acknowledgement loss must never be described as rollback.
    replay = Accounts(database, db_settings).issue_refund(
        ctx, "alice-order", "1.00", "uncertainty test"
    )
    assert replay["outcome"] == "committed"
    with database.transaction(read_only=True) as tx:
        assert tx.execute("SELECT count(*) AS n FROM support.refunds").fetchone()["n"] == 1
