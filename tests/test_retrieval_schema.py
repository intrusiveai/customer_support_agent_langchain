import pytest

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
    with database.pool.connection() as connection:
        connection.execute("ALTER TABLE knowledge.support_chunks DROP CONSTRAINT chunks_model")
        connection.execute("UPDATE knowledge.support_chunks SET embedding_model='incompatible'")
        try:
            with pytest.raises(Rejected, match="no passages"):
                retrieval.search(ctx, "refund")
        finally:
            connection.execute(
                "UPDATE knowledge.support_chunks SET embedding_model='text-embedding-3-large'"
            )
            connection.execute(
                "ALTER TABLE knowledge.support_chunks ADD CONSTRAINT chunks_model CHECK (embedding_model='text-embedding-3-large')"
            )
    with database.transaction() as tx:
        tx.execute("UPDATE knowledge.support_chunks SET content='changed without updating digest'")
    with pytest.raises(Rejected, match="digest"):
        retrieval.search(ctx, "refund")
    with database.transaction() as tx:
        tx.execute("DELETE FROM knowledge.support_chunks")
    assert retrieval.search(ctx, "refund")["outcome"] == "no_matches"


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
