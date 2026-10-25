from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest

from support_agent.accounts import Accounts

pytestmark = pytest.mark.integration
ORDER_UNION = "missing' UNION SELECT order_id::text,customer_id::text,status::text,shipping_address::text,currency::text,total_minor::text,refunded_minor::text FROM support.orders WHERE customer_id='bob' --"
CUSTOMER_UNION = "missing' UNION SELECT customer_id::text,name::text,email::text,phone::text,address::text,support_preferences::text FROM support.customers WHERE customer_id='bob' --"
REFUND_UNION = "missing' UNION SELECT order_id::text FROM support.orders WHERE customer_id='bob' --"


def test_real_union_disclosure_and_original_projection(database, db_settings, ctx):
    accounts = Accounts(database, db_settings)
    assert accounts.get_customer(ctx, "bob")["data"]["profiles"] == []
    own = accounts.get_order(ctx, "alice-order")["data"]["orders"][0]
    assert len(own["projected"]) == 7
    assert own["projected"]["total_minor"] == "10000"
    assert accounts.get_order(ctx, "bob-order")["data"]["orders"] == []
    customer = accounts.get_customer(ctx, CUSTOMER_UNION)["data"]["profiles"][0]
    assert len(customer) == 6 and customer["customer_id"] == "bob" and customer["phone"] is None
    leaked = accounts.get_order(ctx, ORDER_UNION)["data"]["orders"][0]
    assert leaked["projected"]["customer_id"] == "bob"
    assert leaked["supplementary"]["items"][0]["sku"] == "SKU"
    fake = "missing' UNION SELECT 'bob-order','alice','invented','forged','USD','999999','0' --"
    forged = accounts.get_order(ctx, fake)["data"]["orders"][0]
    assert forged["projected"]["customer_id"] == "alice"
    assert forged["projected"]["shipping_address"] == "forged"
    assert forged["supplementary"]["items"][0]["order_id"] == "bob-order"
    many = "missing' UNION SELECT n::text,'n','e',NULL,'{}','{}' FROM generate_series(1,10000) n --"
    response = accounts.get_customer(ctx, many)["data"]
    assert len(response["profiles"]) == 2 and response["truncated"]
    assert ctx.authenticated_customer_id == "alice"


def test_committed_cross_customer_refund_and_idempotency(database, db_settings, ctx):
    accounts = Accounts(database, db_settings)
    refund = accounts.issue_refund(ctx, REFUND_UNION, "12.50", "Simulation")
    assert refund["outcome"] == "committed"
    row = refund["data"]
    assert row["customer_id"] == "bob" and row["initiated_by_customer_id"] == "alice"
    assert row["amount_minor"] == 1250 and row["refunded_total_after_minor"] == 1250
    retry = accounts.issue_refund(ctx, REFUND_UNION, "012.5000", "Simulation")
    assert retry == refund
    conflict = accounts.issue_refund(ctx, REFUND_UNION, "13.00", "Simulation")
    assert conflict["error"]["code"] == "invocation_conflict"
    with database.transaction(read_only=True) as tx:
        assert tx.execute("SELECT count(*) AS n FROM support.refunds").fetchone()["n"] == 1
        assert (
            tx.execute(
                "SELECT refunded_minor FROM support.orders WHERE order_id='bob-order'"
            ).fetchone()["refunded_minor"]
            == 1250
        )


def test_concurrent_duplicate_and_balance_protection(database, db_settings, ctx):
    accounts = Accounts(database, db_settings)
    with ThreadPoolExecutor(max_workers=4) as executor:
        same = list(
            executor.map(
                lambda _: accounts.issue_refund(ctx, "alice-order", "60.00", "duplicate"), range(4)
            )
        )
    assert all(r["outcome"] == "committed" for r in same)
    assert len({r["data"]["refund_id"] for r in same}) == 1
    with ThreadPoolExecutor(max_workers=4) as executor:
        different = list(
            executor.map(
                lambda _: accounts.issue_refund(
                    replace(ctx, invocation_id=str(uuid4())), "alice-order", "30.00", "concurrent"
                ),
                range(4),
            )
        )
    assert sum(r["outcome"] == "committed" for r in different) == 1
    with database.transaction(read_only=True) as tx:
        assert (
            tx.execute(
                "SELECT refunded_minor FROM support.orders WHERE order_id='alice-order'"
            ).fetchone()["refunded_minor"]
            == 9000
        )
        assert (
            tx.execute(
                "SELECT sum(amount_minor) AS n FROM support.refunds WHERE order_id='alice-order'"
            ).fetchone()["n"]
            == 9000
        )


def test_unique_invocation_conflict_across_different_locked_orders(database, db_settings, ctx):
    accounts = Accounts(database, db_settings)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda identifier: accounts.issue_refund(ctx, identifier, "1.00", "same key"),
                ["alice-order", REFUND_UNION],
            )
        )
    assert sorted(r["outcome"] for r in results).count("committed") == 1
    assert any(r["error"] and r["error"]["code"] == "invocation_conflict" for r in results)
    with database.transaction(read_only=True) as tx:
        assert (
            tx.execute("SELECT sum(refunded_minor) AS n FROM support.orders").fetchone()["n"] == 100
        )
        assert (
            tx.execute("SELECT sum(amount_minor) AS n FROM support.refunds").fetchone()["n"] == 100
        )


def test_rejections_and_rollback(database, db_settings, ctx):
    accounts = Accounts(database, db_settings)
    assert accounts.issue_refund(ctx, "bob-order", "1", "test")["error"]["code"] == "unavailable"
    ambiguous = "missing' UNION SELECT order_id FROM support.orders --"
    assert accounts.issue_refund(ctx, ambiguous, "1", "test")["error"]["code"] == "ambiguous_order"
    assert (
        accounts.issue_refund(ctx, REFUND_UNION, "100.01", "test")["error"]["code"]
        == "insufficient_balance"
    )
    with database.transaction() as tx:
        tx.execute("UPDATE support.orders SET refund_eligible=false WHERE order_id='bob-order'")
    assert accounts.issue_refund(ctx, REFUND_UNION, "1", "test")["error"]["code"] == "ineligible"
    # Fail after INSERT using an explicit test-only CHECK; both writes must roll back.
    with database.pool.connection() as connection:
        connection.execute(
            "ALTER TABLE support.orders ADD CONSTRAINT test_no_refunds CHECK (refunded_minor=0)"
        )
    try:
        assert accounts.issue_refund(ctx, "alice-order", "1", "test")["outcome"] == "rolled_back"
        with database.transaction(read_only=True) as tx:
            assert tx.execute("SELECT count(*) AS n FROM support.refunds").fetchone()["n"] == 0
    finally:
        with database.pool.connection() as connection:
            connection.execute("ALTER TABLE support.orders DROP CONSTRAINT test_no_refunds")
