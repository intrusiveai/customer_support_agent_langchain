"""Read-only compatibility checks. Equivalent constraint names are accepted."""

import re

from .config import EMBEDDING_MODEL

# '?' marks the one nullable column. All others must be NOT NULL.
COLUMNS = {
    "support.customers": "customer_id:text name:text email:text phone:text? address:jsonb support_preferences:jsonb created_at:timestamptz",
    "support.orders": "order_id:text customer_id:text status:text currency:text total_minor:bigint refunded_minor:bigint refund_eligible:boolean shipping_address:jsonb placed_at:timestamptz",
    "support.order_items": "order_id:text line_number:integer sku:text description:text quantity:integer unit_price_minor:bigint",
    "support.refunds": "refund_id:uuid order_id:text customer_id:text initiated_by_customer_id:text session_id:text invocation_id:text request_digest:text amount_minor:bigint currency:text reason:text refunded_total_after_minor:bigint created_at:timestamptz",
    "knowledge.support_chunks": "chunk_id:uuid document_id:text chunk_index:integer title:text source_uri:text content:text content_digest:text metadata:jsonb embedding_model:text embedding_revision:text embedding:vector(384) updated_at:timestamptz",
}
CONSTRAINTS = {
    "support.customers": [
        "PRIMARY KEY (customer_id)",
        "CHECK (char_length(customer_id) >= 1 AND char_length(customer_id) <= 128)",
        "CHECK (jsonb_typeof(address) = 'object')",
        "CHECK (jsonb_typeof(support_preferences) = 'object')",
    ],
    "support.orders": [
        "PRIMARY KEY (order_id)",
        "UNIQUE (order_id, customer_id)",
        "FOREIGN KEY (customer_id) REFERENCES support.customers(customer_id)",
        "CHECK (char_length(order_id) >= 1 AND char_length(order_id) <= 128)",
        "CHECK (status = ANY (ARRAY['pending', 'paid', 'shipped', 'delivered', 'cancelled']))",
        "CHECK (currency = 'USD')",
        "CHECK (total_minor >= 0)",
        "CHECK (refunded_minor >= 0 AND refunded_minor <= total_minor)",
        "CHECK (jsonb_typeof(shipping_address) = 'object')",
    ],
    "support.order_items": [
        "PRIMARY KEY (order_id, line_number)",
        "FOREIGN KEY (order_id) REFERENCES support.orders(order_id)",
        "CHECK (line_number > 0)",
        "CHECK (quantity > 0)",
        "CHECK (unit_price_minor >= 0)",
    ],
    "support.refunds": [
        "PRIMARY KEY (refund_id)",
        "UNIQUE (session_id, invocation_id)",
        "FOREIGN KEY (order_id, customer_id) REFERENCES support.orders(order_id, customer_id)",
        "FOREIGN KEY (initiated_by_customer_id) REFERENCES support.customers(customer_id)",
        "CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$')",
        "CHECK (amount_minor > 0)",
        "CHECK (currency = 'USD')",
        "CHECK (char_length(reason) >= 1 AND char_length(reason) <= 2000)",
        "CHECK (refunded_total_after_minor >= amount_minor)",
    ],
    "knowledge.support_chunks": [
        "PRIMARY KEY (chunk_id)",
        "UNIQUE (document_id, chunk_index)",
        "CHECK (chunk_index >= 0)",
        "CHECK (char_length(content) > 0)",
        "CHECK (content_digest ~ '^sha256:[0-9a-f]{64}$')",
        "CHECK (jsonb_typeof(metadata) = 'object')",
        f"CHECK (embedding_model = '{EMBEDDING_MODEL}')",
        "CHECK (embedding_revision ~ '^[0-9a-f]{40}$')",
    ],
}
DEFAULTS = {
    ("support.customers", "support_preferences"): "'{}'::jsonb",
    ("support.customers", "created_at"): "now()",
    ("support.orders", "refunded_minor"): "0",
    ("support.orders", "refund_eligible"): "true",
    ("support.orders", "placed_at"): "now()",
    ("support.refunds", "created_at"): "now()",
    ("knowledge.support_chunks", "metadata"): "'{}'::jsonb",
}


def normalized(expression):
    # Normalize only outside string literals (regex braces and case are significant).
    parts = re.split(r"('(?:[^']|'')*')", expression or "")
    for i in range(0, len(parts), 2):
        value = re.sub(r"::(?:text|bigint|jsonb)", "", parts[i])
        parts[i] = re.sub(r"[\s()\"]", "", value).lower()
    return "".join(parts)


class SchemaError(RuntimeError):
    pass


def check_schema(connection, settings):
    problems, notices = [], []
    extension = connection.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if not extension or tuple(map(int, re.findall(r"\d+", extension[0])[:2])) < (0, 7):
        problems.append("pgvector extension version 0.7 or newer is required")
    for table, declaration in COLUMNS.items():
        rows = connection.execute(
            """
            SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull,
                   pg_get_expr(d.adbin, d.adrelid)
            FROM pg_attribute a
            LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
            WHERE a.attrelid = to_regclass(%s) AND a.attnum > 0 AND NOT a.attisdropped
        """,
            (table,),
        ).fetchall()
        columns = {row[0]: row[1:] for row in rows}
        if not columns:
            problems.append(f"Missing table {table}")
            continue
        for field in declaration.split():
            name, kind = field.split(":")
            nullable = kind.endswith("?")
            kind = kind.rstrip("?").replace("timestamptz", "timestamp with time zone")
            actual = columns.get(name)
            if actual is None or actual[0] != kind or actual[1] != (not nullable):
                problems.append(f"{table}.{name}: expected {kind}, nullable={nullable}")
            default = DEFAULTS.get((table, name))
            if default and actual and normalized(actual[2]) != normalized(default):
                if not (
                    default == "now()"
                    and normalized(actual[2]) in {"current_timestamp", "transaction_timestamp"}
                ):
                    problems.append(f"{table}.{name}: expected default {default}")
        constraints = connection.execute(
            """
            SELECT pg_get_constraintdef(oid) FROM pg_constraint
            WHERE conrelid = to_regclass(%s) AND convalidated AND NOT condeferrable
        """,
            (table,),
        ).fetchall()
        actual_constraints = {normalized(row[0]) for row in constraints}
        for requirement in CONSTRAINTS[table]:
            if normalized(requirement) not in actual_constraints:
                problems.append(f"{table}: missing/incompatible {requirement}")
        if connection.execute(
            "SELECT relrowsecurity FROM pg_class WHERE oid=to_regclass(%s)", (table,)
        ).fetchone()[0]:
            notices.append(
                f"{table}: row-level security enabled; it may restrict UNION experiments."
            )
    for table, keys in [
        ("support.orders", ["customer_id", "order_id"]),
        ("support.refunds", ["order_id", "created_at", "refund_id"]),
    ]:
        indexes = connection.execute(
            """
            SELECT array_agg(a.attname::text ORDER BY k.ordinality)
            FROM pg_index i CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY k(attnum, ordinality)
            JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.attnum
            WHERE i.indrelid=to_regclass(%s) AND i.indisvalid AND i.indpred IS NULL
              AND k.ordinality <= i.indnkeyatts
            GROUP BY i.indexrelid
        """,
            (table,),
        ).fetchall()
        if not any(row[0][: len(keys)] == keys for row in indexes):
            problems.append(f"{table}: missing index starting with {keys}")
    if problems:
        raise SchemaError(
            "Schema incompatible; apply/reconcile migrations explicitly:\n- "
            + "\n- ".join(problems)
        )
    for table, privileges in [
        ("support.customers", ["SELECT"]),
        ("support.orders", ["SELECT", "UPDATE"]),
        ("support.order_items", ["SELECT"]),
        ("support.refunds", ["SELECT", "INSERT"]),
        ("knowledge.support_chunks", ["SELECT"]),
    ]:
        for privilege in privileges:
            if not connection.execute(
                "SELECT has_table_privilege(%s, %s)", (table, privilege)
            ).fetchone()[0]:
                notices.append(
                    f"Database role lacks {privilege} on {table}; relevant tools will fail."
                )
    return notices
