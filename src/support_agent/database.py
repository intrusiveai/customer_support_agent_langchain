from contextlib import contextmanager
from time import monotonic
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .results import Rejected
from .schema import check_schema


def diagnostics(exc):
    # Never retain connection strings or arbitrary exception text.
    diag = getattr(exc, "diag", None)
    return {
        "type": type(exc).__name__,
        "sqlstate": getattr(exc, "sqlstate", None),
        "constraint": getattr(diag, "constraint_name", None),
        "table": getattr(diag, "table_name", None),
        "statement_position": getattr(diag, "statement_position", None),
    }


class Transaction:
    def __init__(self, connection, settings):
        self.connection, self.settings = connection, settings
        self.deadline = monotonic() + settings.transaction_timeout_ms / 1000

    def guard(self):
        remaining = int((self.deadline - monotonic()) * 1000)
        if remaining < 1:
            raise Rejected("transaction_timeout", "Transaction deadline exceeded.")
        self.connection.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(min(remaining, self.settings.query_timeout_ms)),),
        )

    def execute(self, sql, params=()):
        self.guard()
        return self.connection.cursor(row_factory=dict_row).execute(sql, params)

    def bounded(self, sql, params, limit):
        self.guard()
        # Server-side cursor prevents buffering an entire UNION result in the client.
        with self.connection.cursor(name=f"rows_{uuid4().hex}", row_factory=dict_row) as cursor:
            cursor.execute(sql, params)
            self.guard()
            rows = cursor.fetchmany(limit + 1)
        return rows[:limit], len(rows) > limit


class Database:
    def __init__(self, settings):
        self.settings = settings
        self.pool = ConnectionPool(
            settings.database_url.get_secret_value(),
            min_size=settings.pool_min,
            max_size=settings.pool_max,
            timeout=settings.pool_timeout_seconds,
            open=False,
            kwargs={
                "autocommit": True,
                "connect_timeout": max(1, int(settings.pool_timeout_seconds)),
                "options": f"-c timezone=UTC -c statement_timeout={settings.query_timeout_ms}",
            },
        )

    def start(self):
        try:
            self.pool.open(wait=True, timeout=self.settings.pool_timeout_seconds)
            with self.pool.connection() as conn:
                return check_schema(conn, self.settings)
        except Exception:
            self.close()
            raise

    def close(self):
        self.pool.close()

    @contextmanager
    def transaction(self, *, read_only=False):
        with self.pool.connection() as conn:
            with conn.transaction():
                if read_only:
                    conn.execute("SET TRANSACTION READ ONLY")
                conn.execute(
                    "SELECT set_config('idle_in_transaction_session_timeout', %s, true)",
                    (str(self.settings.transaction_timeout_ms),),
                )
                if conn.info.server_version >= 170000:
                    conn.execute(
                        "SELECT set_config('transaction_timeout', %s, true)",
                        (str(self.settings.transaction_timeout_ms),),
                    )
                tx = Transaction(conn, self.settings)
                yield tx
                tx.guard()
