import os
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.conninfo import conninfo_to_dict

from support_agent.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, Settings
from support_agent.context import ToolContext
from support_agent.database import Database
from support_agent.evidence import Evidence
from support_agent.results import digest


class UnitEmbedder:
    def embed(self, query):
        vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        vector[0] = 1
        return vector


def pytest_addoption(parser):
    parser.addoption(
        "--database-url", default=None, help="Disposable PostgreSQL database ending in _test"
    )


@pytest.fixture
def embedding_server():
    """HTTP protocol fixture, not an embedding model. No external credentials or inference."""
    state = {"requests": [], "status": 200, "vector": UnitEmbedder().embed("").tolist()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            assert self.path == "/v1/embeddings"
            assert self.headers["Authorization"] == "Bearer embedding-test-key"
            assert request["model"] == EMBEDDING_MODEL
            assert request["dimensions"] == EMBEDDING_DIMENSIONS
            assert request["encoding_format"] == "float"
            body = {
                "object": "list",
                "model": EMBEDDING_MODEL,
                "data": [{"object": "embedding", "index": 0, "embedding": state["vector"]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            }
            if state["status"] != 200:
                body = {
                    "error": {
                        "message": "sensitive provider diagnostic",
                        "type": "test_error",
                        "code": "test_error",
                    }
                }
            encoded = json.dumps(body).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        database_url="postgresql://unused/unused",
        model_name="scripted-test",
        embedding_api_key=None,
        evidence_path=tmp_path / "evidence",
        result_limit=2,
    )


@pytest.fixture(scope="session")
def database_url(request):
    url = request.config.getoption("--database-url") or os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable database whose name ends in _test")
    if not conninfo_to_dict(url).get("dbname", "").endswith("_test"):
        pytest.fail(
            "TEST_DATABASE_URL database name must end in _test; fixtures erase support/knowledge schemas"
        )
    with psycopg.connect(url, autocommit=True) as connection:
        connection.execute("DROP SCHEMA IF EXISTS support CASCADE")
        connection.execute("DROP SCHEMA IF EXISTS knowledge CASCADE")
        for migration in sorted((Path(__file__).parents[1] / "migrations").glob("*.sql")):
            connection.execute(migration.read_text())
    return url


@pytest.fixture
def db_settings(settings, database_url):
    return Settings(**{**settings.model_dump(), "database_url": database_url})


@pytest.fixture
def database(db_settings):
    database = Database(db_settings)
    database.start()
    with database.pool.connection() as connection:
        connection.execute(
            "TRUNCATE support.refunds, support.order_items, support.orders, support.customers, knowledge.support_chunks CASCADE"
        )
        for customer in ["alice", "bob"]:
            connection.execute(
                """
                INSERT INTO support.customers(customer_id, name, email, address)
                VALUES (%s, %s, %s, '{"recipient":"Test","line1":"1 Main St","line2":"","city":"Town","region":"CA","postal_code":"90000","country_code":"US"}')
            """,
                (customer, customer.title(), customer + "@example.test"),
            )
            connection.execute(
                """
                INSERT INTO support.orders(order_id, customer_id, status, currency, total_minor, shipping_address)
                VALUES (%s,%s,'delivered','USD',10000,'{}')
            """,
                (customer + "-order", customer),
            )
            connection.execute(
                """INSERT INTO support.order_items VALUES (%s,1,'SKU','Test item',2,4000)""",
                (customer + "-order",),
            )
        register_vector(connection)
        content = (
            "Eligible delivered orders can receive a refund up to the remaining charged amount."
        )
        connection.execute(
            """
            INSERT INTO knowledge.support_chunks
            VALUES (%s,'refund-policy',0,'Refund policy','https://support.example.test/refunds',%s,%s,'{}',%s,%s,%s,now())
        """,
            (
                uuid4(),
                content,
                digest(content.encode()),
                EMBEDDING_MODEL,
                EMBEDDING_DIMENSIONS,
                UnitEmbedder().embed(content),
            ),
        )
    yield database
    database.close()


@pytest.fixture
def ctx(settings):
    return ToolContext(
        "alice", str(uuid4()), str(uuid4()), str(uuid4()), Evidence(settings.evidence_path)
    )
