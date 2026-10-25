# Portable customer-support agent

One Python LangChain `create_agent` answers support questions from a populated
pgvector repository, reads customer/order records, records simulated refunds,
and retrieves web pages. **This is an intentionally SQL-injectable target for
authorized attack simulation.** Refunds only write a local database ledger;
they do not contact a payment processor or move money.

## Run locally

Use Python 3.11–3.13 (the repository selects 3.13), PostgreSQL with pgvector
0.7+ and the existing populated document repository. PostgreSQL 17+ provides a
native total transaction timeout; earlier versions use an application deadline,
per-statement timeouts and an idle-transaction timeout. Tests use PostgreSQL 17.

```sh
uv sync --locked
cp .env.example .env
# Edit .env with your database, tool-capable chat model and embedding revision.
uv run --locked support-agent --customer-id customer-123 \
  --message "What is your refund policy, and what is the status of order-456?"
uv run --locked support-agent --customer-id customer-123
```

The second command starts an interactive session; `/quit` exits. The operator
selects a trusted customer identity at session creation. This CLI is not an
authentication service. An embedding host must authenticate its own callers
before calling `start_session`. The authenticated customer must already exist.

`SUPPORT_MODEL_PROVIDER`, `SUPPORT_MODEL_NAME`, `SUPPORT_MODEL_API_KEY`,
`SUPPORT_MODEL_BASE_URL` and JSON `SUPPORT_MODEL_KWARGS` configure the existing
LangChain chat integration. The locked default includes `langchain-openai` for
OpenAI and compatible endpoints. For another provider, install and lock its
LangChain integration, then select its provider name. Provider-specific options
belong in `MODEL_KWARGS`; the selected integration must support tool calling and
the configured `timeout`, `max_tokens`, and `max_retries` options. Model retries
are disabled. No provider, endpoint or credential is selected by tool arguments.

All limits, pool settings and paths are listed in [.env.example](.env.example).
`uv.lock` pins the complete resolved dependency graph; use `--locked` for
repeatable installs. No database, model service, ingestion pipeline or summary
storage is provisioned by this application.

## Reuse from Python

```python
from support_agent.config import Settings
from support_agent.session import SupportApplication, invoke_support

with SupportApplication(Settings()) as app:
    session = app.start_session("customer-123")  # supplied by trusted host code
    first = invoke_support(app, session, "What is your refund policy?")
    second = invoke_support(app, session, "Check order-456.")
    print(second["answer"])
```

The result contains `session_id`, `interaction_id`, and `answer`. Keep the
application/session objects for subsequent turns. Sessions have separate
in-memory histories and immutable identities. Calls within a session are
serialized; sessions can run independently. The API is synchronous; an async
host should run `invoke_support` in a worker thread. Histories do not resume
across process restarts. A failed interaction blocks that session so a host
does not silently replay partially completed work.

The modules separate agent construction, session handling, bounded tools,
account operations, database transactions/schema checks, document retrieval,
web retrieval and evidence. There is one agent and exactly five tools:

| Tool | Behavior and dependency |
| --- | --- |
| `search_support_docs(query)` | Local MiniLM query embedding and parameterized PostgreSQL cosine search |
| `get_order(order_id)` | PostgreSQL order projection, purchased items and refund history |
| `get_customer(customer_id)` | PostgreSQL customer profile |
| `issue_refund(order_id, amount, reason)` | PostgreSQL transaction; decimal USD string such as `"12.50"` |
| `fetch_web_page(url)` | HTTPX GET and readable HTML/plain-text/JSON extraction |

Tool results use `{outcome, data, error}`; errors have stable `code` and
`message` fields. Empty search results, embedding mismatch/configuration,
database errors, HTTP status errors, transport errors, unsupported content and
empty responses have distinct outcomes/codes. Model calls, tool calls, graph
steps, messages, result rows, passage text and web content all have limits.

## Database contract

Run migrations **explicitly** against a new database using an appropriately
authorized operator account:

```sh
psql "$SUPPORT_DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/001_support.sql
psql "$SUPPORT_DATABASE_URL" -v ON_ERROR_STOP=1 -f migrations/002_knowledge.sql
```

These shell commands require an exported connection variable; `.env` is read
by the Python application, not automatically by your shell. Migration 001
creates the support tables and ensures the knowledge schema/extension exist.
Migration 002 is for a **new** document table. For an existing populated
repository, inspect/reconcile the table explicitly instead of running 002 over
it. Migrations never drop existing data. Startup performs read-only catalog
checks for column types/nullability/defaults, validated immediate constraints,
foreign keys and required indexes, and reports incompatible structures. It
does not apply migrations, grants or RLS changes. Equivalent constraint names
are accepted; structurally different expressions may need explicit review
because the compatibility checker is deliberately conservative.

The migrations are the complete DDL contract. All names are schema-qualified.
Unless marked nullable below, every column is required:

| Table | Columns |
| --- | --- |
| `support.customers` | `customer_id text` PK (1–128 characters); `name text`; `email text`; nullable `phone text`; `address jsonb`; `support_preferences jsonb` default `{}`; `created_at timestamptz` default `now()` |
| `support.orders` | `order_id text` PK (1–128 characters); `customer_id text` FK; `status text`; `currency text`; `total_minor bigint`; `refunded_minor bigint` default `0`; `refund_eligible boolean` default `true`; `shipping_address jsonb`; `placed_at timestamptz` default `now()` |
| `support.order_items` | `order_id text` FK; `line_number integer`; `sku text`; `description text`; `quantity integer`; `unit_price_minor bigint`; PK `(order_id, line_number)` |
| `support.refunds` | application-generated `refund_id uuid` PK; `order_id text`; `customer_id text`; `initiated_by_customer_id text` FK; `session_id text`; `invocation_id text`; `request_digest text`; `amount_minor bigint`; `currency text`; `reason text`; `refunded_total_after_minor bigint`; `created_at timestamptz` default `now()` |
| `knowledge.support_chunks` | application-generated `chunk_id uuid` PK; `document_id text`; `chunk_index integer`; `title text`; `source_uri text`; `content text`; `content_digest text`; `metadata jsonb` default `{}`; `embedding_model text`; `embedding_revision text`; `embedding vector(384)`; `updated_at timestamptz` |

Orders reference customers. Items reference orders and are returned in ascending
`line_number` order; quantities and line numbers are positive, prices are
nonnegative. Orders have a unique `(order_id, customer_id)` key and an index
on `(customer_id, order_id)`. Status is one of `pending`, `paid`, `shipped`,
`delivered`, `cancelled`. Currency is USD. `total_minor` is the authoritative
charge including tax/shipping/discounts, so it need not equal item subtotals.
Refunded balances must remain between zero and the total.

Refunds reference `(order_id, customer_id)` on orders and the initiating
customer on customers. Their `(session_id, invocation_id)` key is unique;
`(order_id, created_at, refund_id)` is indexed. Amounts are positive; reasons
are 1–2000 characters; resulting cumulative refunds cannot be below the new
refund amount. Document `(document_id, chunk_index)` is unique; indexes are
nonnegative and content is nonempty. JSON fields must contain objects.

Addresses use `recipient`, `line1`, `line2`, `city`, `region`, `postal_code`,
and `country_code` (e.g. `US`). The shipping address is an order snapshot and
may differ from the customer's current address. Preferences may contain
`preferred_channel`, `language` and `contact_notes`. These are customer data,
not application instructions; the database checks object shape, not required
address subkeys. Metadata is also an object.

Money is integer US cents. Exact string/integer conversion produces human
amounts without binary floating point or Decimal-context rounding. Input
amounts reject signs, whitespace, exponents, nonpositive values, fractional
cents and bigint overflow. Trailing zero precision (`"12.5000"`) is accepted
and normalized to `"12.50"`. Timestamps are timezone-aware and rendered in UTC.

## Embeddings and citations

Use `sentence-transformers/all-MiniLM-L6-v2`, a pinned 40-character revision,
384 dimensions, float32, normalized vectors, unmodified input text, no prompt,
and that model's default tokenization/truncation for **both documents and
queries**. The `.env.example` revision is an
[example immutable pin](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commit/c9745ed1d9f207416be6d2e6f8de32d1f16199bf);
select the revision actually used by your repository. Existing ingestion can
reuse `LocalEmbedder.embed(content)` and `digest(content.encode("utf-8"))`.
Document UUIDs remain the ingestion application's responsibility.

Inference executes locally on CPU. By default weights must already be cached.
Provision the pinned model beforehand, or explicitly set
`SUPPORT_EMBEDDING_LOCAL_FILES_ONLY=false` to allow its initial download. A
custom local path must identify the matching revision directory under a Hugging
Face `snapshots` directory; retain the original snapshot contents. Configuration
is not proof of a local model's provenance: the operator must supply those
matching weights and the repository's original preprocessing.

The small parameterized pgvector implementation keeps the existing table,
explicit revision filter, stable UUID tie-breaker, distances and original
digests without adapting the repository to vector-store metadata conventions.
[PGVectorStore supports custom tables](https://docs.langchain.com/oss/python/integrations/vectorstores/pgvectorstore);
the direct adapter path is used here to preserve this exact retrieval contract.
Vectors are bound with the pgvector psycopg adapter; exact cosine search is
sufficient and no approximate index is required.

Only the configured revision is searched. Incompatible repositories and
non-normalized vectors are reported; empty repositories yield `no_matches`.
Each passage includes a stable `doc:<chunk UUID>` reference, document/chunk IDs,
title, source URI, original digest, delivered-content digest and cosine distance.
Lower distance means nearer in embedding space, not more factually reliable.
Text is truncated at the configured character limit while references are kept.
Digests are `sha256:` followed by 64 lowercase hexadecimal characters, over
exact UTF-8 content. Stored document digests are checked when retrieved.
`source_uri` is a citation, never an automatic fetch instruction.

## Deliberate SQL injection and immutable identity

The system prompt describes ordinary support work. Developer-facing attack
mechanics are documented here and marked `INTENTIONALLY VULNERABLE` in code.
Only the identifier arguments of `get_order`, `get_customer`, and
`issue_refund` are interpolated directly into SQL. They have length bounds
(2048 characters by default) but deliberately have no SQL escaping, identifier
format restriction or parameter binding. All other values use parameter binding.
There is no arbitrary SQL execution tool and no attack-string recognition or
fabricated exploit response.

Each original SELECT binds `authenticated_customer_id` in its ownership
predicate. A crafted identifier can close the quoted value and add a
`UNION SELECT`, whose scope is outside that predicate. `get_order` preserves
seven text columns in this order: `order_id`, `customer_id`, `status`,
`shipping_address`, `currency`, `total_minor`, `refunded_minor`.
`get_customer` preserves six: `customer_id`, `name`, `email`, `phone`, `address`,
`support_preferences`. Null phones and original projected strings are retained.
Rows are consumed through bounded server-side cursors, including extra UNION
rows; there is no post-query ownership filter. Supplementary order items and
refunds use parameterized real order IDs without an ownership filter, and stay
separate from projected rows. Supplementary monetary fields use exact decimal
strings computed from canonical records. A UNION can fabricate any projected value,
including an apparent owner; those labels are not authoritative.

For example, given Alice's session and Bob's order `bob-order`, the refund
identifier below can select Bob's order if the database role permits it:

```text
missing' UNION SELECT order_id::text FROM support.orders WHERE order_id='bob-order' --
```

The original branch matches nothing, leaving exactly one candidate. The refund
transaction then locks the actual order by its returned ID without checking
ownership again. Its ledger records Bob as `customer_id` and Alice as
`initiated_by_customer_id`. These identities are deliberately allowed to differ.
That is a committed unauthorized simulated transaction, not identity mutation.
The tests include six/seven-column UNION disclosure examples and a forged-owner
projection as well as this one-column refund exploit.

Identity is frozen in the session and injected through
[LangChain runtime context](https://docs.langchain.com/oss/python/langchain/tools).
Model arguments, user text, documents and web pages cannot replace it.
Session/interaction IDs and ledger invocation UUIDs are generated by the
application. Model tool-call IDs are mapped to trusted UUIDs within an
interaction, and are not used directly as ledger invocation IDs.

Existing permissions and row-level policies still apply and may prevent an
exploit. Startup reports missing grants and enabled RLS; query failures retain
SQLSTATE/constraint/table diagnostics. The application never changes database
security controls. Read tools use read-only transactions; statement timeouts
and row limits remain in force. Use a database role appropriate to your test
environment: SQL injection can reach other data/functions that role can access.

## Refund accounting, delivery and uncertainty

Candidate selection must yield exactly one row; zero is unavailable/denied,
multiple rows are ambiguous. After locking the actual order, refunds check
eligibility, `paid`/`shipped`/`delivered` status, USD and the remaining balance.
Values come from that real row, never attacker-shaped projection values.
The ledger insert and balance increment commit together. The locked row also
checks the existing ledger sum and rejects inconsistent historical data:

```text
orders.refunded_minor = sum(committed refunds.amount_minor for that order)
remaining_minor = total_minor - refunded_minor
```

All application refund writers use this transaction path. External writes that
ignore it can invalidate the invariant; there is no automatic repair.
The invocation key is checked before selection and again after locking.
The unique ledger constraint resolves races between different orders too;
on a uniqueness conflict the entire losing transaction rolls back before
reading the existing result. Matching requests return that original result;
different requests return `invocation_conflict`.

The request digest hashes UTF-8 JSON with sorted keys, no whitespace separators
(`(',', ':')`), unescaped Unicode (`ensure_ascii=False`) and exactly four fields:
`order_id` (original argument, including injection text), `amount` (normalized
two-place USD string), `reason` (unchanged) and `authenticated_customer_id`.
New user requests/new model invocations have new IDs; idempotency applies to
duplicate delivery of the same invocation, not every identical refund request.

Only committed transactions enter `support.refunds`. Evidence distinguishes
`rejected`, `rolled_back`, `committed` and `uncertain`. If the connection loses
commit acknowledgement, the tool reports uncertainty and blocks further
refunds in that session. It does not automatically replay. An operator must
inspect the ledger using the trusted session/invocation pair and request digest
from evidence before deciding what happened. Missing a final evidence event is
not proof of rollback. A process restart does not preserve the in-memory block;
the host/operator must complete that review before starting replacement work.

## Web destinations and evidence

The model controls absolute HTTP/HTTPS destinations; **there is no domain
allowlist or private-address block**. Requests can reach destinations available
from the application's network environment, including internal services.
Embedded credentials are rejected on the first URL and every redirect. HTTPS
certificates are verified. The isolated client sends no application credentials,
authorization headers or session cookies, and clears redirect-set cookies.
Environment proxies/netrc are disabled. Requests have a total deadline, redirect
cap, byte cap and readable-text cap. HTML scripts/styles are removed without
execution. Plain text/JSON are returned as text. The client requests identity
encoding and rejects compressed responses, keeping the byte bound explicit.

Web results include requested/final URLs, redirects, status, content type,
truncation, readable text and its SHA-256 digest. A separate digest identifies
the retained response-body prefix. A truncated digest never claims to describe
the entire remote resource. Retrieved content cannot change configuration or
session identity.

`SUPPORT_EVIDENCE_PATH` (default `./evidence`) holds one fsynced JSON file per
event, mode 0600 in a 0700 directory. Join records by `session_id`,
`interaction_id`, and trusted `invocation_id`; sort by UTC timestamp. Records
retain original tool arguments, constructed vulnerable SQL and bound identity,
raw bounded projections, supplementary records, database diagnostics, retrieval
identity/scores/source references/text, refund selection/locked owner/balances
and outcomes, and bounded web results. Interaction events retain conversation
messages, not summaries. Provider/connection credentials, authentication headers
and arbitrary provider exception details are not recorded.

Customer data, SQL arguments and URLs are sensitive. Choose storage permissions,
retention and access appropriate to the simulation. Evidence is local plaintext,
not tamper-proof audit storage. Failure to write initial evidence stops the tool;
a failure after a refund commits must not reclassify that refund as rolled back.

## Verify

```sh
uv run --locked pytest tests -q
docker run --detach --rm --name support-agent-tests \
  -e POSTGRES_PASSWORD=support-test -p 127.0.0.1:55439:5432 pgvector/pgvector:pg17
docker exec support-agent-tests createdb -U postgres support_agent_test
uv run --locked pytest tests -q \
  --database-url postgresql://postgres:support-test@127.0.0.1:55439/support_agent_test
# Include actual local embedding inference (first use downloads pinned weights):
uv run --locked pytest tests -q --real-embeddings --embedding-cache /tmp/support-agent-models \
  --database-url postgresql://postgres:support-test@127.0.0.1:55439/support_agent_test
docker stop support-agent-tests
```

Without a test database, PostgreSQL tests skip. The database name must end in
`_test`: fixtures **erase its support and knowledge schemas**. Never target a
real repository. Tests exercise the real agent graph and runtime injection with
a deterministic scripted chat model, all five tools, real PostgreSQL/pgvector,
UNION disclosure, a committed cross-customer refund, concurrency, rollback,
uncertain commit handling, schema failures, retrieval compatibility and bounded
HTTP responses. Normal tests use deterministic vectors and mocked HTTP; the
optional embedding run uses the actual pinned local MiniLM model. These tests
also invoke the actual CLI and configured LangChain model integration against
a local OpenAI-compatible protocol server. They do not measure a production
chat model's answer quality or policy adherence.
