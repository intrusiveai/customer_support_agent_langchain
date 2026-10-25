-- Explicit operator-run migration. Never applied by application startup.
BEGIN;
CREATE SCHEMA support;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE support.customers (
    customer_id text PRIMARY KEY,
    name text NOT NULL,
    email text NOT NULL,
    phone text,
    address jsonb NOT NULL,
    support_preferences jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT customers_id_length CHECK (char_length(customer_id) BETWEEN 1 AND 128),
    CONSTRAINT customers_address_object CHECK (jsonb_typeof(address) = 'object'),
    CONSTRAINT customers_preferences_object CHECK (jsonb_typeof(support_preferences) = 'object')
);
CREATE TABLE support.orders (
    order_id text PRIMARY KEY,
    customer_id text NOT NULL REFERENCES support.customers(customer_id),
    status text NOT NULL,
    currency text NOT NULL,
    total_minor bigint NOT NULL,
    refunded_minor bigint NOT NULL DEFAULT 0,
    refund_eligible boolean NOT NULL DEFAULT true,
    shipping_address jsonb NOT NULL,
    placed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (order_id, customer_id),
    CONSTRAINT orders_id_length CHECK (char_length(order_id) BETWEEN 1 AND 128),
    CONSTRAINT orders_status CHECK (status IN ('pending', 'paid', 'shipped', 'delivered', 'cancelled')),
    CONSTRAINT orders_currency CHECK (currency = 'USD'),
    CONSTRAINT orders_total CHECK (total_minor >= 0),
    CONSTRAINT orders_refunded CHECK (refunded_minor >= 0 AND refunded_minor <= total_minor),
    CONSTRAINT orders_address_object CHECK (jsonb_typeof(shipping_address) = 'object')
);
CREATE INDEX orders_customer_order_idx ON support.orders(customer_id, order_id);
CREATE TABLE support.order_items (
    order_id text NOT NULL REFERENCES support.orders(order_id),
    line_number integer NOT NULL,
    sku text NOT NULL,
    description text NOT NULL,
    quantity integer NOT NULL,
    unit_price_minor bigint NOT NULL,
    PRIMARY KEY (order_id, line_number),
    CONSTRAINT items_line_positive CHECK (line_number > 0),
    CONSTRAINT items_quantity_positive CHECK (quantity > 0),
    CONSTRAINT items_price_nonnegative CHECK (unit_price_minor >= 0)
);
CREATE TABLE support.refunds (
    refund_id uuid PRIMARY KEY,
    order_id text NOT NULL,
    customer_id text NOT NULL,
    initiated_by_customer_id text NOT NULL REFERENCES support.customers(customer_id),
    session_id text NOT NULL,
    invocation_id text NOT NULL,
    request_digest text NOT NULL,
    amount_minor bigint NOT NULL,
    currency text NOT NULL,
    reason text NOT NULL,
    refunded_total_after_minor bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (order_id, customer_id) REFERENCES support.orders(order_id, customer_id),
    UNIQUE (session_id, invocation_id),
    CONSTRAINT refunds_digest CHECK (request_digest ~ '^sha256:[0-9a-f]{64}$'),
    CONSTRAINT refunds_positive CHECK (amount_minor > 0),
    CONSTRAINT refunds_currency CHECK (currency = 'USD'),
    CONSTRAINT refunds_reason_length CHECK (char_length(reason) BETWEEN 1 AND 2000),
    CONSTRAINT refunds_total CHECK (refunded_total_after_minor >= amount_minor)
);
CREATE INDEX refunds_order_history_idx ON support.refunds(order_id, created_at, refund_id);
COMMIT;
