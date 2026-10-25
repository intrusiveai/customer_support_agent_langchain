import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID


class Rejected(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


def result(outcome="ok", data=None, code=None, message=None):
    return {
        "outcome": outcome,
        "data": data,
        "error": {"code": code, "message": message} if code else None,
    }


def json_default(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (UUID, Decimal)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_text(value):
    return json.dumps(
        value, default=json_default, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def digest(content: bytes):
    return "sha256:" + hashlib.sha256(content).hexdigest()


def amount_minor(amount: str):
    if (
        not isinstance(amount, str)
        or len(amount) > 32
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", amount)
    ):
        raise Rejected(
            "invalid_amount", "Use a positive decimal USD string without signs or exponents."
        )
    # Integer arithmetic avoids Decimal context rounding on long inputs.
    whole, _, fraction = amount.partition(".")
    if any(c != "0" for c in fraction[2:]):
        raise Rejected("fractional_cent", "Fractional cents are unsupported.")
    cents = int(whole) * 100 + int((fraction[:2] + "00")[:2])
    if not 0 < cents <= 9223372036854775807:
        raise Rejected("invalid_amount", "Amount must be positive and fit a PostgreSQL bigint.")
    return cents


def usd(cents: int):
    return f"{cents // 100}.{cents % 100:02d}"
