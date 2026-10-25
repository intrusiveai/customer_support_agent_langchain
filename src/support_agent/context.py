from dataclasses import dataclass, field
from itertools import count
from threading import Lock
from uuid import uuid4

from .results import Rejected


@dataclass(frozen=True)
class ToolContext:
    authenticated_customer_id: str
    session_id: str
    interaction_id: str
    invocation_id: str
    evidence: object = field(repr=False)

    def record(self, event, **payload):
        return self.evidence.write(self, event, invocation_id=self.invocation_id, **payload)


@dataclass(frozen=True)
class RuntimeContext:
    authenticated_customer_id: str
    session_id: str
    interaction_id: str
    evidence: object = field(repr=False)
    tool_call_limit: int = 12
    # Mapping is private application state; no model-controlled invocation IDs.
    _calls: dict = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _call_numbers: object = field(default_factory=lambda: count(1), repr=False)
    uncertain_refunds: set = field(default_factory=set, repr=False)

    def consume_tool_call(self):
        with self._lock:
            if next(self._call_numbers) > self.tool_call_limit:
                raise Rejected(
                    "tool_call_limit", "The tool-call limit for this interaction has been reached."
                )

    def call(self, model_call_id):
        with self._lock:
            if model_call_id not in self._calls:
                self._calls[model_call_id] = ToolContext(
                    self.authenticated_customer_id,
                    self.session_id,
                    self.interaction_id,
                    str(uuid4()),
                    self.evidence,
                )
            return self._calls[model_call_id]
