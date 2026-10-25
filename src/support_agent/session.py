from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from langchain_core.messages import HumanMessage

from .agent import build_agent
from .context import RuntimeContext
from .database import Database
from .evidence import Evidence
from .results import Rejected


@dataclass(frozen=True)
class Session:
    authenticated_customer_id: str
    session_id: str = field(default_factory=lambda: str(uuid4()))
    _messages: list = field(default_factory=list, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _uncertain_refunds: set = field(default_factory=set, repr=False)
    _failed_interactions: set = field(default_factory=set, repr=False)


class SupportApplication:
    def __init__(self, settings, *, model=None, embedder=None, web_transport=None):
        self.settings = settings
        self.evidence = Evidence(settings.evidence_path)
        self.database = Database(settings)
        self.notices = self.database.start()
        self._sessions = {}
        try:
            self.agent = build_agent(settings, self.database, model, embedder, web_transport)
        except Exception:
            self.close()
            raise

    def start_session(self, authenticated_customer_id: str):
        """The host authenticates first; this trusted API is never exposed as a tool."""
        if (
            not isinstance(authenticated_customer_id, str)
            or not 1 <= len(authenticated_customer_id) <= 128
        ):
            raise Rejected(
                "invalid_identity", "Authenticated customer IDs must contain 1–128 characters."
            )
        with self.database.transaction(read_only=True) as tx:
            row = tx.execute(
                "SELECT customer_id FROM support.customers WHERE customer_id = %s",
                (authenticated_customer_id,),
            ).fetchone()
        if not row:
            raise Rejected(
                "unknown_identity", "Authenticated customer does not exist or is inaccessible."
            )
        session = Session(authenticated_customer_id)
        self._sessions[session.session_id] = session
        return session

    def invoke(self, session, message):
        if self._sessions.get(session.session_id) is not session:
            raise Rejected("unknown_session", "Session was not established by this application.")
        if (
            not isinstance(message, str)
            or not 1 <= len(message) <= self.settings.user_message_char_limit
        ):
            raise Rejected(
                "invalid_message", "Message is empty or exceeds the configured length limit."
            )
        with session._lock:
            if session._failed_interactions:
                raise Rejected(
                    "session_requires_review",
                    "A prior interaction failed. Inspect its evidence before starting a new session; do not replay refunds.",
                )
            if len(session._messages) >= self.settings.session_message_limit:
                raise Rejected("session_limit", "Conversation limit reached; start a new session.")
            context = RuntimeContext(
                session.authenticated_customer_id,
                session.session_id,
                str(uuid4()),
                self.evidence,
                tool_call_limit=self.settings.agent_tool_call_limit,
                uncertain_refunds=session._uncertain_refunds,
            )
            self.evidence.write(context, "interaction_started", user_message=message)
            try:
                state = self.agent.invoke(
                    {"messages": [*session._messages, HumanMessage(content=message)]},
                    context=context,
                    config={
                        "recursion_limit": self.settings.agent_recursion_limit,
                        "max_concurrency": 1,
                    },
                )
                messages = state["messages"]
                self.evidence.write(
                    context,
                    "interaction_finished",
                    messages=[
                        {
                            "type": m.type,
                            "content": m.content,
                            "tool_calls": getattr(m, "tool_calls", []),
                            "tool_call_id": getattr(m, "tool_call_id", None),
                        }
                        for m in messages[len(session._messages) :]
                    ],
                )
                session._messages[:] = messages
                return {
                    "session_id": session.session_id,
                    "interaction_id": context.interaction_id,
                    "answer": messages[-1].content,
                }
            except Exception as exc:
                session._failed_interactions.add(context.interaction_id)
                self.evidence.write(
                    context,
                    "interaction_failed",
                    error_type=type(exc).__name__,
                    message="Inspect tool outcomes before attempting another interaction.",
                )
                raise

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def invoke_support(application: SupportApplication, session: Session, message: str):
    """Reusable invocation; retain the same session object for subsequent turns."""
    return application.invoke(session, message)
